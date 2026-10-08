"""Run the fixed historical G1 protocol only with an attended, pinned review.

The review must include ``full_checkpoints`` with keys ``"0"``, ``"1"`` and
``"2"``, each containing a checkpoint ``path`` and its ``sha256``. Relative
paths in the review, including ``prereg_path``, are repository-relative.
The historical gate at ``rl/gate.json`` remains a required runtime pin; this
runner does not supply a gate, model, or preregistration for a new experiment.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SEM = {"CELLO_ENVELOPE_MIN_S": "0.5", "CELLO_PHRASE_SCORING": "1", "CELLO_TIMING_PENALTY": "1"}
ALT2 = {**SEM, "CELLO_ALT_LOOP": "1", "CELLO_ZERO_INIT": "1",
        "CELLO_TARGET_ENTROPY_SCALE": "0.5", "CELLO_ENT_COEF_INIT": "0.005"}
ORDERS = (("Base", "FULL", "Base", "OFF", "Base"),
          ("Base", "OFF", "Base", "FULL", "Base"))
REQUIRED_RUNTIME_FILES = frozenset({
    "rl/guard_config.py", "rl/piece_env.py", "rl/train_piece.py", "rl/train_piece_logged.py",
    "rl/perform.py", "rl/perform_audit.py", "rl/piece_hardware.py", "rl/g1_session.py",
    "rl/train_supervisor.py", "rl/ckpt_entropy.py", "rl/g1_analyze.py", "vocalise.mxl",
    "rl/posthoc_score.py", "rl/gate.json", "rl/loudness.py", "rl/timing_model.json",
    "BaselineControls/play_midi_pieces.py", "SoundClassifier/classifier.py", "SoundClassifier/audio_features.py",
    "SoundClassifier/checkpoints/loudness_model.json",
    "SoundClassifier/quality_classifier.py", "SoundClassifier/dataset.py",
    "SoundClassifier/checkpoints/quality_cnn_multilength.pt",
    "SoundClassifier/checkpoints/quality_cnn_human_A1_A5.pt",
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def review_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("review input path must be a nonempty string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def reviewed_full_checkpoints(review):
    entries = review.get("full_checkpoints")
    if not isinstance(entries, dict) or set(entries) != {"0", "1", "2"}:
        raise ValueError("review must pin FULL checkpoint path and sha256 for seeds 0, 1, and 2")
    checkpoints = {}
    for seed in range(3):
        entry = entries[str(seed)]
        if not isinstance(entry, dict) or not re.fullmatch(r"[0-9a-fA-F]{64}", str(entry.get("sha256", ""))):
            raise ValueError(f"invalid FULL checkpoint SHA-256 for seed {seed}")
        path = review_path(entry.get("path"))
        if not path.is_file():
            raise ValueError(f"missing reviewed FULL checkpoint for seed {seed}: {path}")
        if digest(path) != entry["sha256"].lower():
            raise ValueError(f"reviewed FULL checkpoint changed for seed {seed}: {path}")
        checkpoints[seed] = path
    return checkpoints


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line] if Path(path).exists() else []


def child_environment(training=False, arm="Base"):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CELLO_")}
    environment.update(ALT2 if training else SEM)
    environment.update(CELLO_GUARD_MODE="off" if arm == "OFF" else "full", CELLO_W_QUALITY="0.50")
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return environment


def training_command(seed, output):
    return [sys.executable, "-u", str(ROOT / "rl/train_supervisor.py"), "--max-restarts", "0", "--",
            str(ROOT / "vocalise.mxl"), "--real", "--timesteps", "1400", "--ckpt-every", "10",
            "--save-stroke-audio", "--no-calibrated-dynamics", "--reference-every", "5", "--seed", str(seed),
            "--save-dir", str(output)]


def evaluation_command(seed, arm, checkpoint=None):
    command = [sys.executable, "-u", str(ROOT / "rl/perform.py")]
    if arm != "Base":
        command.append(str(checkpoint))
    command += [str(ROOT / "vocalise.mxl"), "--real", "--tempo-scale", "1.0", "--seed", str(seed)]
    command += ["--baseline"] if arm == "Base" else ["--policy-input-mode", "normal"]
    return command


def mapped_path(output, prefix):
    matches = re.findall(r"^" + re.escape(prefix) + r"(.+?)\s*$", output, flags=re.MULTILINE)
    paths = list(dict.fromkeys(matches))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {prefix!r} path, found {paths}")
    path = Path(paths[0])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_take(directory, arm):
    import numpy as np
    import soundfile as sf
    directory = Path(directory)
    performance_files = list(directory.glob("*_performance.json"))
    if len(performance_files) != 1:
        raise ValueError("expected exactly one performance file")
    performance = read_json(performance_files[0])
    timeline = performance["timeline"]
    if len(timeline) != 14:
        raise ValueError("incomplete Vocalise take")
    for note in timeline:
        if not all(np.isfinite(float(note[field])) for field in ("actual_onset", "actual_duration")) or note["actual_duration"] <= 0:
            raise ValueError("invalid timing")
    stem = str(performance_files[0]).removesuffix("_performance.json")
    audio, sample_rate = sf.read(stem + ".wav", always_2d=True)
    if sample_rate != 44100 or audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("invalid audio format or samples")
    if not np.any(audio):
        raise ValueError("all-zero recording")
    if (float(performance.get("audio_lead_in_s", 0)) + max(float(note["actual_onset"]) + float(note["actual_duration"]) for note in timeline)) > len(audio) / sample_rate + .010:
        raise ValueError("recording does not cover the full take")
    state = np.load(stem + "_state.npy", allow_pickle=True)
    times = np.asarray([row["t"] for row in state], dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("invalid state timeline")
    audit = read_json(directory / "ep01_strokes.json")
    if audit["status"] != "complete":
        raise ValueError("audit is incomplete")
    outcomes = [row for row in audit["records"] if row.get("event") == "result"]
    if len(outcomes) != 14 or any(row.get("truncated") or row.get("info", {}).get("truncated_by_disconnect") for row in outcomes):
        raise ValueError("incomplete or disconnected audit")
    snapshot = read_json(directory / "runtime_snapshot.json")
    expected = "off" if arm == "OFF" else "full"
    if snapshot["guard_reward"]["mode"] != expected or snapshot["argv"].get("mock") or snapshot["scoring_mode"] != "synchronous":
        raise ValueError("wrong guard mode or mock/async take")
    return {str(path.relative_to(directory)): digest(path) for path in directory.iterdir() if path.is_file()}


class Session:
    def __init__(self, options):
        self.options = options
        self.directory = Path(options.session).resolve()
        self.journal = self.directory / "journal.jsonl"
        self.control = Path(options.control_dir).resolve() if options.control_dir else self.directory
        self.review = None
        self.full_checkpoints = {}

    def append(self, record):
        with self.journal.open("a") as stream:
            stream.write(json.dumps({"unix_s": time.time(), **record}, allow_nan=False) + "\n")

    def check_controls(self):
        while True:
            if (self.control / "STOP_CHAIN").exists():
                raise RuntimeError("STOP_CHAIN: halt between commands, not an emergency stop")
            if not (self.control / "PAUSE_CHAIN").exists():
                return
            time.sleep(2)

    def verify_review(self):
        review = read_json(self.options.review)
        if self.review is not None and review != self.review:
            raise ValueError("session review changed; do not silently refresh pins")
        if review.get("status") != "G1_REVIEW_PASS" or not self.options.attended:
            raise ValueError("requires a passed review and an attended lab session")
        if review.get("guard_off_nonzero_terms_removed") != 9 or review.get("historical_full_seed2_timesteps") != 1318:
            raise ValueError("review must acknowledge the nine-term definition and the historical seed-2 budget mismatch")
        runtime_files = review.get("runtime_files")
        if not isinstance(runtime_files, dict) or not REQUIRED_RUNTIME_FILES.issubset(runtime_files):
            raise ValueError("review is missing required source/model/gate pins")
        for relative, expected in runtime_files.items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"reviewed runtime/input changed: {relative}")
        if digest(review_path(review["prereg_path"])) != review["prereg_sha256"]:
            raise ValueError("preregistration changed")
        self.full_checkpoints = reviewed_full_checkpoints(review)
        self.review = review

    def launch(self, identifier, command, environment):
        self.check_controls()
        self.verify_review()
        previous = [row for row in records(self.journal) if row.get("id") == identifier]
        if previous:
            finished = [row for row in previous if row.get("status") == "valid"]
            if len(finished) == 1:
                return finished[0]
            raise RuntimeError(f"{identifier} incomplete/failed; no automatic retry, preserve evidence and amend before replacement")
        log = self.directory / (identifier + ".log")
        self.append({"id": identifier, "status": "started", "argv": command,
                     "environment": {key: value for key, value in environment.items() if key.startswith("CELLO_")}})
        print(f"G1 START {identifier}: guard={environment.get('CELLO_GUARD_MODE')} log={log}", flush=True)
        with log.open("x") as stream:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        output = log.read_text()
        self.append({"id": identifier, "status": "process_exited", "exit": process.returncode, "log": str(log)})
        print(f"G1 EXIT {identifier}: {process.returncode}", flush=True)
        if process.returncode:
            raise RuntimeError(f"{identifier} failed with exit {process.returncode}; chain halted")
        return {"id": identifier, "output": output}

    def train(self, seed):
        identifier = f"train_OFF_s{seed}"
        result = self.launch(identifier, training_command(seed, self.directory / "training" / f"seed{seed}"), child_environment(True, "OFF"))
        if result.get("status") == "valid":
            if digest(result["checkpoint"]) != result["checkpoint_sha256"]:
                raise ValueError("recorded OFF checkpoint changed")
            self.entropy(seed, result["run"])
            return result
        run = mapped_path(result["output"], "run dir -> ")
        if not run.is_relative_to(self.directory / "training" / f"seed{seed}"):
            raise ValueError("trainer directory escaped the assigned seed root")
        checkpoint = run / "sac_piece_vocalise_final.zip"
        with zipfile.ZipFile(checkpoint) as archive:
            checkpoint_data = json.loads(archive.read("data"))
        if checkpoint_data["num_timesteps"] != 1400 or checkpoint_data["seed"] != seed:
            raise ValueError("FINAL checkpoint budget/seed differs")
        log = records(run / "stroke_log.jsonl")
        strokes = [row for row in log if "stroke_seq" in row]
        if len(strokes) != 1400 or not log or log[0]["guard_reward"]["mode"] != "off":
            raise ValueError("wrong training budget or reward mode")
        snapshot = read_json(run / "runtime_snapshot.json")
        if snapshot["guard_reward"] != log[0]["guard_reward"] or snapshot["argv"].get("real") is not True:
            raise ValueError("training snapshot/header mismatch or not real training")
        record = {"id": identifier, "status": "valid", "seed": seed, "arm": "OFF", "run": str(run),
                  "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "learned_strokes": len(strokes)}
        self.append(record)
        with (self.directory / "training_runs.tsv").open("a") as stream:
            stream.write(f"OFF\t{seed}\t{run}\t{record['checkpoint_sha256']}\n")
        self.entropy(seed, run)
        return record

    def entropy(self, seed, run):
        result = self.launch(f"entropy_s{seed}", [sys.executable, str(ROOT / "rl/ckpt_entropy.py"), str(run)], child_environment())
        if result.get("status") != "valid":
            self.append({"id": f"entropy_s{seed}", "status": "valid"})

    def evaluate(self, seed):
        self.verify_review()
        trained = [row for row in records(self.journal) if row.get("id") == f"train_OFF_s{seed}" and row.get("status") == "valid"]
        if len(trained) != 1 or digest(trained[0]["checkpoint"]) != trained[0]["checkpoint_sha256"]:
            raise ValueError(f"missing or changed OFF FINAL seed {seed}")
        checkpoints = {"OFF": trained[0]["checkpoint"], "FULL": str(self.full_checkpoints[seed])}
        for order_index, order in enumerate(ORDERS):
            for slot, arm in enumerate(order):
                identifier = f"eval_s{seed}_o{order_index}_t{slot}_{arm}"
                result = self.launch(identifier, evaluation_command(seed, arm, checkpoints.get(arm)), child_environment(arm=arm))
                if result.get("status") == "valid":
                    continue
                directory = mapped_path(result["output"], "perform -> ")
                if not directory.is_relative_to(ROOT / "rl/checkpoints_piece"):
                    raise ValueError("take path escaped repository recording directory")
                hashes = validate_take(directory, arm)
                snapshot = read_json(directory / "runtime_snapshot.json")
                if arm != "Base" and snapshot["policy_checkpoint"]["sha256"] != digest(checkpoints[arm]):
                    raise ValueError("take used the wrong checkpoint")
                if snapshot["scorer_checkpoint"]["sha256"] != self.review["runtime_files"]["SoundClassifier/checkpoints/quality_cnn_multilength.pt"]:
                    raise ValueError("take used the wrong live classifier")
                record = {"id": identifier, "status": "valid", "seed": seed, "order": order_index, "slot": slot,
                          "arm": arm, "take_dir": str(directory), "raw_hashes": hashes,
                          "checkpoint_sha256": digest(checkpoints[arm]) if arm != "Base" else None}
                self.append(record)
                with (self.directory / "takes.tsv").open("a") as stream:
                    stream.write(f"{seed}\t{order_index}\t{slot}\t{arm}\t{directory}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval", "all"))
    parser.add_argument("--session", required=True)
    parser.add_argument("--control-dir")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--attended", action="store_true")
    parser.add_argument("--review", help="review JSON with runtime pins (including rl/gate.json), preregistration, and full_checkpoints")
    options = parser.parse_args()
    if not options.execute:
        full_checkpoints = reviewed_full_checkpoints(read_json(options.review)) if options.review else {}
        for seed in range(3):
            if options.mode in ("train", "all"):
                print(json.dumps({"seed": seed, "env": ALT2, "guard_mode": "off", "argv": training_command(seed, Path(options.session) / "training" / f"seed{seed}")}))
            if options.mode in ("eval", "all"):
                for order_index, order in enumerate(ORDERS):
                    for slot, arm in enumerate(order):
                        checkpoint = full_checkpoints.get(seed, f"<review full_checkpoints[{seed}]>") if arm == "FULL" else "<locked OFF FINAL>"
                        print(json.dumps({"seed": seed, "order": order_index, "slot": slot, "arm": arm,
                                          "guard_mode": "off" if arm == "OFF" else "full", "argv": evaluation_command(seed, arm, checkpoint)}))
        return
    if not options.review:
        parser.error("--execute requires --review and --attended; dry-run is the default")
    session = Session(options)
    session.verify_review()
    session.directory.mkdir(parents=True, exist_ok=True)
    review_copy = session.directory / "review_lock.json"
    if review_copy.exists():
        if read_json(review_copy) != session.review:
            raise ValueError("session review lock changed; do not silently refresh pins")
    else:
        with review_copy.open("x") as stream:
            json.dump(session.review, stream, indent=2)
    lock = session.directory / "RUNNING.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        if options.mode in ("train", "all"):
            for seed in range(3):
                session.train(seed)
        if options.mode in ("eval", "all"):
            for seed in range(3):
                session.evaluate(seed)
            result = session.launch("dual_judge", [sys.executable, str(ROOT / "rl/g1_analyze.py"), "--session", str(session.directory)], child_environment())
            if result.get("status") != "valid":
                session.append({"id": "dual_judge", "status": "valid"})
        print("G1 requested stages complete; public production-quality scoring remains offline follow-up")
    except BaseException as error:
        session.append({"status": "chain_aborted", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        os.close(descriptor)
        lock.unlink()


if __name__ == "__main__":
    main()
