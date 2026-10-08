"""Validation and evidence helpers for policy inputs and performance records.

Most helpers use only NumPy and the standard library. ``write_snapshot`` imports
``rl.piece_env`` to obtain the effective reward configuration, so that operation
also requires the environment's planning, audio and optional driver dependencies.
Importing this module alone does not initialize the environment.
"""

import dataclasses
import hashlib
import json
import math
import os
import sys
import importlib
import time
from pathlib import Path

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [json_value(item) for item in value]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=repr)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def file_record(path, include_text=False):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    record = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    if include_text:
        record["text"] = raw.decode("utf-8")
    return record


def checkpoint_file(path):
    path = Path(path)
    if not path.is_file() and path.suffix != ".zip":
        path = Path(str(path) + ".zip")
    return path.resolve(strict=True)


def finite_array(value, shape, label, bounded=False):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.shape != tuple(shape):
        raise ValueError(f"{label}: expected numeric shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite value")
    if bounded and ((array < -1).any() or (array > 1).any()):
        raise ValueError(f"{label}: value outside [-1, 1]")
    result = array.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{label}: value overflows float32")
    return result


def validate_mode(args, observation_shape, action_shape):
    frozen = args.policy_input_mode == "frozen_nominal"
    c1_mode = frozen or args.c1_cohort is not None
    if frozen and (args.baseline or args.replay_actions or not args.checkpoint):
        raise ValueError("frozen_nominal requires a policy, not baseline or replay")
    if c1_mode and (args.compile or args.async_score):
        raise ValueError("C1 requires the live synchronous path")
    if c1_mode and (tuple(observation_shape) != (22,) or tuple(action_shape) != (6,)):
        raise ValueError("C1 requires the pinned 22D observation / 6D action lineage")
    if frozen and not all((args.frozen_references, args.frozen_reference_id, args.c1_cohort)):
        raise ValueError("frozen_nominal requires references, reference ID and C1 cohort")
    if not frozen and (args.frozen_references or args.frozen_reference_id):
        raise ValueError("frozen-reference options require frozen_nominal")
    if args.replay_preflight_hash and not args.replay_actions:
        raise ValueError("replay hash requires --replay-actions")


def load_replay(path, piece, action_dim, expected_hash=None):
    record = file_record(path)
    if expected_hash and record["sha256"] != expected_hash:
        raise ValueError("replay file hash mismatch")
    document = json.loads(Path(record["path"]).read_text())
    score = file_record(piece)
    if document.get("piece_sha256") != score["sha256"]:
        raise ValueError("replay requires the actual score's piece_sha256")
    values = np.asarray(document["actions"])
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("replay requires a nonempty N x action_dim array")
    actions = finite_array(values, (len(values), action_dim), "replay", bounded=True)
    record.update(piece=score, count=len(actions), source_mode=document.get("mode"))
    return actions, record


def load_scoring_approval(path):
    if not path:
        raise ValueError("C1 execution requires an explicitly approved scoring configuration")
    manifest = file_record(path, include_text=True)
    document = json.loads(manifest["text"])
    if document.get("status") != "approved":
        raise ValueError("scoring configuration is not approved")
    for field in ("author", "offline_pipeline", "window_alignment"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"scoring approval requires {field}")
    if document.get("live_mode") != "synchronous":
        raise ValueError("C1 scoring approval must specify synchronous live scoring")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("scoring approval requires pinned source/checkpoint artifacts")
    for artifact in artifacts:
        if file_record(artifact["path"])["sha256"] != artifact["sha256"]:
            raise ValueError("approved scoring artifact changed")
    return {"manifest": manifest, "configuration": document}


def bind_cohort(path, piece, checkpoint=None):
    manifest = file_record(path, include_text=True)
    document = json.loads(manifest["text"])
    if document.get("version") != "C1.plan.r4":
        raise ValueError("C1 cohort must use the r4 version")
    score = file_record(piece)
    matches = [entry for entry in document["pieces"]
               if entry["score"]["sha256"] == score["sha256"]]
    if len(matches) != 1:
        raise ValueError("score hash does not identify exactly one cohort piece")
    entry = matches[0]
    selected = None
    if checkpoint is not None:
        identity = file_record(checkpoint_file(checkpoint))
        matches = [item for item in entry["checkpoints"] if item["sha256"] == identity["sha256"]]
        if len(matches) != 1:
            raise ValueError("checkpoint is not pinned for this piece")
        selected = dict(matches[0], actual_path=identity["path"])
    return {"manifest": manifest, "piece": entry["piece"], "score": score,
            "expected_strokes": entry["expected_strokes"], "checkpoint": selected}


def load_reference(path, reference_id, binding, alpha):
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("EMA alpha must be finite in [0, 1)")
    manifest = file_record(path, include_text=True)
    document = json.loads(manifest["text"])
    if (document.get("version") != "C1.plan.r4"
            or document.get("mode") != "frozen_nominal"
            or document.get("initial_pair") != [0.5, 0.5]
            or document.get("policy_observation_indices") != [13, 14]):
        raise ValueError("unsupported frozen-reference schema")
    matches = [entry for entry in document["references"] if entry["reference_id"] == reference_id]
    if len(matches) != 1 or binding["checkpoint"] is None:
        raise ValueError("reference ID must uniquely bind a policy")
    reference = matches[0]
    if (reference["piece"] != binding["piece"]
            or reference["seed"] != binding["checkpoint"]["seed"]
            or reference["checkpoint_sha256"] != binding["checkpoint"]["sha256"]):
        raise ValueError("reference piece/seed/checkpoint mismatch")
    constant = reference["constant"]
    if isinstance(constant, bool) or not isinstance(constant, (int, float)):
        raise ValueError("reference constant must be numeric")
    if not math.isfinite(constant) or not 0 <= constant <= 1:
        raise ValueError("reference constant must be finite in [0, 1]")
    means = []
    takes = set()
    for source in reference["sources"]:
        actual = file_record(source["path"])
        if actual["sha256"] != source["sha256"] or source["take"] in takes:
            raise ValueError("reference source hash mismatch or duplicate take")
        takes.add(source["take"])
        scores = json.loads(Path(source["path"]).read_text())["scores"]
        if len(scores) != source["n"] or source["n"] != binding["expected_strokes"]:
            raise ValueError("reference source stroke-count mismatch")
        values = [row["overall"] for row in scores]
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   and 0 <= value <= 1 for value in values):
            raise ValueError("invalid source overall score")
        means.append(sum(values) / len(values))
    if len(means) != 3 or abs(sum(means) / 3 - constant) > 1e-12:
        raise ValueError("constant is not the mean of the three pinned take means")
    return {"reference_id": reference_id, "constant": float(constant),
            "alpha": float(alpha), "checkpoint_sha256": reference["checkpoint_sha256"],
            "piece": reference["piece"], "seed": reference["seed"], "manifest": manifest}


class PolicyInput:
    def __init__(self, shape, reference=None):
        self.shape = tuple(shape)
        self.reference = reference
        self.completed = 0
        self.ema = 0.5

    def deliver(self, observation):
        original = finite_array(observation, self.shape, "observation")
        delivered = original.copy()
        if self.reference:
            if self.completed == 0:
                if not np.array_equal(original[13:15], np.array([0.5, 0.5], dtype=np.float32)):
                    raise ValueError("frozen episode must reset auditory inputs to (0.5, 0.5)")
            else:
                delivered[13] = self.reference["constant"]
                delivered[14] = self.ema
        return original.copy(), delivered

    def advance(self, completed):
        if completed:
            self.completed += 1
            if self.reference:
                alpha = self.reference["alpha"]
                self.ema = alpha * self.ema + (1 - alpha) * self.reference["constant"]


def new_take_dir(root, now=None, retries=5):
    """Allocate rl/checkpoints_piece/perform_YYYYMMDD_HHMMSS exclusively.

    Keeps the second-resolution name that the chain logs and the offline tools
    parse (`perform -> .*/(perform_\\d+_\\d+)`); on a same-second collision
    waits for the next second instead of appending a suffix."""
    import datetime as _dt
    root = Path(root)
    for _ in range(retries):
        stamp = (now or _dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
        path = root / f"perform_{stamp}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            now = None
            time.sleep(max(0.0, 1.0 - _dt.datetime.now().microsecond / 1e6) + 0.002)
    raise RuntimeError("could not allocate a unique take directory")


def require_complete(terminated, truncated, info):
    if truncated or info.get("truncated_by_disconnect") or (info.get("stroke") or {}).get("disconnect"):
        raise RuntimeError("take truncated/disconnected; no automatic retry")
    return bool(terminated)


def numeric_fields(namespace, uppercase=False):
    included, omitted = {}, []
    for key, value in vars(namespace).items():
        if key.startswith("_") or (uppercase and not key.isupper()):
            continue
        try:
            included[key] = json_value(value)
        except TypeError:
            omitted.append(key)
    return {"values": included, "nonserializable_fields": omitted}


def validate_scorer_approval(scorer, approval):
    inner = getattr(scorer, "inner", scorer)
    checkpoint = getattr(inner, "checkpoint_path", None)
    if approval and checkpoint:
        expected = {(Path(item["path"]).resolve(), item["sha256"])
                    for item in approval["configuration"]["artifacts"]}
        actual = file_record(checkpoint)
        if (Path(actual["path"]), actual["sha256"]) not in expected:
            raise ValueError("live classifier is not pinned in the scoring approval")


def write_snapshot(path, root, env, scorer, args, reference, replay, binding, approval):
    root = Path(root).resolve()
    sources = {}
    for module in tuple(sys.modules.values()):
        module_path = getattr(module, "__file__", None)
        if not module_path or not os.path.isabs(str(module_path)):
            continue          # synthetic modules (e.g. torch's "_ops.py") carry relative names
        module_path = Path(module_path).resolve()
        if module_path.suffix == ".py" and module_path.is_relative_to(root) and module_path.is_file():
            sources[str(module_path.relative_to(root))] = file_record(module_path)
    env_module = sys.modules[type(env).__module__]
    inner_scorer = getattr(scorer, "inner", scorer)
    classifier_path = getattr(inner_scorer, "checkpoint_path", None)
    validate_scorer_approval(scorer, approval)
    calibration = getattr(env, "loudness", None)
    modules = {}
    for module_name in ("rl.piece_env", "rl.driver_piece", "rl.piece_hardware",
                        "rl.loudness", "play_midi_pieces", "classifier", "audio_features"):
        if module_name in sys.modules:
            modules[module_name] = numeric_fields(sys.modules[module_name], uppercase=True)
    value = {
        "created_unix_s": time.time(), "python": sys.version,
        "argv": json_value(vars(args)), "source_files": sources,
        "runtime_constants": modules, "runtime_environment": numeric_fields(env),
        "runtime_scorer": numeric_fields(inner_scorer),
        "runtime_classifier": numeric_fields(inner_scorer.clf) if hasattr(inner_scorer, "clf") else None,
        "runtime_executor": numeric_fields(env.executor) if hasattr(env, "executor") else None,
        "library_versions": {name: str(getattr(sys.modules[name], "__version__", "unavailable"))
                             for name in ("numpy", "torch", "stable_baselines3", "gymnasium")
                             if name in sys.modules},
        "scorer_checkpoint": file_record(classifier_path) if classifier_path else None,
        "policy_checkpoint": file_record(checkpoint_file(args.checkpoint)) if args.checkpoint else None,
        "score": file_record(args.piece), "reference": reference, "replay": replay, "cohort": binding,
        "scoring_approval": approval,
        "calibration_file": file_record(calibration.path, include_text=True) if calibration else None,
        "calibration_runtime": numeric_fields(calibration) if calibration else None,
        "plan": [dataclasses.asdict(stroke) for stroke in env.plan],
        "notes": [dataclasses.asdict(note) for note in env.notes],
        "cello_environment": {key: value for key, value in os.environ.items() if key.startswith("CELLO_")},
        "scoring_mode": "async" if args.async_score else "synchronous",
        "module": env_module.__name__,
        "guard_reward": importlib.import_module("rl.piece_env").reward_configuration(),
    }
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    return value


class EpisodeAudit:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata = metadata
        self.records = []
        self.serialization_s = []
        self.write_s = []
        self.stream = None

    def __enter__(self):
        self.stream = self.path.with_suffix(".jsonl").open("x")
        self.record({"event": "begin", "metadata": self.metadata, "unix_s": time.time()})
        return self

    def record(self, record):
        started = time.perf_counter()
        safe = json_value(record)
        payload = json.dumps(safe, allow_nan=False)
        encoded = time.perf_counter()
        self.stream.write(payload + "\n")
        self.stream.flush()
        self.serialization_s.append(encoded - started)
        self.write_s.append(time.perf_counter() - encoded)
        self.records.append(safe)

    def __exit__(self, error_type, error, traceback):
        try:
            self.record({"event": "end", "status": "aborted" if error_type else "complete",
                         "error": None if error is None else f"{error_type.__name__}: {error}",
                         "unix_s": time.time()})
        finally:
            self.stream.close()
        value = {"metadata": self.metadata, "status": "aborted" if error_type else "complete",
                 "records": self.records, "serialization_s": self.serialization_s, "write_s": self.write_s}
        self.path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        return False
