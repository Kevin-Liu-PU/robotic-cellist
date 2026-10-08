"""Prepare a local evaluation lock from explicitly configured private inputs.

Required: CELLO_PAIR_MANIFEST, CELLO_CACHE_SCORES, CELLO_CACHE_MANIFEST.
CELLO_REPO_ROOT defaults to this checkout; CELLO_DATA_ROOT defaults to its
rl/checkpoints_piece. Optional external provenance is read only when configured:
CELLO_PAPER_ROOT, CELLO_PAM_ROOT, CELLO_HF_HUB_CACHE, CELLO_AUDIOBOX_SOURCE.
No sibling research project, manuscript checkout, or home cache is searched.
Relocated data require a new lock; existing input/hash checks remain unchanged.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


from fetch_public import FILES, RECORD, digest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from analysis_paths import configured_path, repo_root, data_root, require_file, require_directory
REPO = repo_root()
DATA_ROOT = data_root()
PAIR_SOURCE = configured_path("CELLO_PAIR_MANIFEST")
CACHE_SOURCE = configured_path("CELLO_CACHE_SCORES")
CACHE_MANIFEST = configured_path("CELLO_CACHE_MANIFEST")
PAPER_ROOT = configured_path("CELLO_PAPER_ROOT")
PAM_ROOT = configured_path("CELLO_PAM_ROOT")
HUB_CACHE = configured_path("CELLO_HF_HUB_CACHE")
AUDIOBOX_SOURCE = configured_path("CELLO_AUDIOBOX_SOURCE")

HEADS = ["overall", "tone_quality", "attack_quality", "release_quality", "bow_control", "dynamic_accuracy"]


def write_csv(filename: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with filename.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(filename: Path, value: object) -> None:
    filename.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def git_head(directory: Path | None) -> str:
    if directory is None:
        return "not_configured"
    require_directory(directory, "configured source repository")
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def audio_info(filename: Path) -> dict:
    import numpy as np
    import soundfile as sf
    info = sf.info(filename)
    audio, sample_rate = sf.read(filename, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {filename}")
    peak = np.max(np.abs(audio), axis=0)
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2, axis=0))
    limit = 1.0 - (2.0 ** -15 if info.subtype == "PCM_16" else 2.0 ** -23)
    block = max(1, sample_rate // 100)
    usable = len(audio) // block * block
    silent = np.sqrt(np.mean(audio[:usable].reshape(-1, block, info.channels).astype(np.float64) ** 2, axis=1)) < 1e-4
    return {
        "duration_s": len(audio) / sample_rate, "sample_rate": sample_rate,
        "channels": info.channels, "frames": len(audio), "format": info.format,
        "subtype": info.subtype, "peak_by_channel": json.dumps(peak.tolist()),
        "rms_by_channel": json.dumps(rms.tolist()),
        "near_full_scale_fraction_by_channel": json.dumps(np.mean(np.abs(audio) >= limit, axis=0).tolist()),
        "quiet_10ms_fraction_by_channel": json.dumps(np.mean(silent, axis=0).tolist()),
        "audio_integrity": "readable_finite_nonempty",
    }


def pitch_probe(audio: np.ndarray, sample_rate: int, onset: float, duration: float) -> dict:
    import numpy as np
    from scipy.signal import find_peaks, resample_poly
    window_duration = min(duration, 0.25)
    start = max(0, int((onset + (duration - window_duration) / 2) * sample_rate))
    stop = min(len(audio), start + int(window_duration * sample_rate))
    excerpt = np.asarray(audio[start:stop], dtype=np.float64)
    if len(excerpt) < sample_rate * 0.04:
        return {"pitch_hz": "", "periodicity": "", "pitch_check": "too_short"}
    excerpt = resample_poly(excerpt, 1, 4)
    effective_rate = sample_rate / 4
    excerpt -= excerpt.mean()
    energy = float(np.dot(excerpt, excerpt))
    if energy < 1e-10:
        return {"pitch_hz": "", "periodicity": 0, "pitch_check": "quiet_or_unvoiced"}
    fft_size = 1 << (2 * len(excerpt) - 1).bit_length()
    spectrum = np.fft.rfft(excerpt, n=fft_size)
    correlation = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size)[:len(excerpt)]
    lower = max(2, int(effective_rate / 500))
    upper = min(len(correlation) - 2, int(effective_rate / 60))
    candidates, _ = find_peaks(correlation[lower:upper + 1])
    if not len(candidates):
        return {"pitch_hz": "", "periodicity": 0, "pitch_check": "no_periodic_peak"}
    candidates += lower
    best = int(candidates[np.argmax(correlation[candidates])])
    peak = float(correlation[best] / correlation[0])
    previous, current, following = correlation[best - 1:best + 2]
    denominator = previous - 2 * current + following
    displacement = 0.5 * (previous - following) / denominator if denominator != 0 else 0
    frequency = effective_rate / (best + displacement)
    return {"pitch_hz": frequency, "periodicity": peak,
            "pitch_check": "periodic" if peak >= 0.6 else "low_periodicity"}


def source_row(filename: Path, source_id: str, source_type: str, url: str = "", license_name: str = "not_verified") -> dict:
    row = {"source_id": source_id, "source_type": source_type, "path": str(filename), "url": url,
           "license": license_name, "status": "available" if filename.exists() else "missing",
           "version": "Zenodo record 10696680" if source_type == "external_original" else "local_snapshot",
           "sha256": digest(filename) if filename.exists() else "",
           "size_bytes": filename.stat().st_size if filename.exists() else ""}
    if filename.exists() and filename.suffix.lower() == ".wav":
        row.update(audio_info(filename))
    return row


def model_inventory() -> dict:
    cache = HUB_CACHE
    files = {
        "training_checkpoint": REPO / "SoundClassifier/checkpoints/quality_cnn_multilength.pt",
        "heldout_checkpoint": REPO / "SoundClassifier/checkpoints/quality_cnn_human_A1_A5.pt",
        "classifier": REPO / "SoundClassifier/classifier.py",
        "classifier_architecture": REPO / "SoundClassifier/quality_classifier.py",
        "audio_features": REPO / "SoundClassifier/audio_features.py",
        "dataset": REPO / "SoundClassifier/dataset.py",
        "posthoc_score": REPO / "rl/posthoc_score.py",
        "posthoc_heldout": REPO / "rl/posthoc_heldout.py",
        "piece_env": REPO / "rl/piece_env.py",
        "loudness_calibration": REPO / "SoundClassifier/checkpoints/loudness_model.json",
        "existing_public_adapter": ROOT.parents[1] / "external_judges/score_takes.py",
        "pam_source": PAM_ROOT / "PAM/PAM.py" if PAM_ROOT else None,
        "pam_config": PAM_ROOT / "PAM/config.yml" if PAM_ROOT else None,
        "pam_dataset": PAM_ROOT / "dataset.py" if PAM_ROOT else None,
        "audiobox_source": AUDIOBOX_SOURCE,
    }
    for model_name, repository, weight_name in [
        ("audiobox", "facebook--audiobox-aesthetics", "model.safetensors"),
        ("pam", "microsoft--msclap", "CLAP_weights_2023.pth"),
        ("gpt2", "gpt2", "model.safetensors"),
    ]:
        if cache is None:
            continue
        reference = cache / f"models--{repository}/refs/main"
        if reference.exists():
            revision = reference.read_text().strip()
            snapshot = reference.parent.parent / "snapshots" / revision
            files[f"{model_name}_weights"] = snapshot / weight_name
            files[f"{model_name}_revision"] = reference
            for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"]:
                if (snapshot / name).exists():
                    files[f"{model_name}_{name}"] = snapshot / name
    return {name: {"path": str(filename) if filename is not None else "",
                   "sha256": digest(filename) if filename is not None and filename.exists() else "",
                   "size_bytes": filename.stat().st_size if filename is not None and filename.exists() else 0}
            for name, filename in files.items()}


def main() -> None:
    require_file(PAIR_SOURCE, "CELLO_PAIR_MANIFEST")
    require_file(CACHE_SOURCE, "CELLO_CACHE_SCORES")
    require_file(CACHE_MANIFEST, "CELLO_CACHE_MANIFEST")
    require_directory(DATA_ROOT, "CELLO_DATA_ROOT")
    if PAPER_ROOT is not None:
        require_directory(PAPER_ROOT, "CELLO_PAPER_ROOT")
    if PAM_ROOT is not None:
        require_directory(PAM_ROOT, "CELLO_PAM_ROOT")
        for relative in ["PAM/PAM.py", "PAM/config.yml", "dataset.py"]:
            require_file(PAM_ROOT / relative, "CELLO_PAM_ROOT")
    if HUB_CACHE is not None:
        require_directory(HUB_CACHE, "CELLO_HF_HUB_CACHE")
    if AUDIOBOX_SOURCE is not None:
        require_file(AUDIOBOX_SOURCE, "CELLO_AUDIOBOX_SOURCE")
    import soundfile as sf
    if (ROOT / "SCORES.csv").exists():
        raise SystemExit("A scored run already exists. Use a new run directory rather than overwriting its lock.")
    ROOT.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(CACHE_MANIFEST.read_text())
    pairs = json.loads(PAIR_SOURCE.read_text())["pairs"]
    if len(pairs) != 45:
        raise ValueError("Authoritative pair index is no longer the expected 45 pairs.")
    source_rows = []
    pair_rows = []
    segment_rows = []
    input_checks = []
    take_map = {}
    download_log = ROOT / "download_attempts.jsonl"
    downloads = [json.loads(line) for line in download_log.read_text().splitlines()] if download_log.exists() else []
    metadata_file = ROOT / "sources/zenodo_record.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    license_name = metadata.get("metadata", {}).get("license", {}).get("id", "not_verified")
    for name in FILES:
        filename = ROOT / "raw" / name
        row = source_row(filename, f"vienna:{name}", "external_original", f"{RECORD}/files/{name}?download=1", license_name)
        matching = [entry for entry in downloads if entry["destination"] == str(filename)]
        row["downloaded_utc"] = next((entry["finished_utc"] for entry in matching if entry["success"]), "")
        row["download_attempts"] = len(matching)
        row["last_http_status"] = matching[-1]["http_status"] if matching else ""
        source_rows.append(row)
    for pair in pairs:
        pair_id = f"{pair['piece']}_s{pair['seed']}_p{pair['pair']}"
        for key, arm in [("baseline", "Our Base"), ("policy", "Our RL")]:
            record = pair[key]
            take_id = record["dir"]
            directory = require_directory(DATA_ROOT / take_id, "CELLO_DATA_ROOT (take directory)")
            pair_rows.append({"pair_id": pair_id, "piece": pair["piece"], "seed": pair["seed"],
                              "pair": pair["pair"], "arm": arm, "take_id": take_id,
                              "session_id": Path(pair["log"]).stem, "source_log": str(directory.parent / pair["log"]),
                              "launch_time_local": record.get("launch", ""),
                              "capture_date_local": take_id.split("_")[1],
                              "mode": record.get("performance_mode", ""), "path": str(directory)})
            if take_id in take_map:
                continue
            saved = provenance["takes"].get(take_id)
            if saved is None:
                raise ValueError(f"No pinned provenance for {take_id}")
            for name, expected in saved["inputs_sha256"].items():
                filename = directory / name
                row = source_row(filename, f"{take_id}:{name}", "our_original", license_name="private_research_no_external_upload")
                row["capture_date_local"] = take_id.split("_")[1]
                row["capture_time_local"] = take_id.split("_")[2]
                source_rows.append(row)
                input_checks.append({"path": str(filename), "expected_sha256": expected,
                                     "actual_sha256": row["sha256"], "match": expected == row["sha256"]})
            stem = saved["stem"]
            wav = directory / f"{stem}.wav"
            perf_file = directory / f"{stem}_performance.json"
            if not wav.exists() or not perf_file.exists():
                continue
            performance = json.loads(perf_file.read_text())
            audio, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
            info = next(row for row in source_rows if row["path"] == str(wav))
            take_map[take_id] = {"take_id": take_id, "arm": arm, "piece": pair["piece"], "seed": pair["seed"],
                                 "pair_id": pair_id, "session_id": Path(pair["log"]).stem, "wav": str(wav),
                                 "wav_sha256": info["sha256"], "performance": str(perf_file),
                                 "state": str(directory / f"{stem}_state.npy"), "duration_s": info["duration_s"],
                                 "sample_rate": sample_rate, "channels": audio.shape[1], "stem": stem,
                                 "mode": performance.get("mode", "unknown"), "n_timeline": len(performance["timeline"])}
            segment_rows.append({"segment_id": f"{take_id}:full", "take_id": take_id,
                                 "system": "our", "arm": arm, "piece": pair["piece"], "seed": pair["seed"],
                                 "pair_id": pair_id, "session_id": Path(pair["log"]).stem,
                                 "source_file": str(wav), "source_sha256": info["sha256"],
                                 "sample_rate": sample_rate, "start_frame": 0, "end_frame": len(audio),
                                 "start_s": 0, "end_s": len(audio) / sample_rate,
                                 "duration_s": len(audio) / sample_rate, "event_id": "", "note_position": "",
                                 "intended_pitch": "A3_open_string_reported", "segment_type": "full_recording_overview",
                                 "boundary_source": "file boundaries", "included": True,
                                 "reason": "overview_only_unmatched_music_and_silence",
                                 "analysis": "UNMATCHED_OVERVIEW", "pitch_hz": "", "periodicity": "", "pitch_check": "see_bow_windows"})
            for position, note in enumerate(performance["timeline"]):
                onset = float(performance.get("audio_lead_in_s", 0)) + float(note["actual_onset"])
                duration = float(note["actual_duration"])
                start_frame = int(onset * sample_rate)
                stop_frame = min(len(audio), start_frame + int(duration * sample_rate))
                valid = start_frame >= 0 and stop_frame - start_frame >= sample_rate // 20
                segment_rows.append({"segment_id": f"{take_id}:note:{position:03d}", "take_id": take_id,
                                     "system": "our", "arm": arm, "piece": pair["piece"], "seed": pair["seed"],
                                     "pair_id": pair_id, "session_id": Path(pair["log"]).stem,
                                     "source_file": str(wav), "source_sha256": info["sha256"],
                                     "sample_rate": sample_rate, "start_frame": start_frame, "end_frame": stop_frame,
                                     "start_s": start_frame / sample_rate, "end_s": stop_frame / sample_rate,
                                     "duration_s": (stop_frame - start_frame) / sample_rate,
                                     "event_id": note.get("note_index", position), "note_position": position,
                                     "intended_pitch": "A3_open_string_reported", "segment_type": "executed_complete_bow_window",
                                     "boundary_source": str(perf_file), "included": valid,
                                     "reason": "legacy_timeline_window" if valid else "too_short_or_out_of_bounds",
                                     "analysis": "OUR_REFERENCE_ONLY", **pitch_probe(audio[:, 0], sample_rate, onset, duration)})
            print(f"Checked {take_id}: {len(performance['timeline'])} bow windows", flush=True)
    inventory = model_inventory()
    for key, model_name in [("multilength", "training_checkpoint"), ("human_A1_A5", "heldout_checkpoint")]:
        expected = provenance["checkpoints_sha256"][key]
        input_checks.append({"path": inventory[model_name]["path"], "expected_sha256": expected,
                             "actual_sha256": inventory[model_name]["sha256"], "match": expected == inventory[model_name]["sha256"]})
    for filename, expected in provenance["code_sha256"]["SoundClassifier (repo == code_snapshot)"].items():
        actual = digest(REPO / "SoundClassifier" / filename)
        input_checks.append({"path": str(REPO / "SoundClassifier" / filename), "expected_sha256": expected,
                             "actual_sha256": actual, "match": expected == actual})
    for relative in ["rl/posthoc_score.py", "rl/posthoc_heldout.py"]:
        expected = provenance["code_sha256"][relative]
        actual = digest(REPO / relative)
        input_checks.append({"path": str(REPO / relative), "expected_sha256": expected,
                             "actual_sha256": actual, "match": expected == actual})
    if any(not row["match"] for row in input_checks):
        write_json(ROOT / "INPUT_CHECKS.json", input_checks)
        raise SystemExit("Source/code mismatch: do not reuse the prior cache.")
    write_csv(ROOT / "SOURCE_MANIFEST.csv", source_rows)
    write_csv(ROOT / "PAIRS.csv", pair_rows)
    write_csv(ROOT / "SEGMENTS.csv", segment_rows)
    write_json(ROOT / "INPUT_CHECKS.json", input_checks)
    write_json(ROOT / "TAKES.json", take_map)
    write_json(ROOT / "MODEL_MANIFEST.json", inventory)
    missing = [name for name in FILES if not (ROOT / "raw" / name).exists()]
    protocol = {
        "task": "VIE01", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PARTIAL_LOCK_EXTERNAL_INPUTS_PENDING" if missing else "EXTERNAL_SEGMENT_REVIEW_REQUIRED",
        "paper_head": git_head(PAPER_ROOT), "robot_head": git_head(REPO),
        "external_record": RECORD, "missing_external_inputs": missing,
        "n_our_pairs": len(pairs), "n_unique_our_recordings": len(take_map),
        "n_our_bow_windows": sum(row["segment_type"] == "executed_complete_bow_window" for row in segment_rows), "models": inventory,
        "interface_probe_take_ids": [pairs[0][key]["dir"] for key in ["baseline", "policy"]],
        "selection": "All original 45 pairs retained; no selection by scores. Pitch checks are descriptive and do not remove poor/unpitched bows.",
        "pitch_probe": "Central <=250 ms, resample_poly /4, unwindowed autocorrelation; strongest local peak in 60-500 Hz; periodicity>=0.6 flags periodic. Diagnostic, not ground-truth transcription.",
        "classifier": {
            "preprocessing": "LEGACY: first channel, no peak normalization; executed onset+lead-in and duration; center-crop or reflect-pad to 22050 samples at 44100 Hz; window_pos=0.5",
            "physical": ["commanded_depth_m", "fixed_zero", "mean_abs_measured_bow_speed", "mean_commanded_bow_position_u", "mean_measured_torque_magnitude", "max_measured_torque_magnitude"],
            "physical_normalization": "checkpoint z-score then clip to [-2.5,2.5]",
            "composite": {"base_weights": {"tone_quality": 0.45, "attack_quality": 0.25, "release_quality": 0.15, "overall": 0.15},
                          "fill": "min(actual_duration_s / 0.5, 1)",
                          "short_fill_rule": "attack+=0.30*(1-fill); release+=0.10*(1-fill); tone=max(0,tone-0.30*(1-fill)); overall=max(0,overall-0.10*(1-fill)); divide by weight sum",
                          "aggregation": "arithmetic mean over valid full-bow windows; not complete RL reward"},
            "external_status": "not_computable_pending_equivalent_physical_metadata; no zero imputation and no untrained audio-only substitution",
            "heads": HEADS,
        },
        "public_models": {
            "PAM": "existing local PAM/MS-CLAP 2023; first channel, 44100 Hz; 7 s chunks; short inputs tiled and cropped; long tail replaced by last7s; arithmetic mean of chunk scores; no custom normalization",
            "Audiobox_PQ_PC": "same Audiobox Aesthetics 0.0.4 model; mono channel mean, torchaudio resample16000; 10s nonoverlap windows, zero pad final window with valid mask, duration-weighted mean; PC is complexity, not higher-is-better",
            "main_analysis": "Do not treat native short-input padding/tiling as validation of complete-bow tone/attack/release quality. Final shared excerpts require external README/score/audio boundary checks before a full lock.",
            "overview": "Full recordings are a separately labeled unmatched-content overview, not the pitch-matched V2 primary comparison.",
        },
        "V1": "Pending actual external audio; corresponding score events if supported, otherwise explicitly unpaired pitch-group/recording descriptions.",
        "V2": "Pending external A3 pitch and reliable complete-bow boundaries; no cross-laboratory causal inference.",
        "missing_outputs": "Use not_computable/not_comparable/missing_source, never numeric zeros; retain every error and attempted input.",
        "statistics": "Descriptive only; original recording, pair, seed, piece and session counts remain distinct. Windows are not independent robot experiments. No significance testing or averaged multi-model win rate.",
        "sensitivity": "No amplitude matching, denoising, EQ, pitch shift or time stretching in this first run.",
        "cache": {"path": str(CACHE_SOURCE), "sha256": digest(CACHE_SOURCE), "condition": "LEGACY only; PEAK excluded",
                  "provenance_path": str(CACHE_MANIFEST), "provenance_sha256": digest(CACHE_MANIFEST),
                  "validation": "All raw WAV/performance/state pins and classifier/posthoc source/checkpoint pins checked; each reused head further compared to original rounded posthoc cache by note identity."},
        "files": {name: digest(ROOT / name) for name in ["SOURCE_MANIFEST.csv", "PAIRS.csv", "SEGMENTS.csv", "TAKES.json", "MODEL_MANIFEST.json"]},
        "scope": "No robot operation, training, C1 changes, original recording writes, manuscript edits, commits or pushes.",
        "final_external_lock_required_before_external_scoring": True,
    }
    write_json(ROOT / "PROTOCOL_LOCK.json", protocol)
    print(json.dumps({"pairs": len(pairs), "takes": len(take_map), "segments": len(segment_rows), "input_checks": len(input_checks), "missing_external": missing}, indent=2))


if __name__ == "__main__":
    main()
