"""Reproducible computational evaluation for saved cello performances.

This version reports stroke-level metrics, run-level summaries, paired run
comparisons, and piece-level breakdowns. It keeps the Base=DIR/RL=DIR CLI of
older evaluators while making missing data and aggregation explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SOUND_CLASSIFIER = REPO_ROOT / "SoundClassifier"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOUND_CLASSIFIER) not in sys.path:
    sys.path.insert(0, str(SOUND_CLASSIFIER))

from audio_features import SCALAR_FEATURE_NAMES, extract_scalar_features
from classifier import BowingQualityClassifier
from rl.loudness import LoudnessModel, measure_dbfs

WINDOW_SEC = 0.5
PRE_ROLL_SEC = 0.05
MIN_ANALYSIS_SEC = 0.08
BOW_LENGTH_M = 0.5249483880502777

CSV_FIELDS = [
    "piece", "run", "episode", "stroke", "duration_s", "duration_error_s",
    "drift_s", "window_s", "dbfs_raw", "dbfs_model", "dynamic_zone",
    "dynamic_zone_hit", "dynamic_distance_db", "dynamic_centre_error_db",
    "quality_eff", "overall", "tone_quality", "attack_quality",
    "release_quality", "bow_control", "dynamic_accuracy", "defect_score",
    "bow_speed_measured", "depth_m", "bow_position", "force_mean",
    "force_std", "torque_mean", "torque_max", "bow_usage_m",
    "smoothness_jerk", "logged_quality", "logged_tone",
]


def window_bounds(start_t: float, end_t: float) -> tuple[float, float]:
    duration = max(0.0, end_t - start_t)
    if duration > WINDOW_SEC + 2 * PRE_ROLL_SEC:
        centre = start_t + PRE_ROLL_SEC + (duration - PRE_ROLL_SEC) / 2.0
        return centre - WINDOW_SEC / 2.0, centre + WINDOW_SEC / 2.0
    return start_t, start_t + max(duration, MIN_ANALYSIS_SEC)


def load_state(path: Path | None) -> tuple[np.ndarray, np.ndarray | None]:
    if path is None or not path.exists():
        return np.empty(0, dtype=object), None
    data = np.load(path, allow_pickle=True)
    if len(data) == 0 or not isinstance(data[0], dict):
        return data, None
    timestamps = np.asarray([float(frame.get("t", np.nan)) for frame in data])
    if not np.all(np.isfinite(timestamps)):
        return data, None
    return data, timestamps - timestamps[0]


def state_features(state_data: np.ndarray, timestamps: np.ndarray | None,
                   start_t: float, end_t: float) -> dict:
    if timestamps is None or len(state_data) == 0:
        return {}
    mask = (timestamps >= start_t) & (timestamps <= end_t)
    frames = state_data[mask]
    if len(frames) == 0:
        return {}

    def field(name: str) -> np.ndarray:
        return np.asarray([float(frame[name]) for frame in frames
                           if name in frame], dtype=np.float64)

    result = {}
    position = field("bow_position")
    speed = field("bow_speed")
    force = field("ft_fz")
    if len(position):
        result["bow_position"] = float(np.mean(position))
    if len(speed):
        result["bow_speed_measured"] = float(np.mean(np.abs(speed)))
    if len(force):
        result["force_mean"] = float(np.mean(np.abs(force)))
        result["force_std"] = float(np.std(force))

    # Build torque vectors frame by frame. Independent filtering of each
    # component can produce arrays with different lengths and invalid norms.
    torque_rows = []
    for frame in frames:
        if all(name in frame for name in ("ft_tx", "ft_ty", "ft_tz")):
            torque_rows.append(np.sqrt(sum(float(frame[name]) ** 2
                                           for name in ("ft_tx", "ft_ty", "ft_tz"))))
    if torque_rows:
        torque = np.asarray(torque_rows, dtype=np.float64)
        result["torque_mean"] = float(np.mean(torque))
        result["torque_max"] = float(np.max(torque))
    return result


def motion_metrics(state_data: np.ndarray, timestamps: np.ndarray | None) -> dict:
    if timestamps is None or len(state_data) < 3:
        return {}
    positions = np.asarray([float(frame["bow_position"]) for frame in state_data
                            if "bow_position" in frame], dtype=np.float64)
    result = {}
    if len(positions) >= 2:
        result["bow_usage_m"] = float(np.sum(np.abs(np.diff(positions))) * BOW_LENGTH_M)
    if len(timestamps) < 4:
        return result
    dt = float(np.median(np.diff(timestamps)))
    if dt <= 0:
        return result
    jerk_values = []
    for joint in ("j0", "j1", "j2", "j3", "j4", "j5"):
        values = np.asarray([float(frame[joint]) for frame in state_data
                             if joint in frame], dtype=np.float64)
        if len(values) >= 4:
            acceleration = np.gradient(np.gradient(values, dt), dt)
            jerk_values.append(float(np.mean(np.square(np.gradient(acceleration, dt)))))
    if jerk_values:
        result["smoothness_jerk"] = float(np.mean(jerk_values))
    return result


def physical_params(timing: dict, measured: dict) -> np.ndarray:
    return np.asarray([
        float(timing.get("depth_m", 0.0)),
        measured.get("force_std", 0.0),
        measured.get("bow_speed_measured", timing.get("mean_speed", 0.0)),
        measured.get("bow_position", (float(timing.get("u_start", 0.0)) +
                                        float(timing.get("u_end", 0.0))) / 2.0),
        measured.get("torque_mean", 0.0),
        measured.get("torque_max", measured.get("torque_mean", 0.0)),
    ], dtype=np.float32)


def quality_eff(scores: dict, duration_s: float) -> float | None:
    names = ("tone_quality", "attack_quality", "release_quality", "overall")
    if not all(name in scores for name in names):
        return scores.get("overall")
    fill = np.clip(duration_s / WINDOW_SEC, 0.0, 1.0)
    weighted = ((0.45 - 0.30 * (1.0 - fill)) * scores["tone_quality"] +
                (0.25 + 0.30 * (1.0 - fill)) * scores["attack_quality"] +
                (0.15 + 0.10 * (1.0 - fill)) * scores["release_quality"] +
                (0.15 - 0.10 * (1.0 - fill)) * scores["overall"])
    return float(np.clip(0.5 + (0.85 + 0.15 * fill) * (weighted - 0.5), 0.0, 1.0))


def defect_score(audio: np.ndarray) -> float | None:
    if len(audio) == 0:
        return None
    features = dict(zip(SCALAR_FEATURE_NAMES,
                        extract_scalar_features(audio.astype(np.float32))))
    defects = []
    if "hnr_db_mean" in features:
        defects.append(np.clip((12.0 - features["hnr_db_mean"]) / 9.0, 0, 1))
    if "voiced_fraction" in features:
        defects.append(np.clip((0.98 - features["voiced_fraction"]) / 0.28, 0, 1))
    if "attack_overshoot" in features:
        defects.append(np.clip((features["attack_overshoot"] - 0.85) / 0.55, 0, 1))
    if "f0_stability_cents" in features:
        defects.append(np.clip((features["f0_stability_cents"] - 1.0) / 19.0, 0, 1))
    return float(np.mean(defects)) if defects else None


def score_stroke(audio: np.ndarray, sample_rate: int, timing: dict,
                 classifier: BowingQualityClassifier, loudness: LoudnessModel,
                 measured: dict, motion: dict, piece: str, label: str,
                 episode: int, stroke: int, log_row: dict | None = None) -> dict | None:
    start_t = float(timing.get("actual_onset", timing.get("t0", 0.0)))
    if "actual_duration" in timing:
        duration = float(timing["actual_duration"])
    else:
        duration = float(timing.get("t1", start_t) - start_t)
    end_t = start_t + max(0.0, duration)
    win_start, win_end = window_bounds(start_t, end_t)
    start = max(0, int(round(win_start * sample_rate)))
    end = min(len(audio), int(round(win_end * sample_rate)))
    raw_window = audio[start:end]
    if len(raw_window) == 0:
        return None

    dbfs_raw = float(measure_dbfs(raw_window))
    peak = float(np.max(np.abs(raw_window)))
    model_window = raw_window / peak if peak > 0.05 else raw_window * 0.0
    position = np.clip((win_start - start_t) / max(duration, 1e-9), 0.0, 1.0)
    scores = classifier.predict_detailed(model_window, physical_params(timing, measured),
                                         window_pos=float(position))
    zone = loudness.zone_for_dynamic(str(timing.get("dynamic", "mf")))
    dbfs_model = dbfs_raw - loudness.gain_offset_db
    lo, hi = loudness.zone_bounds(zone)
    distance = 0.0 if lo <= dbfs_model <= hi else min(abs(dbfs_model - lo),
                                                        abs(dbfs_model - hi))
    result = {
        "piece": piece, "run": label, "episode": episode, "stroke": stroke,
        "duration_s": duration,
        "duration_error_s": duration - float(timing.get("planned_duration", duration)),
        "drift_s": float(timing.get("drift", 0.0)),
        "window_s": (end - start) / sample_rate,
        "dbfs_raw": dbfs_raw, "dbfs_model": dbfs_model, "dynamic_zone": zone,
        "dynamic_zone_hit": int(distance == 0.0), "dynamic_distance_db": float(distance),
        "dynamic_centre_error_db": float(dbfs_model - loudness.zone_centre(zone)),
        "quality_eff": quality_eff(scores, duration), **scores,
        "defect_score": defect_score(model_window), **motion,
        "bow_speed_measured": measured.get("bow_speed_measured", timing.get("mean_speed")),
        "depth_m": float(timing.get("depth_m", 0.0)),
        "bow_position": measured.get("bow_position"),
        "force_mean": measured.get("force_mean"), "force_std": measured.get("force_std"),
        "torque_mean": measured.get("torque_mean"), "torque_max": measured.get("torque_max"),
    }
    if log_row:
        result["logged_quality"] = log_row.get("quality")
        result["logged_tone"] = log_row.get("tone")
    return result


def piece_name(run_dir: Path, perf_files: list[Path]) -> str:
    if perf_files:
        try:
            source = json.loads(perf_files[0].read_text(encoding="utf-8"))["score"]["source"]
            return Path(source).stem
        except (KeyError, OSError, json.JSONDecodeError, TypeError):
            pass
    return run_dir.name


def summarize(rows: list[dict], run_dir: Path, label: str, piece: str,
              skipped: int = 0) -> dict:
    fields = ("quality_eff", "overall", "tone_quality", "attack_quality",
              "release_quality", "bow_control", "dynamic_accuracy", "dbfs_raw",
              "dbfs_model", "dynamic_distance_db", "dynamic_centre_error_db",
              "dynamic_zone_hit", "defect_score", "bow_usage_m", "smoothness_jerk",
              "duration_error_s", "drift_s")
    summary = {"run": str(run_dir), "label": label, "piece": piece,
               "strokes": len(rows), "skipped": skipped}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary[field] = float(np.mean(values)) if values else None
    return summary


def score_run(run_dir: Path, classifier: BowingQualityClassifier,
              loudness: LoudnessModel, writer=None, label="") -> dict:
    timeline_path = run_dir / "timeline.json"
    perf_files = sorted(run_dir.glob("*_performance.json"))
    metadata = None
    if timeline_path.exists():
        metadata = json.loads(timeline_path.read_text(encoding="utf-8"))
    elif perf_files:
        metadata = json.loads(perf_files[0].read_text(encoding="utf-8")).get("timeline", [])
    piece = piece_name(run_dir, perf_files)
    rows, skipped = [], 0

    if metadata is not None:
        wav_files = sorted(run_dir.glob("*.wav"))
        state_files = sorted(run_dir.glob("*.npy"))
        if not wav_files:
            print(f"Warning: no WAV file found in {run_dir}")
        else:
            audio, sample_rate = sf.read(wav_files[0], dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            state_data, timestamps = load_state(state_files[0] if state_files else None)
            motion = motion_metrics(state_data, timestamps)
            for index, timing in enumerate(metadata):
                onset = float(timing.get("actual_onset", 0.0))
                duration = float(timing.get("actual_duration", 0.0))
                row = score_stroke(audio, sample_rate, timing, classifier, loudness,
                                   state_features(state_data, timestamps, onset, onset + duration),
                                   motion, piece, label, 0,
                                   int(timing.get("note_index", index)) + 1)
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)
    elif (run_dir / "episode_audio").exists():
        audio_dir = run_dir / "episode_audio"
        logs = {}
        log_path = run_dir / "stroke_log.jsonl"
        if log_path.exists():
            for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if "episode" in item and "stroke" in item:
                        logs[(int(item["episode"]), int(item["stroke"]))] = item
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    print(f"Warning: skipped malformed log line {line_number} in {log_path}: {exc}")
        for times_path in sorted(audio_dir.glob("ep*_times*.json")):
            prefix = times_path.name.split("_times")[0]
            episode = int(prefix[2:])
            wav_files = sorted(audio_dir.glob(f"{prefix}_full*.wav"))
            if not wav_files:
                skipped += 1
                continue
            audio, sample_rate = sf.read(wav_files[0], dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            state_files = sorted(audio_dir.glob(f"{prefix}_state*.npy"))
            state_data, timestamps = load_state(state_files[0] if state_files else None)
            motion = motion_metrics(state_data, timestamps)
            for timing in json.loads(times_path.read_text(encoding="utf-8")).get("strokes", []):
                stroke = int(timing["stroke"])
                row = score_stroke(audio, sample_rate, timing, classifier, loudness,
                                   state_features(state_data, timestamps,
                                                  float(timing.get("t0", 0.0)),
                                                  float(timing.get("t1", 0.0))),
                                   motion, piece, label, episode, stroke,
                                   logs.get((episode, stroke)))
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)
    else:
        print(f"Warning: no recognized evaluation structure in {run_dir}")

    if writer:
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    return summarize(rows, run_dir, label, piece, skipped)


def mean_difference(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def print_comparisons(summaries: list[dict]) -> None:
    by_label = {row["label"]: row for row in summaries}
    pairs = []
    for base in summaries:
        if "Base" not in base["label"]:
            continue
        suffix = base["label"].replace("Base", "", 1)
        policy = by_label.get(f"RL{suffix}")
        if policy and base["piece"] == policy["piece"]:
            pairs.append((base, policy))
    if not pairs:
        return
    print("\nPaired run differences (RL - Baseline; run is the statistical unit):")
    for field in ("quality_eff", "bow_control", "dynamic_accuracy", "dynamic_zone_hit",
                  "dynamic_distance_db", "defect_score", "drift_s"):
        differences = [policy[field] - base[field] for base, policy in pairs
                       if base.get(field) is not None and policy.get(field) is not None]
        mean, sd = mean_difference(differences)
        if mean is not None:
            print(f"  {field:22s} {mean:+.4f} (SD {sd:.4f}, n={len(differences)})")


def print_piece_breakdown(summaries: list[dict]) -> None:
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[summary["piece"]].append(summary)
    if len(grouped) <= 1:
        return
    print("\nPiece breakdown (mean of run summaries):")
    for piece, rows in sorted(grouped.items()):
        print(f"  {piece}:")
        for label_group, selected in (("Baseline", [r for r in rows if "Base" in r["label"]]),
                                      ("RL", [r for r in rows if "RL" in r["label"]])):
            values = [r["quality_eff"] for r in selected if r.get("quality_eff") is not None]
            if values:
                print(f"    {label_group:9s} quality_eff={np.mean(values):.3f} n_runs={len(values)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Computational evaluation of saved cello performances")
    parser.add_argument("runs", nargs="+", metavar="LABEL=RUN_DIR")
    parser.add_argument("--checkpoint", default=str(SOUND_CLASSIFIER / "checkpoints" / "quality_cnn_human_A1_A5.pt"))
    parser.add_argument("--loudness-model", default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    classifier = BowingQualityClassifier(args.checkpoint)
    loudness = LoudnessModel(args.loudness_model)
    csv_file = args.csv.open("w", newline="", encoding="utf-8") if args.csv else None
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS) if csv_file else None
    if writer:
        writer.writeheader()
    summaries = []
    try:
        for item in args.runs:
            label, raw_dir = item.split("=", 1)
            run_dir = Path(raw_dir)
            if not run_dir.is_absolute():
                run_dir = REPO_ROOT / run_dir
            summary = score_run(run_dir, classifier, loudness, writer, label)
            summaries.append(summary)
            print(f"\n{label} [{summary['piece']}]: {summary['strokes']} strokes, skipped={summary['skipped']}")
            if summary["strokes"]:
                print(f"  quality_eff={summary['quality_eff']:.3f}  "
                      f"tone={summary['tone_quality']:.3f}  "
                      f"attack={summary['attack_quality']:.3f}  "
                      f"release={summary['release_quality']:.3f}")
                print(f"  dynamic_hit={summary['dynamic_zone_hit']:.3f}  "
                      f"distance={summary['dynamic_distance_db']:.3f} dB  "
                      f"drift={summary['drift_s'] * 1000:.2f} ms")
    finally:
        if csv_file:
            csv_file.close()
    print_comparisons(summaries)
    print_piece_breakdown(summaries)


if __name__ == "__main__":
    main()
