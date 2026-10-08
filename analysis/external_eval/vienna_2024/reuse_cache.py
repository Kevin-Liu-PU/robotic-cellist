from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from prepare import CACHE_MANIFEST, CACHE_SOURCE, DATA_ROOT, HEADS, REPO, ROOT, digest, write_csv, write_json, require_file


def composite(values: dict, duration: float) -> float:
    shift = 1.0 - min(duration / 0.5, 1.0)
    weights = {"tone_quality": max(0.0, 0.45 - 0.30 * shift), "attack_quality": 0.25 + 0.30 * shift,
               "release_quality": 0.15 + 0.10 * shift, "overall": max(0.0, 0.15 - 0.10 * shift)}
    return sum(float(values[name]) * weight for name, weight in weights.items()) / sum(weights.values())


def validate_cache_provenance(protocol: dict) -> None:
    """Bind the selected cohort manifest to the path and hash saved by prepare.py."""
    cache = protocol.get("cache", {})
    if not cache.get("provenance_path") or not cache.get("provenance_sha256"):
        raise ValueError("Missing cache provenance pin; run prepare.py to create a fresh protocol lock.")
    manifest = require_file(CACHE_MANIFEST, "CELLO_CACHE_MANIFEST")
    if manifest.resolve() != Path(cache["provenance_path"]).expanduser().resolve():
        raise ValueError(
            "CELLO_CACHE_MANIFEST differs from the protocol lock. "
            "Restore the locked manifest or run prepare.py to create a fresh lock."
        )
    if digest(manifest) != cache["provenance_sha256"]:
        raise ValueError("Cache provenance manifest changed after protocol lock")


def main() -> None:
    require_file(CACHE_SOURCE, "CELLO_CACHE_SCORES")
    require_file(CACHE_MANIFEST, "CELLO_CACHE_MANIFEST")
    protocol_path = ROOT / "PROTOCOL_LOCK.json"
    protocol = json.loads(protocol_path.read_text())
    validate_cache_provenance(protocol)
    for entry in json.loads((ROOT / "INPUT_CHECKS.json").read_text()):
        if digest(Path(entry["path"])) != entry["actual_sha256"]:
            raise ValueError(f"Original input/code changed: {entry['path']}")
    for name, entry in protocol["models"].items():
        if entry["sha256"] and digest(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Model/interface changed after protocol lock: {name}")
    if digest(CACHE_SOURCE) != protocol["cache"]["sha256"]:
        raise ValueError("Cache changed after protocol lock")
    for name, expected in protocol["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Locked input table changed: {name}")
    segments = {row["segment_id"]: row for row in csv.DictReader((ROOT / "SEGMENTS.csv").open())}
    takes = json.loads((ROOT / "TAKES.json").read_text())
    provenance = json.loads(CACHE_MANIFEST.read_text())
    original = {}
    original_names = {"overall": "overall", "tone_quality": "tone_head", "attack_quality": "attack", "release_quality": "release", "composite": "tone"}
    for take_id in takes:
        for checkpoint in ["multilength", "human_A1_A5"]:
            filename = DATA_ROOT / take_id / f"posthoc_{checkpoint}.json"
            expected = provenance["takes"][take_id]["caches_sha256"][filename.name]
            if digest(filename) != expected:
                raise ValueError(f"Historical posthoc cache changed: {filename}")
            cache = json.loads(filename.read_text())
            original[(take_id, checkpoint)] = {row["note_index"]: row for row in cache["per_note"]}
    scores = []
    checked = 0
    max_difference = 0.0
    keys = set()
    for row in csv.DictReader(CACHE_SOURCE.open()):
        if row["cond"] != "LEGACY" or row["dir"] not in takes:
            continue
        segment_id = f"{row['dir']}:note:{int(row['note_pos']):03d}"
        segment = segments[segment_id]
        if int(row["note_index"]) != int(segment["event_id"]):
            raise ValueError("Cache/event identity mismatch")
        if int(row["seg_len"]) != int(segment["end_frame"]) - int(segment["start_frame"]):
            raise ValueError("Cache/audio window mismatch")
        values = {metric: float(row[metric]) for metric in HEADS + ["composite"]}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
            raise ValueError("Nonfinite or out-of-range classifier cache output")
        if abs(composite(row, float(row["dur_s"])) - values["composite"]) > 1e-12:
            raise ValueError("Composite formula mismatch")
        historical = original[(row["dir"], row["ckpt"])][int(row["note_index"])]
        for metric, cache_name in original_names.items():
            difference = abs(values[metric] - historical[cache_name])
            max_difference = max(max_difference, difference)
            if difference > 5.00001e-5:
                raise ValueError(f"Head cache mismatch {segment_id} {row['ckpt']} {metric}")
            checked += 1
        key = (segment_id, row["ckpt"])
        if key in keys:
            raise ValueError("Duplicated cached note/checkpoint row")
        keys.add(key)
        for metric, value in values.items():
            scores.append({"analysis": "OUR_REFERENCE_ONLY", "segment_id": segment_id, "take_id": row["dir"],
                           "arm": segment["arm"], "piece": segment["piece"], "seed": segment["seed"],
                           "pair_id": segment["pair_id"], "session_id": segment["session_id"],
                           "model": row["ckpt"], "metric": metric, "value": value, "unit": "0_to_1",
                           "status": "verified_cache", "reason": "same_original_bow_window_and_LEGACY_model_inputs",
                           "cache_source": str(CACHE_SOURCE), "source_condition": "LEGACY"})
    expected = 2 * protocol["n_our_bow_windows"]
    if len(keys) != expected:
        raise ValueError(f"Incomplete cache cohort: {len(keys)} != {expected}")
    for arm, filename in [("Human", "sound_player.wav"), ("Vienna A", "sound_caseA.wav"), ("Vienna B", "sound_caseB.wav")]:
        if (ROOT / "raw" / filename).exists():
            continue
        for model_name, metrics in [("multilength", HEADS + ["composite"]), ("human_A1_A5", HEADS + ["composite"]), ("PAM", ["PAM"]), ("Audiobox", ["PQ", "PC"])]:
            for metric in metrics:
                scores.append({"analysis": "V1_PENDING", "segment_id": "", "take_id": filename,
                               "arm": arm, "piece": "Spring", "seed": "", "pair_id": "", "session_id": "not_available",
                               "model": model_name, "metric": metric, "value": "", "unit": "0_to_1" if model_name != "Audiobox" else "1_to_10",
                               "status": "missing_source", "reason": "Zenodo_download_failed_no_local_copy",
                               "cache_source": "", "source_condition": "not_computed"})
    probe_path = ROOT / "PUBLIC_SCORES.json"
    if not probe_path.exists():
        probe_path = ROOT / "PUBLIC_PROBE.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        if probe["protocol_sha256"] != digest(protocol_path):
            raise ValueError("Public probe used another protocol lock")
        for result in probe["results"]:
            take = takes[result["take_id"]]
            for metric, value in result["scores"].items():
                scores.append({"analysis": result["analysis"], "segment_id": result["segment_id"],
                               "take_id": result["take_id"], "arm": take["arm"], "piece": take["piece"], "seed": take["seed"],
                               "pair_id": take["pair_id"], "session_id": take["session_id"], "model": result["model"],
                               "metric": metric, "value": value, "unit": "0_to_1" if metric == "PAM" else "1_to_10",
                               "status": "new_public_inference", "reason": "not_external_or_matched_V2_evidence",
                               "cache_source": str(probe_path), "source_condition": "original_full_take_native_public_preprocessing"})
        for error in probe["errors"]:
            affected = [error["take_id"]] if error.get("take_id") else list(takes)
            for take_id in affected:
                take = takes[take_id]
                for metric in (["PQ", "PC"] if error["model"] == "Audiobox" else ["PAM"]):
                    scores.append({"analysis": "OUR_UNMATCHED_OVERVIEW", "segment_id": f"{take_id}:full",
                                   "take_id": take_id, "arm": take["arm"], "piece": take["piece"], "seed": take["seed"],
                                   "pair_id": take["pair_id"], "session_id": take["session_id"], "model": error["model"],
                                   "metric": metric, "value": "", "unit": "0_to_1" if metric == "PAM" else "1_to_10",
                                   "status": "failed", "reason": error["message"], "cache_source": str(probe_path),
                                   "source_condition": "original_full_take_native_public_preprocessing"})
    write_csv(ROOT / "SCORES.csv", scores)
    write_json(ROOT / "CACHE_CHECKS.json", {"protocol_sha256": digest(protocol_path), "unique_note_checkpoint_rows": len(keys),
                                           "metric_values_reused": len(keys) * 7, "original_rounded_head_comparisons": checked,
                                           "max_difference_to_4dp_cache": max_difference, "condition": "LEGACY",
                                           "PEAK_values_used": 0, "n_our_recordings": len(takes),
                                           "external_metric_values_computed": 0,
                                           "new_public_values": sum(row["status"] == "new_public_inference" for row in scores)})
    print(json.dumps({"cached_note_model_rows": len(keys), "metric_rows": len(scores), "comparisons": checked}, indent=2))


if __name__ == "__main__":
    main()
