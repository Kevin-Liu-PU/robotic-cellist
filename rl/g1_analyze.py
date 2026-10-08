import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.g1_session import ORDERS, digest, read_json, records

HEADS = ("overall", "tone_quality", "attack_quality", "release_quality", "bow_control", "dynamic_accuracy")


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_adjusted(value, left, right):
    if not left["midpoint"] < value["midpoint"] < right["midpoint"]:
        raise ValueError("baseline midpoint order invalid")
    weight = (value["midpoint"] - left["midpoint"]) / (right["midpoint"] - left["midpoint"])
    reference = (1 - weight) * left["value"] + weight * right["value"]
    return {"delta": value["value"] - reference, "reference": reference,
            "right_weight": weight, "baseline_change": right["value"] - left["value"]}


def cohort(session):
    rows = [row for row in records(Path(session) / "journal.jsonl") if row.get("status") == "valid" and row.get("id", "").startswith("eval_")]
    expected = {(seed, order_index, slot): arm for seed in range(3) for order_index, order in enumerate(ORDERS) for slot, arm in enumerate(order)}
    if len(rows) != 30 or len({row["take_dir"] for row in rows}) != 30:
        raise ValueError("need exactly 30 distinct valid takes; incomplete/failed cohorts are reported, not silently dropped")
    if {(row["seed"], row["order"], row["slot"]): row["arm"] for row in rows} != expected:
        raise ValueError("arm mapping differs from fixed schedule")
    for row in rows:
        for relative, expected_hash in row["raw_hashes"].items():
            if digest(Path(row["take_dir"]) / relative) != expected_hash:
                raise ValueError("recorded take changed")
    return rows


def task_measures(row):
    directory = Path(row["take_dir"])
    performance_file = next(directory.glob("*_performance.json"))
    performance = read_json(performance_file)
    state = np.load(str(performance_file).removesuffix("_performance.json") + "_state.npy", allow_pickle=True)
    midpoint = (float(state[0]["t"]) + float(state[-1]["t"])) / 2
    audit = read_json(directory / "ep01_strokes.json")
    strokes = [record["info"]["stroke"] for record in audit["records"] if record.get("event") == "result"]
    snapshot = read_json(directory / "runtime_snapshot.json")
    constants = snapshot["runtime_constants"]["rl.piece_env"]["values"]
    calibration = json.loads(snapshot["calibration_file"]["text"])
    speed_variation = [sum(abs(right["speed"] - left["speed"]) for left, right in zip(stroke["commanded_segments"], stroke["commanded_segments"][1:])) for stroke in strokes]
    depth = [segment["depth_m"] for stroke in strokes for segment in stroke["commanded_segments"]]
    dynamics = [stroke["r_dynamic"] for stroke in strokes if stroke.get("zone") is not None]
    if len(dynamics) != 14:
        raise ValueError("loudness-band endpoint unavailable: cannot substitute speed proxy")
    result = {
        "speed_total_variation_mps": float(np.mean(speed_variation)),
        "envelope_cost_unweighted": float(np.mean([-stroke["r_envelope"] for stroke in strokes])),
        "loudness_zone_fraction_runtime": float(np.mean([value >= .999 for value in dynamics])),
        "loudness_zone_fraction_written": float(np.mean([
            calibration["zones"][stroke["zone"]]["db_lo"] <= stroke["dbfs"] - calibration["gain_offset_db"] <= calibration["zones"][stroke["zone"]]["db_hi"]
            for stroke in strokes])),
        "absolute_onset_error_s": float(np.mean([abs(note["actual_onset"] - note["scheduled_onset"]) for note in performance["timeline"]])),
        "duration_ratio": float(np.mean([note["actual_duration"] / note["planned_duration"] for note in performance["timeline"]])),
        "commanded_depth_excursion_fraction": float(np.mean([value < constants["DEPTH_LO"] or value > constants["DEPTH_HI"] for value in depth])),
        "bow_shortfall_m": float(np.mean([stroke["shortfall"] for stroke in strokes])),
    }
    return midpoint, result


def score_take(row, judge):
    from rl import posthoc_score
    captured = []

    class CaptureHeads:
        def score_detailed(self, *arguments, **keywords):
            detail = judge.score_detailed(*arguments, **keywords)
            captured.append({head: float(detail[head]) for head in HEADS})
            return detail

    previous = posthoc_score._JUDGE
    posthoc_score._JUDGE = CaptureHeads()
    try:
        performance = next(Path(row["take_dir"]).glob("*_performance.json"))
        result = posthoc_score.score_stem(Path(str(performance).removesuffix("_performance.json")))
    finally:
        posthoc_score._JUDGE = previous
    if result is None or result["n"] != 14 or result["skipped"] or len(captured) != 14:
        raise ValueError("missing scored bows; do not silently change the analysis population")
    return result, captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    options = parser.parse_args()
    session = Path(options.session)
    review = read_json(session / "review_lock.json")
    if review.get("status") != "G1_REVIEW_PASS":
        raise ValueError("analysis requires the same reviewed session lock")
    for relative, expected_hash in review["runtime_files"].items():
        if digest(ROOT / relative) != expected_hash:
            raise ValueError(f"reviewed scoring input/source changed: {relative}")
    rows = cohort(session)
    output = session / "analysis"
    output.mkdir(exist_ok=False)
    (output / "failures_and_attempts.json").write_text(json.dumps(records(session / "journal.jsonl"), indent=2) + "\n")
    metric_rows = []
    note_rows = []
    for row in rows:
        midpoint, measures = task_measures(row)
        row["midpoint"] = midpoint
        for metric, value in measures.items():
            metric_rows.append({"seed": row["seed"], "order": row["order"], "slot": row["slot"], "arm": row["arm"],
                                "judge": "task", "metric": metric, "value": value, "midpoint": midpoint, "take_dir": row["take_dir"]})
    from rl.piece_env import RealScorer
    for judge_name in ("multilength", "human_A1_A5"):
        checkpoint = ROOT / "SoundClassifier/checkpoints" / f"quality_cnn_{judge_name}.pt"
        judge = RealScorer(checkpoint_path=checkpoint, device="cpu")
        for row in rows:
            result, heads = score_take(row, judge)
            for note_index, (note, detail) in enumerate(zip(result["per_note"], heads)):
                note_rows.append({"seed": row["seed"], "order": row["order"], "slot": row["slot"], "arm": row["arm"],
                                  "judge": judge_name, "checkpoint_sha256": digest(checkpoint), "note": note_index,
                                  "composite_legacy_4dp": note["tone"], **detail, "take_dir": row["take_dir"]})
            measures = {head: float(np.mean([detail[head] for detail in heads])) for head in HEADS}
            measures["composite_legacy_4dp"] = result["mean_tone"]
            for metric, value in measures.items():
                metric_rows.append({"seed": row["seed"], "order": row["order"], "slot": row["slot"], "arm": row["arm"],
                                    "judge": judge_name, "metric": metric, "value": value,
                                    "midpoint": row["midpoint"], "take_dir": row["take_dir"]})
        del judge
    csv_write(output / "notes.csv", note_rows)
    csv_write(output / "takes.csv", metric_rows)
    contrasts = []
    for seed in range(3):
        for order_index, order in enumerate(ORDERS):
            for judge_name, metric in sorted({(row["judge"], row["metric"]) for row in metric_rows}):
                block = {row["slot"]: row for row in metric_rows if row["seed"] == seed and row["order"] == order_index and row["judge"] == judge_name and row["metric"] == metric}
                gains = {}
                for slot in (1, 3):
                    gains[order[slot]] = baseline_adjusted(block[slot], block[slot - 1], block[slot + 1])
                    contrasts.append({"seed": seed, "order": order_index, "judge": judge_name, "metric": metric,
                                      "contrast": order[slot] + "-Base", **gains[order[slot]]})
                contrasts.append({"seed": seed, "order": order_index, "judge": judge_name, "metric": metric,
                                  "contrast": "OFF-FULL_adjusted", "delta": gains["OFF"]["delta"] - gains["FULL"]["delta"],
                                  "reference": None, "right_weight": None, "baseline_change": None})
                contrasts.append({"seed": seed, "order": order_index, "judge": judge_name, "metric": metric,
                                  "contrast": "FULL-OFF_raw", "delta": block[order.index("FULL")]["value"] - block[order.index("OFF")]["value"],
                                  "reference": None, "right_weight": None, "baseline_change": None})
    csv_write(output / "contrasts.csv", contrasts)
    seed_rows = []
    for seed, judge_name, metric, contrast_name in sorted({(row["seed"], row["judge"], row["metric"], row["contrast"]) for row in contrasts}):
        values = [row["delta"] for row in contrasts if (row["seed"], row["judge"], row["metric"], row["contrast"]) == (seed, judge_name, metric, contrast_name)]
        seed_rows.append({"seed": seed, "judge": judge_name, "metric": metric, "contrast": contrast_name,
                          "mean_two_orders": float(np.mean(values)), "n_orders": len(values)})
    csv_write(output / "seeds.csv", seed_rows)
    summaries = []
    for judge_name, metric, contrast_name in sorted({(row["judge"], row["metric"], row["contrast"]) for row in seed_rows}):
        for label, selected_seeds in (("all_three_historical_controls", (0, 1, 2)), ("matched_transition_budget_sensitivity", (0, 1))):
            values = [row["mean_two_orders"] for row in seed_rows if row["seed"] in selected_seeds and
                      (row["judge"], row["metric"], row["contrast"]) == (judge_name, metric, contrast_name)]
            summaries.append({"view": label, "judge": judge_name, "metric": metric, "contrast": contrast_name,
                              "n_seeds": len(values), "mean": float(np.mean(values)),
                              "minimum": min(values), "maximum": max(values)})
    csv_write(output / "summary.csv", summaries)
    cohort(session)
    print(f"G1 LEGACY dual judge: 30 takes, 420 bows, 840 bow-by-judge rows -> {output}")


if __name__ == "__main__":
    main()
