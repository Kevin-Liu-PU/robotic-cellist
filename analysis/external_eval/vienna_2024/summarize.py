from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "plot_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare import ROOT, digest, write_csv, write_json


def main() -> None:
    scores = pd.read_csv(ROOT / "SCORES.csv")
    segments = pd.read_csv(ROOT / "SEGMENTS.csv")
    pairs = pd.read_csv(ROOT / "PAIRS.csv")
    valid = scores[scores.status.isin(["verified_cache", "new_public_inference"])].copy()
    grouping = ["analysis", "piece", "arm", "model", "metric"]
    take_grouping = grouping + ["take_id", "seed", "pair_id", "session_id"]
    by_take = valid.groupby(take_grouping, dropna=False).agg(score=("value", "mean"), n_segments=("segment_id", "nunique")).reset_index()
    by_take.to_csv(ROOT / "TAKE_SCORES.csv", index=False)
    summaries = []
    for key, group in by_take.groupby(grouping, dropna=False):
        row = dict(zip(grouping, key))
        row.update(n_original_recordings=group.take_id.nunique(), n_segments=int(group.n_segments.sum()),
                   n_sessions=group.session_id.nunique(), n_seed_groups=group.seed.nunique(),
                   mean=float(group.score.mean()), median=float(group.score.median()),
                   sample_sd=float(group.score.std(ddof=1)) if len(group) > 1 else "",
                   minimum=float(group.score.min()), maximum=float(group.score.max()),
                   aggregation_unit="original_recording_mean", status="our_reference_only_not_V1_or_V2")
        summaries.append(row)
    for analysis in ["V1", "V2"]:
        summaries.append({"analysis": analysis, "piece": "not_available", "arm": "not_available", "model": "all", "metric": "all",
                          "n_original_recordings": 0, "n_segments": 0, "n_sessions": "", "n_seed_groups": "",
                          "mean": "", "median": "", "sample_sd": "", "minimum": "", "maximum": "",
                          "aggregation_unit": "not_computed", "status": "external_original_audio_unavailable"})
    write_csv(ROOT / "SUMMARY.csv", summaries)
    pair_values = by_take.pivot(index=["analysis", "piece", "seed", "pair_id", "model", "metric"], columns="arm", values="score")
    pair_values["RL_minus_Base"] = pair_values["Our RL"] - pair_values["Our Base"]
    pair_values.to_csv(ROOT / "OUR_PAIRED_REFERENCE.csv")
    by_seed = by_take.groupby(["analysis", "piece", "seed", "arm", "model", "metric"], dropna=False).score.mean().reset_index()
    by_seed.to_csv(ROOT / "OUR_SEED_REFERENCE.csv", index=False)
    notes = segments[segments.segment_type == "executed_complete_bow_window"]
    pitch = notes.groupby(["piece", "arm"]).agg(n_bows=("segment_id", "size"),
                                                duration_median_s=("duration_s", "median"),
                                                pitch_median_hz=("pitch_hz", "median"),
                                                periodicity_median=("periodicity", "median")).reset_index()
    pitch.to_csv(ROOT / "PITCH_DIAGNOSTICS.csv", index=False)
    downloads = [json.loads(line) for line in (ROOT / "download_attempts.jsonl").read_text().splitlines()]
    final_downloads = {}
    for entry in downloads:
        name = Path(entry["destination"]).name
        final_downloads[name] = {"file": name, "url": entry["url"], "attempts": entry["attempt"],
                                 "last_finished_utc": entry["finished_utc"], "last_http_status": entry["http_status"],
                                 "success": entry["success"], "path": entry["destination"]}
    write_csv(ROOT / "DOWNLOAD_FINAL_STATUS.csv", list(final_downloads.values()))
    probe = json.loads((ROOT / "PUBLIC_PROBE.json").read_text())
    public = json.loads((ROOT / "PUBLIC_SCORES.json").read_text())
    outputs = {(entry["take_id"], entry["model"]): entry for entry in public["results"]}
    probe_differences = [abs(value - outputs[(entry["take_id"], entry["model"])]["scores"][metric])
                         for entry in probe["results"] for metric, value in entry["scores"].items()]
    checks = {
        "n_pairs": int(pairs.pair_id.nunique()), "n_unique_recordings": int(pairs.take_id.nunique()),
        "n_mapping_rows": len(pairs), "n_bow_windows": len(notes),
        "unique_segment_ids": bool(segments.segment_id.is_unique),
        "all_45_pairs_have_both_arms": bool((pairs.groupby("pair_id").arm.nunique() == 2).all()),
        "all_input_hash_checks_pass": all(row["match"] for row in json.loads((ROOT / "INPUT_CHECKS.json").read_text())),
        "all_real_scores_finite": bool(np.isfinite(valid.value).all()),
        "missing_values_are_blank_not_zero": bool(scores[scores.status == "missing_source"].value.isna().all()),
        "model_values_have_unique_keys": not bool(valid.duplicated(["analysis", "segment_id", "model", "metric"]).any()),
        "n_new_public_metric_values": int((valid.status == "new_public_inference").sum()),
        "public_model_errors": public["errors"],
        "max_probe_to_batch_difference": max(probe_differences),
        "external_original_recordings_downloaded": sum((ROOT / "raw" / name).exists() for name in ["sound_player.wav", "sound_caseA.wav", "sound_caseB.wav"]),
        "external_scores_computed": 0,
        "periodic_bows": int((notes.pitch_check == "periodic").sum()),
        "bows_within_100_cents_A3": int((np.abs(1200 * np.log2(notes.pitch_hz / 220)) <= 100).sum()),
    }
    for key in ["unique_segment_ids", "all_45_pairs_have_both_arms", "all_input_hash_checks_pass", "all_real_scores_finite", "missing_values_are_blank_not_zero", "model_values_have_unique_keys"]:
        if not checks[key]:
            raise ValueError(f"Integrity check failed: {key}")
    if checks["n_new_public_metric_values"] != 270 or public["errors"] or checks["max_probe_to_batch_difference"] > 1e-6:
        raise ValueError("Public-model completeness or repeatability check failed")
    write_json(ROOT / "VALIDATION.json", checks)
    colors = {"twinkle": "#31688e", "maysong": "#35b779", "bach": "#8d5a97", "vocalise": "#e69f00", "popper": "#d95f59"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(12, 7.5), layout="constrained")
    grid = figure.add_gridspec(2, 6, height_ratios=[0.8, 2])
    for title, message, selection in [
        ("V1 | Vienna Human / A / B", "NOT COMPUTED\n0 of 3 external WAVs obtained\nZenodo returned HTTP 504", grid[0, :3]),
        ("V2 | Cross-system comparison", "NOT COMPUTED\nNo external matched-pitch/bow inputs\nNo cross-system performance claim", grid[0, 3:]),
    ]:
        axis = figure.add_subplot(selection)
        axis.set_facecolor("#f2f4f7")
        axis.text(0.04, 0.88, title, fontsize=12, weight="bold", transform=axis.transAxes)
        axis.text(0.04, 0.12, message, fontsize=11, transform=axis.transAxes, linespacing=1.5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    for position, metric in enumerate(["PAM", "PQ", "PC"]):
        axis = figure.add_subplot(grid[1, 2 * position:2 * position + 2])
        data = by_seed[(by_seed.metric == metric) & (by_seed.analysis == "OUR_UNMATCHED_OVERVIEW")]
        for piece_index, (piece, color) in enumerate(colors.items()):
            group = data[data.piece == piece].pivot(index="seed", columns="arm", values="score")
            for seed, values in group.iterrows():
                offset = (piece_index - 2) * 0.03 + (seed - 1) * 0.008
                axis.plot([offset, 1 + offset], [values["Our Base"], values["Our RL"]], color=color, alpha=0.45, linewidth=0.8)
                axis.scatter([offset, 1 + offset], [values["Our Base"], values["Our RL"]], color=color, s=30)
            axis.plot([], [], "o", color=color, label=piece)
        axis.set_xticks([0, 1], ["Our Base", "Our RL"])
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylabel("PAM [0, 1]" if metric == "PAM" else f"Audiobox {metric} [1, 10]")
        axis.set_title(f"Our reference only: {metric}" + ("\nComplexity, not quality" if metric == "PC" else ""), fontsize=11)
        axis.grid(axis="y", alpha=0.2)
        if position == 0:
            axis.legend(fontsize=8, loc="best")
    figure.suptitle("VIE01 | Partial execution - no Vienna comparison results", fontsize=16, weight="bold")
    figure.supxlabel("Each dot: mean of 3 full recordings for one piece/seed (15 groups per arm; 45 takes per arm).\nOriginal recording levels, content, durations and silence are not cross-system matched. PQ and PC share one model.", fontsize=9)
    figure.savefig(ROOT / "RESULTS_PARTIAL.png", dpi=180)
    plt.close(figure)
    print(json.dumps(checks, indent=2))
    print("Saved descriptive tables and clearly marked partial-result figure.")


if __name__ == "__main__":
    main()
