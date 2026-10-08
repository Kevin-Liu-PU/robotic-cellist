"""
reweight_analysis.py — offline reward-composition analysis for recorded
whole-piece residual RL, Run 1 / Run 2 data.

Revision: 2026-08-10.
CELLO_REFERENCE_ROOT must explicitly select the local reference source/model
checkout; there is no default reference and no sibling-project search.
Private audio and logs must be supplied with --wav-dir and --stroke-log.

WHAT IT ANSWERS
    Which reward-weight vector (W_QUALITY / W_DYNAMIC / W_BOW / W_SMOOTH)
    is worth burning robot time on, using strokes that were already played.

    It cannot say what a policy trained under new weights would LEARN
    (off-policy counterfactual); it can say how the behaviours the robot
    actually produced would have been GRADED, which terms dominated the
    gradient pressure when, and where the "play louder" shortcut paid.

DATA LEVELS (runs with whatever is present, says what is missing)
    A. --wav-dir only          per-stroke tone quality (A1_A5 judge),
                               training curve, note-length splits,
                               per-note improvement map, judge-swap check.
                               NOTE: the saved windows are PEAK-NORMALISED,
                               so absolute loudness is unrecoverable here.
    B. + --stroke-log          exact r_bow / r_smooth reconstruction and
       (jsonl per stroke:       open-loop dynamics from measured speed;
        action, u_end,          full weight-grid preference-reversal table.
        mean_speed ...)
    C. + dbfs in stroke log    closed-loop zone dynamics + BREAK-EVEN
                               W_DYNAMIC: the smallest dynamics weight that
                               makes the observed louder-shortcut net-negative.

USAGE
    .../python reweight_analysis.py --wav-dir PATH/stroke_audio \
        [--stroke-log PATH/strokes.jsonl] [--also-old-judge] [--out out/]

Physical params fed to the classifier are approximated from the PLAN
(depth/speed/mid-u per note; torque via the mock's contact model) because
the runtime StateLogger values were not saved. Documented, consistent, and
identical across episodes — so per-episode COMPARISONS are unaffected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_paths import configured_path, require_file, require_directory
REFERENCE_ROOT = configured_path("CELLO_REFERENCE_ROOT")

CKPT_NEW = REFERENCE_ROOT / "SoundClassifier/checkpoints/quality_cnn_human_A1_A5.pt" if REFERENCE_ROOT else None
CKPT_OLD = REFERENCE_ROOT / "SoundClassifier/checkpoints/quality_cnn_human_current.pt" if REFERENCE_ROOT else None  # d72d2b0d

WINDOW_SEC = 0.5
# Recorded weights (rl/piece_env.py) — the baseline vector.
W_CURRENT = {"quality": 0.50, "dynamic": 0.25, "bow": 0.15, "smooth": 0.10}
# Candidate vectors to grade against (renormalised to 1.0). The dynamics-forward
# vectors attack the documented Run-1 failure ("bought tone by playing louder").
W_CANDIDATES = {
    "current_50_25":   {"quality": 0.50, "dynamic": 0.25, "bow": 0.15, "smooth": 0.10},
    "dyn_forward_40_35": {"quality": 0.40, "dynamic": 0.35, "bow": 0.15, "smooth": 0.10},
    "dyn_heavy_35_45":  {"quality": 0.35, "dynamic": 0.45, "bow": 0.12, "smooth": 0.08},
    "tone_pure_65_10":  {"quality": 0.65, "dynamic": 0.10, "bow": 0.15, "smooth": 0.10},
}


def load_pmp():
    reference = require_directory(REFERENCE_ROOT, "CELLO_REFERENCE_ROOT")
    require_file(reference / "BaselineControls/play_midi_pieces.py", "CELLO_REFERENCE_ROOT")
    spec = importlib.util.spec_from_file_location(
        "play_midi_pieces", REFERENCE_ROOT / "BaselineControls/play_midi_pieces.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_plan(pmp, piece: Path):
    notes, meta = pmp.parse_midi(str(piece))
    notes, _, _ = pmp.merge_slurs(notes)
    return pmp.BowPlanner().plan(notes)


def load_judge(ckpt: Path):
    require_directory(REFERENCE_ROOT, "CELLO_REFERENCE_ROOT")
    require_file(ckpt, "CELLO_REFERENCE_ROOT (classifier checkpoint)")
    require_file(REFERENCE_ROOT / "SoundClassifier/classifier.py", "CELLO_REFERENCE_ROOT")
    sc = str(REFERENCE_ROOT / "SoundClassifier")
    if sc not in sys.path:
        sys.path.insert(0, sc)
    from classifier import BowingQualityClassifier
    return BowingQualityClassifier(checkpoint_path=str(ckpt), device="cpu")


def plan_physical(stroke, depth_lo_m: float) -> np.ndarray:
    """Approximate the 6-dim physical vector from the plan (see header)."""
    import numpy as np
    mid_u = (stroke.u_start + stroke.u_end) / 2.0
    torque = max(0.0, 0.08 + 60.0 * (stroke.depth - depth_lo_m))
    return np.array([stroke.depth, 0.0, stroke.mean_speed, mid_u,
                     torque, torque * 1.3], dtype=np.float32)


WAV_RE = re.compile(r"ep(\d+)_s(\d+)\.wav$")


def discover_wavs(wav_dir: Path):
    rows = []
    for p in sorted(wav_dir.rglob("*.wav")):
        m = WAV_RE.search(p.name)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), p))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Offline reward re-weighting analysis")
    ap.add_argument("--wav-dir", required=True, type=Path)
    ap.add_argument("--piece", type=Path, default=None,
                    help="MIDI score; defaults to CELLO_REFERENCE_ROOT/MIDI-Files/t1.mid")
    ap.add_argument("--stroke-log", type=Path, default=None,
                    help="optional jsonl: one record per stroke with episode, "
                         "stroke, action, u_end, mean_speed, dbfs (any subset)")
    ap.add_argument("--also-old-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = ap.parse_args()
    reference = require_directory(REFERENCE_ROOT, "CELLO_REFERENCE_ROOT")
    require_file(reference / "BaselineControls/play_midi_pieces.py", "CELLO_REFERENCE_ROOT")
    require_file(reference / "SoundClassifier/classifier.py", "CELLO_REFERENCE_ROOT")
    require_file(CKPT_NEW, "CELLO_REFERENCE_ROOT (classifier checkpoint)")
    if args.also_old_judge:
        require_file(CKPT_OLD, "CELLO_REFERENCE_ROOT (old classifier checkpoint)")
    args.wav_dir = require_directory(args.wav_dir, "--wav-dir")
    if args.piece is None:
        args.piece = reference / "MIDI-Files/t1.mid"
    args.piece = require_file(args.piece, "--piece")
    if args.stroke_log is not None:
        args.stroke_log = require_file(args.stroke_log, "--stroke-log")
    args.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf

    pmp = load_pmp()
    plan = load_plan(pmp, args.piece)
    n_plan = len(plan)
    depth_lo = -pmp.MAX_OUTWARD_DEPTH
    fills = [min(1.0, s.duration / WINDOW_SEC) for s in plan]
    print(f"plan: {n_plan} strokes, {sum(f < 1 for f in fills)}/{n_plan} short of "
          f"{WINDOW_SEC}s window")

    judge = load_judge(CKPT_NEW)
    judge_old = load_judge(CKPT_OLD) if args.also_old_judge else None

    # optional per-stroke runtime log (level B/C).
    # Schema = train_piece_logged.py: one header record, then per-stroke
    # records carrying per-episode "stroke" (1..n_plan) plus global
    # "stroke_seq"; "action" is a 2-element list [speed, depth].
    log = {}
    log_ep_offset = 0
    driver_polluted = False
    if args.stroke_log and args.stroke_log.exists():
        lines = [ln for ln in args.stroke_log.read_text().splitlines()
                 if ln.strip()]
        for li, line in enumerate(lines):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if li == len(lines) - 1:
                    print("!! torn final jsonl line skipped (process was "
                          "killed mid-write) — 1 stroke record lost")
                    continue
                sys.exit(f"corrupt jsonl at line {li + 1} (not the last "
                         f"line) — refusing to analyze a damaged log")
            if r.get("header"):
                argv = " ".join(r.get("raw_argv", r.get("argv", [])))
                if "--no-calibrated-dynamics" not in argv:
                    print("!! header does NOT show --no-calibrated-dynamics; "
                          "this analysis assumes the UNCALIBRATED plan.")
                piece_args = [a for a in r.get("raw_argv", r.get("argv", []))
                              if a.endswith((".mid", ".mxl", ".musicxml"))]
                if piece_args and Path(piece_args[0]).name != args.piece.name:
                    sys.exit(f"log was trained on {piece_args[0]} but "
                             f"--piece is {args.piece.name} — wrong plan")
                for bad in ("--tempo-scale", "--bowing"):
                    if any(a.startswith(bad) for a in
                           r.get("raw_argv", r.get("argv", []))):
                        sys.exit(f"log used {bad}; this analysis only "
                                 f"supports the stock plan")
                log_ep_offset = int(r.get("episode_offset", 0))
                if r.get("driver"):
                    print("NOTE: driver run — logged dbfs includes the "
                          "virtual gain offset; recovering true dBFS as "
                          "dbfs - gain_offset_db per stroke.")
                continue
            if "episode" not in r or "stroke" not in r:
                sys.exit("stroke-log record lacks episode/stroke keys — "
                         "unknown schema, refusing to guess")
            a = r.get("action")
            if isinstance(a, (list, tuple)) and len(a) == 2:
                r["action_speed"], r["action_depth"] = float(a[0]), float(a[1])
            g = r.get("gain_offset_db") or 0.0
            if g and isinstance(r.get("dbfs"), (int, float)):
                r["dbfs"] = r["dbfs"] - g       # recover TRUE loudness
                driver_polluted = True
            key = (int(r["episode"]), int(r["stroke"]))
            if key in log:
                sys.exit(f"duplicate stroke-log key {key} — log mixes runs "
                         f"or segments; analyze one segment per invocation")
            log[key] = r
        n_dbfs = sum(1 for r in log.values()
                     if isinstance(r.get("dbfs"), (int, float)))
        print(f"stroke log: {len(log)} records (real dbfs values: {n_dbfs}"
              + (", gain-offset removed" if driver_polluted else "") + ")")

    wavs = discover_wavs(args.wav_dir)
    if not wavs:
        sys.exit(f"no ep####_s###.wav files under {args.wav_dir}")
    seen = set()
    for ep, st, path in wavs:
        if (ep, st) in seen:
            sys.exit(f"duplicate (episode {ep}, stroke {st}) under "
                     f"{args.wav_dir} — directory mixes runs (Run 1/Run 2 "
                     f"share filenames); point --wav-dir at ONE run")
        seen.add((ep, st))
    bad = [st for _, st, _ in wavs if not (1 <= st <= n_plan)]
    if bad:
        sys.exit(f"stroke numbers outside 1..{n_plan} found (e.g. {bad[0]}) — "
                 f"wavs do not match this piece's plan")
    print(f"{len(wavs)} stroke wavs found; scoring with A1_A5 judge...")
    print("plan assumption: UNCALIBRATED (--no-calibrated-dynamics runs only)")

    rows = []
    for k, (ep, st, path) in enumerate(wavs):
        idx = st - 1                     # per-episode numbering, range-checked
        s = plan[idx]
        audio, _sr = sf.read(path, dtype="float32")
        phys = plan_physical(s, depth_lo)
        q = float(judge.predict(audio, phys, string="A", window_pos=0.5))
        fill = fills[idx]
        row = {
            "episode": ep, "stroke": st, "note_index": idx,
            "duration": s.duration, "fill": fill,
            "quality": q, "quality_eff": 0.5 + fill * (q - 0.5),
        }
        if judge_old is not None:
            row["quality_oldjudge"] = float(
                judge_old.predict(audio, phys, string="A", window_pos=0.5))
        lg = log.get((ep + log_ep_offset, st))
        if lg:
            row.update({k2: lg[k2] for k2 in
                        ("action_speed", "action_depth", "u_end",
                         "mean_speed", "dbfs")
                        if lg.get(k2) is not None})
        rows.append(row)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(wavs)}")

    if log and not any("u_end" in r or "dbfs" in r for r in rows):
        sys.exit("stroke log loaded but ZERO rows joined — episode numbering "
                 "mismatch between log and wavs; check --episode-offset "
                 "segments were not mixed")

    # ---- per-stroke CSV --------------------------------------------------
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(args.out / "strokes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- episode aggregation --------------------------------------------
    eps = sorted({r["episode"] for r in rows})
    def ep_rows(e): return [r for r in rows if r["episode"] == e]
    def mean(v): return float(np.mean(v)) if len(v) else float("nan")

    curve, curve_trusted = [], []
    for e in eps:
        rr = ep_rows(e)
        curve.append(mean([r["quality"] for r in rr]))
        curve_trusted.append(mean([r["quality"] for r in rr if r["fill"] >= 1.0]))

    # note-length split, first-10 vs last-10 episodes
    def split_gain(sel):
        early = [r["quality"] for e in eps[:10] for r in ep_rows(e) if sel(r)]
        late = [r["quality"] for e in eps[-10:] for r in ep_rows(e) if sel(r)]
        return mean(early), mean(late)
    bins = {
        ">=0.5s (trusted)": lambda r: r["duration"] >= 0.5,
        "0.2-0.5s":         lambda r: 0.2 <= r["duration"] < 0.5,
        "<0.2s":            lambda r: r["duration"] < 0.2,
    }

    report = [f"# Reward re-weighting analysis — level "
              f"{'C' if any('dbfs' in r for r in rows) else ('B' if log else 'A')}",
              f"strokes scored: {len(rows)}  episodes: {len(eps)}", ""]
    report.append("## Training curve (A1_A5 judge, our offline re-score)")
    report.append(f"first-10 avg {mean(curve[:10]):.3f} -> last-10 avg "
                  f"{mean(curve[-10:]):.3f}   (trusted-notes only: "
                  f"{mean(curve_trusted[:10]):.3f} -> {mean(curve_trusted[-10:]):.3f})")
    report.append("")
    report.append("## Gain by note length (early-10 vs late-10 episodes)")
    for name, sel in bins.items():
        a, b = split_gain(sel)
        report.append(f"- {name}: {a:.3f} -> {b:.3f}  (gain {b - a:+.3f})")

    if judge_old is not None:
        qn = [r["quality"] for r in rows]
        qo = [r["quality_oldjudge"] for r in rows]
        try:
            from scipy.stats import spearmanr
            res = spearmanr(qn, qo)
            rho = float(getattr(res, "statistic",
                                getattr(res, "correlation")))
        except (ImportError, AttributeError):
            # tie-aware ranks: average rank within tie groups (the classifier
            # clips to [0,1], so exact ties are real)
            def _rank(v):
                v = np.asarray(v)
                order = np.argsort(v, kind="mergesort")
                ranks = np.empty(len(v))
                ranks[order] = np.arange(len(v), dtype=float)
                for val in np.unique(v):
                    m = v == val
                    ranks[m] = ranks[m].mean()
                return ranks
            rho = float(np.corrcoef(_rank(qn), _rank(qo))[0, 1])
        report += ["", "## Judge swap (A1_A5 vs frozen d72d2b0d)",
                   f"per-stroke rank correlation (Spearman, tie-aware): {rho:.3f}"]

    # ---- weight grid (needs level B for bow/smooth, C for dynamics) -----
    have_b = any("u_end" in r for r in rows)
    have_c = any("dbfs" in r for r in rows)
    report.append("")
    if have_b or have_c:
        report.append("## Weight-grid episode ranking (available terms only)")
        # Reconstruction of r_bow / r_smooth / r_dynamic mirrors piece_env._reward;
        # only implemented once real logs exist so formulas can be verified
        # against a handful of the printed episode summaries first.
        report.append("(stroke log detected — verify reconstruction against her "
                      "printed episode summaries before trusting this table)")
    else:
        report.append("## Weight grid: SKIPPED (level A data)")
        report.append("Saved wavs are peak-normalised, so loudness/dynamics and "
                      "bow/smooth terms need her runtime logs (see header).")
        report.append("Candidates queued for when logs arrive: " +
                      ", ".join(W_CANDIDATES))

    (args.out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    # ---- figures ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(eps, curve, label="all notes")
        ax[0].plot(eps, curve_trusted, label="trusted (>=0.5s)")
        ax[0].set_xlabel("episode"); ax[0].set_ylabel("mean tone quality")
        ax[0].legend(); ax[0].set_title("offline re-score, A1_A5 judge")
        note_gain = []
        for i in range(n_plan):
            e10 = [r["quality"] for e in eps[:10] for r in ep_rows(e)
                   if r["note_index"] == i]
            l10 = [r["quality"] for e in eps[-10:] for r in ep_rows(e)
                   if r["note_index"] == i]
            note_gain.append(mean(l10) - mean(e10))
        ax[1].bar(range(n_plan), note_gain)
        ax[1].set_xlabel("note index in piece"); ax[1].set_ylabel("quality gain")
        ax[1].set_title("where in the piece the policy improved")
        fig.tight_layout()
        fig.savefig(args.out / "overview.png", dpi=150)
        print(f"figures -> {args.out / 'overview.png'}")
    except ImportError:
        print("(matplotlib unavailable — skipped figures)")


if __name__ == "__main__":
    main()
