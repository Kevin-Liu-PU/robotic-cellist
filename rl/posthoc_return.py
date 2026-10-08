"""
rl/posthoc_return.py

Offline composite-return scoring of recorded A/B takes: what total reward
R_t would the training env have paid for this take?

Mechanism (no reward formula is reimplemented):
  1. REPLAY the deterministic compile rollout (same checkpoint, NeutralScorer,
     mock executor, seed 0) to recover the exact per-stroke residual actions,
     exec strokes and shortfalls that produced the take.
  2. For each stroke, call the env's own _reward() with a StrokeResult
     rebuilt from the TAKE's recorded data: audio slice at the executed
     timeline, measured speed/torque from the state log, dBFS from the
     recorded slice. Every term is priced by env code.

Alignment guard: the replayed exec stroke's planned duration must match the
take timeline's planned_duration per stroke (catches tempo/piece mismatch).

Condition flags must match how the cell was TRAINED (they change the reward):
  --alt        sets CELLO_ALT_LOOP=1 before the env import
  (default)    stock loop
Gain-offset caveat: r_dynamic grades absolute dBFS against zones through
loudness_model.json's calibration. Use --gain-shift DB for sensitivity
sweeps when the recording session's offset is not the one in the repo.

Usage:
  .venv/bin/python rl/posthoc_return.py --piece twinkle-short.mxl \
      [--alt] [--checkpoint CKPT | --baseline] [--gain-shift DB] DIR ...


"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Parse BEFORE importing piece_env: the ALT flags are read at import time.
ap = argparse.ArgumentParser(description="offline composite-return scoring")
ap.add_argument("dirs", nargs="+", metavar="[LABEL=]PERFORM_DIR")
ap.add_argument("--piece", required=True)
ap.add_argument("--checkpoint", default=None,
                help="SAC zip that produced the policy takes")
ap.add_argument("--baseline", action="store_true",
                help="score dirs as zero-residual baseline takes")
ap.add_argument("--alt", action="store_true",
                help="ALT loop reward (must match how the cell trained)")
ap.add_argument("--gain-shift", type=float, default=0.0,
                help="add DB to measured dBFS before zone grading "
                     "(sensitivity sweeps)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
if args.baseline == (args.checkpoint is not None):
    sys.exit("exactly one of --baseline / --checkpoint")
if args.alt:
    os.environ["CELLO_ALT_LOOP"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from rl.piece_env import (ACTION_DIM, PieceResidualEnv, RealScorer,  # noqa: E402
                          StrokeResult)
from rl.loudness import measure_dbfs  # noqa: E402
from rl.perform import CapturingExecutor, NeutralScorer  # noqa: E402


def replay_actions(piece: str, ckpt: str | None, seed: int):
    """Deterministic compile replay -> [(plan, action, exec, shortfall)]."""
    env = PieceResidualEnv(piece_path=piece, executor=CapturingExecutor(),
                           scorer=NeutralScorer(), calibrated_dynamics=True)
    model = None
    if ckpt is not None:
        from stable_baselines3 import SAC
        model = SAC.load(ckpt, env=None, device="cpu")
    obs, _ = env.reset(seed=seed)
    rows, done = [], False
    while not done:
        if model is None:
            action = np.zeros(ACTION_DIM, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        plan = env.plan[env._i]
        exec_stroke, shortfall = env._build_exec_stroke(plan, action)
        obs, _r, done, _tr, _info = env.step(action)
        rows.append((plan, np.asarray(action, dtype=np.float32),
                     exec_stroke, shortfall))
    env.close()
    return rows


def load_take(d: Path):
    perfs = sorted(d.glob("*_performance.json"),
                   key=lambda p: p.stat().st_mtime)
    perf = json.loads(perfs[-1].read_text())
    stem = str(perfs[-1])[: -len("_performance.json")]
    audio, sr = sf.read(stem + ".wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    S = np.load(stem + "_state.npy", allow_pickle=True)
    T = np.array([row["t"] for row in S])
    return perf, audio, sr, S, T


def main() -> None:
    replay = replay_actions(args.piece, args.checkpoint, args.seed)

    # The reward env: real classifier, loudness model, same loop flags.
    env = PieceResidualEnv(piece_path=args.piece, executor=CapturingExecutor(),
                           scorer=RealScorer(), calibrated_dynamics=True)
    env.reset(seed=args.seed)

    results = []
    for index, raw in enumerate(args.dirs, 1):
        label, raw_dir = (raw.split("=", 1) if "=" in raw
                          else (f"Run{index}", raw))
        d = Path(raw_dir)
        perf, audio, sr, S, T = load_take(d)
        lead = float(perf.get("audio_lead_in_s", 0.0))
        wall0 = T[0]
        timeline = perf["timeline"]
        if len(timeline) != len(replay):
            sys.exit(f"{d.name}: {len(timeline)} strokes in take vs "
                     f"{len(replay)} replayed — piece/config mismatch")

        totals, comps_sum, n = [], {}, 0
        env._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for i, t in enumerate(timeline):
            plan, action, exec_stroke, shortfall = replay[i]
            if abs(float(t.get("planned_duration", exec_stroke.duration))
                   - exec_stroke.duration) > 0.05:
                sys.exit(f"{d.name}: stroke {i} planned_duration "
                         f"{t.get('planned_duration'):.3f} vs replayed "
                         f"{exec_stroke.duration:.3f} — alignment broken")
            on, dur = float(t["actual_onset"]), float(t["actual_duration"])
            lo = int((lead + on) * sr)
            seg = audio[lo:lo + int(dur * sr)]
            m = (T >= wall0 + on) & (T <= wall0 + on + dur)
            if len(seg) < sr // 20 or not m.any():
                continue
            spd = float(np.mean(np.abs([r["bow_speed"] for r in S[m]])))
            tq = np.sqrt([r["ft_tx"] ** 2 + r["ft_ty"] ** 2 + r["ft_tz"] ** 2
                          for r in S[m]])
            phys = np.array([t["depth_m"], 0.0, spd,
                             (t["u_start"] + t["u_end"]) / 2.0,
                             float(tq.mean()), float(tq.max())],
                            dtype=np.float32)
            raw_dbfs = measure_dbfs(seg)
            dbfs = (raw_dbfs + args.gain_shift) if raw_dbfs is not None else None
            result = StrokeResult(audio=seg, physical=phys,
                                  measured_dbfs=dbfs,
                                  measured_mean_speed=spd,
                                  achieved_u_end=float(exec_stroke.u_end),
                                  segment_profile=None)
            total, comps = env._reward(plan, exec_stroke, result,
                                       shortfall, action)
            env._prev_action = action.copy()
            totals.append(float(total))
            n += 1
            for k, v in comps.items():
                if isinstance(v, (int, float)):
                    comps_sum[k] = comps_sum.get(k, 0.0) + float(v)

        mean_R = float(np.mean(totals))
        comp_means = {k: v / n for k, v in comps_sum.items()}
        results.append((label, mean_R, n, comp_means))
        print(f"{label:8s} [{d.name}]  mean_R={mean_R:+.4f}  n={n}")
        print("   " + "  ".join(
            f"{k}={comp_means[k]:+.3f}" for k in
            ("quality_eff", "r_dynamic", "r_speed_target", "r_timing",
             "r_bow", "r_smooth", "r_defect", "r_speak", "r_onset")
            if k in comp_means))

    by_label = {r[0]: r for r in results}
    pairs = [(b, f"RL{b[4:]}") for b in by_label
             if b.startswith("Base") and f"RL{b[4:]}" in by_label]
    if pairs:
        print("\nPaired composite-return differences (RL - Baseline):")
        diffs = [by_label[p][1] - by_label[b][1] for b, p in pairs]
        per = " ".join(f"{x:+.4f}" for x in diffs)
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        print(f"  Delta R_t  {float(np.mean(diffs)):+.4f} (SD {sd:.4f}, "
              f"n={len(diffs)})  [{per}]")
        keys = ["quality_eff", "r_dynamic", "r_speed_target", "r_timing",
                "r_bow", "r_smooth", "r_defect", "r_speak"]
        print("  component means (RL - Baseline):")
        for k in keys:
            ds = [by_label[p][3].get(k, 0.0) - by_label[b][3].get(k, 0.0)
                  for b, p in pairs]
            print(f"    {k:15s} {float(np.mean(ds)):+.4f}")


if __name__ == "__main__":
    main()
