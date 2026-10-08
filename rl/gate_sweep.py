"""
Per-stroke judge sensitivity by whole-piece extreme sweeps.

, from his design: instead of
sampling three strokes with random residuals (rl/reward_noise.py), play the
WHOLE piece under a few fixed residuals and read every stroke.

    zero    a = 0                       the planner's nominal
    plus    a = +1 on all six dims      +35% speed, +1.0 mm depth (deep, fast)
    minus   a = -1 on all six dims      -35% speed, -1.0 mm depth (shallow, slow)
    deepslow / shallowfast              the other two corners (optional)

Each condition is an episode; conditions are interleaved round-robin so that
contact drift over the session is shared equally between them. Per stroke i
and reward term t:

    within sd_i  = pooled sd over the repeats of every condition
    range_i      = max - min of the condition means
    SNR_i        = range_i / within sd_i
    sign_i       = mean(plus) - mean(minus)

Every stroke gets a number, so the fill-vs-visibility question is answered
with N = strokes, not N = 3, and the extremes bound what any policy can do.

    CELLO_ENVELOPE_MIN_S=0.5 CELLO_PHRASE_SCORING=1 CELLO_TIMING_PENALTY=1 \
      .venv/bin/python -u rl/gate_sweep.py --piece maysong_m0-11.mxl --real \
      --repeats 4 --no-calibrated-dynamics --out rl/sweep_maysong_m0-11_20260902.json

    .venv/bin/python rl/gate_sweep.py --analyze rl/sweep_maysong_m0-11_20260902.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.reward_noise import build_env, raw_row  # noqa: E402

TERMS = ("quality", "tone", "quality_eff", "total", "r_dynamic", "r_defect")


def condition_vector(name: str, dim: int) -> np.ndarray:
    half = dim // 2                       # [spd x half, depth x half]
    spd, dep = {
        "zero": (0.0, 0.0), "plus": (1.0, 1.0), "minus": (-1.0, -1.0),
        "deepslow": (-1.0, 1.0), "shallowfast": (1.0, -1.0),
        # single-variable conditions (2026-09-02, lab note): separate the depth
        # lever from the speed lever, which plus/minus move together
        "deep": (0.0, 1.0), "shallow": (0.0, -1.0),
        "fast": (1.0, 0.0), "slow": (-1.0, 0.0),
    }[name]
    return np.array([spd] * half + [dep] * (dim - half), dtype=np.float32)


def play_episode(env, action: np.ndarray) -> tuple[list[dict], bool]:
    """Whole piece at one fixed residual. Returns (rows, completed)."""
    rows = []
    try:
        env.reset(seed=0)
        for i in range(len(env.plan)):
            _, reward, term, trunc, info = env.step(action)
            row = raw_row(dict(info["stroke"]), action)
            row["reward"] = float(reward)
            row["stroke_idx"] = i
            rows.append(row)
            if term or trunc:
                if i < len(env.plan) - 1:
                    print(f"      episode ended early at stroke {i}", flush=True)
                    return rows, False
                break
    except Exception as e:  # one dead episode must not lose the session
        print(f"      EPISODE FAILED: {type(e).__name__}: {e}", flush=True)
        return rows, False
    return rows, True


def analyze(data: dict, terms=TERMS) -> dict:
    conds = data["conditions"]
    eps = data["episodes"]
    n_strokes = data["n_strokes"]
    out = {"per_stroke": [], "terms": list(terms)}
    for i in range(n_strokes):
        entry = {"stroke": i}
        by_cond = {c: [e["rows"][i] for e in eps if e["condition"] == c and len(e["rows"]) > i]
                   for c in conds}
        any_row = next((r[0] for r in by_cond.values() if r), None)
        if any_row is None:
            continue
        entry["zone"] = any_row.get("zone")
        entry["fill"] = any_row.get("window_fill")
        entry["n"] = {c: len(v) for c, v in by_cond.items()}
        for t in terms:
            if t not in any_row:
                continue
            means = {c: float(np.mean([r[t] for r in v])) for c, v in by_cond.items() if v}
            # pooled within-condition variance, df-weighted
            num, df = 0.0, 0
            for v in by_cond.values():
                if len(v) >= 2:
                    x = np.array([r[t] for r in v], dtype=float)
                    num += float(x.var(ddof=1)) * (len(x) - 1)
                    df += len(x) - 1
            sd = np.sqrt(num / df) if df > 0 else float("nan")
            rng = max(means.values()) - min(means.values()) if len(means) >= 2 else float("nan")
            snr = rng / sd if sd and sd > 0 else float("inf")
            entry[t] = {"means": means, "within_sd": sd, "range": rng, "snr": snr,
                        "sign": (means.get("plus", np.nan) - means.get("minus", np.nan))}
        out["per_stroke"].append(entry)
    return out


def print_report(data: dict, rep: dict, terms=("quality", "tone", "total")) -> None:
    conds = data["conditions"]
    print(f"\npiece {data['piece']}  strokes {data['n_strokes']}  conditions {conds}  "
          f"episodes {len(data['episodes'])}")
    hdr = f"{'i':>3} {'zone':>4} {'fill':>5}"
    for t in terms:
        hdr += f" | {t:>11}" + "".join(f" {c[:5]:>6}" for c in conds) + f" {'sd':>6} {'SNR':>5}"
    print(hdr)
    for e in rep["per_stroke"]:
        line = f"{e['stroke']:>3} {str(e.get('zone')):>4} {e.get('fill', float('nan')):5.2f}"
        for t in terms:
            if t not in e:
                continue
            d = e[t]
            line += f" | {'':>11}" + "".join(f" {d['means'].get(c, float('nan')):6.2f}" for c in conds)
            line += f" {d['within_sd']:6.3f} {d['snr']:5.2f}"
        print(line)
    print()
    for t in terms:
        snrs = np.array([e[t]["snr"] for e in rep["per_stroke"] if t in e], dtype=float)
        fills = np.array([e["fill"] for e in rep["per_stroke"] if t in e], dtype=float)
        signs = np.array([e[t]["sign"] for e in rep["per_stroke"] if t in e], dtype=float)
        finite = np.isfinite(snrs)
        if finite.sum() == 0:
            continue
        s, f = snrs[finite], fills[finite]
        corr = float(np.corrcoef(s, f)[0, 1]) if len(s) > 2 and f.std() > 0 else float("nan")
        print(f"{t:12s} mean SNR {s.mean():.2f}  median {np.median(s):.2f}  "
              f"frac>=2 {(s >= 2).mean():.2f}  frac>=1 {(s >= 1).mean():.2f}  "
              f"corr(SNR,fill) {corr:.2f}  sign(plus-minus) +:{(signs > 0).sum()} -:{(signs < 0).sum()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="whole-piece extreme sweeps, per-stroke SNR")
    ap.add_argument("--piece")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--mock", dest="real", action="store_false")
    ap.add_argument("--repeats", type=int, default=4, help="episodes per condition")
    ap.add_argument("--conditions", default="zero,plus,minus",
                    help="comma list from zero,plus,minus,deepslow,shallowfast,deep,shallow,fast,slow")
    ap.add_argument("--tempo-scale", type=float, default=1.0)
    ap.add_argument("--calibrated-dynamics", action="store_true", default=True)
    ap.add_argument("--no-calibrated-dynamics", dest="calibrated_dynamics",
                    action="store_false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None, help="re-analyse an existing json and exit")
    args = ap.parse_args()

    if args.analyze:
        data = json.load(open(args.analyze))
        print_report(data, analyze(data))
        return
    if not args.piece:
        ap.error("--piece is required unless --analyze")

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    env = build_env(args)
    dim = env.action_space.shape[0]
    n_plan = len(env.plan)
    data = {"piece": args.piece, "real": args.real, "n_strokes": n_plan,
            "conditions": conds, "repeats": args.repeats,
            "tempo_scale": args.tempo_scale, "calibrated_dynamics": args.calibrated_dynamics,
            "t": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes": []}
    print(f"piece {args.piece}  strokes {n_plan}  dim {dim}  conditions {conds}  "
          f"repeats {args.repeats}  -> {len(conds) * args.repeats} episodes", flush=True)
    try:
        for k in range(args.repeats):
            for c in conds:                       # interleaved: drift shared
                print(f"  episode {len(data['episodes']) + 1}/{len(conds) * args.repeats}"
                      f"  {c} (repeat {k + 1})", flush=True)
                rows, ok = play_episode(env, condition_vector(c, dim))
                data["episodes"].append({"condition": c, "repeat": k, "completed": ok,
                                         "rows": rows})
                if rows:
                    q = [r.get("quality", float("nan")) for r in rows]
                    print(f"      strokes {len(rows)}  mean q {np.nanmean(q):.3f}", flush=True)
                if args.out:                      # checkpoint after every episode
                    Path(args.out).write_text(json.dumps(data))
    finally:
        env.close()
    print_report(data, analyze(data))


if __name__ == "__main__":
    main()
