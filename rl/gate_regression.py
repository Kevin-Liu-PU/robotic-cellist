"""
rl/gate_regression.py

Per-dimension SNR table from a reward_noise.py gate file, computed the way
the paper defines it (Method III-D): for every reward term k and action
dimension j,

    sigma_k    sd of term k when the SAME zero action is replayed
               (the "repeat" rows -- pure measurement noise)
    rho_kj     change in the MEAN of term k when dimension j is swept
               across the policy's full authority [-1, +1], read off by
               regressing the term's score on the sampled residuals:
               rho_kj = 2 * |b_kj| where b_kj is the OLS slope per unit
               action (probes are uniform in [-1,1]^dim, so 2*b is the
               end-to-end swing)
    SNR_kj     rho_kj / sigma_k   -- d-prime of signal detection theory,
               measured per term, per dimension, through the plant

The aggregate gate that reward_noise.py already prints (probe sd / repeat
sd) answers "does choosing an action move this term at all"; this script
answers "WHICH dimension moves it, and by how much" -- the go/no-go is
taken on the terms the action space is intended to move, not on the
total.

Requires a gate file with RAW rows (repeat_rows / probe_rows), written by
reward_noise.py from 2026-08-31 on. Earlier gate.json files kept only
(mean, sd) summaries; on those this script can only tell you so.

Usage:
    .venv/bin/python rl/gate_regression.py rl/gate.json
    ... --term quality            only this term
    ... --pooled                  also fit one pooled model across strokes
                                  (per-stroke intercepts; more df, assumes
                                  slopes shared across the piece)
    ... --json out.json           machine-readable table


"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TERMS = ("total", "quality", "r_dynamic", "r_defect", "r_onset",
         "r_speak", "r_envelope", "r_bow")


def fit(rows: list[dict], term: str, dim: int):
    """OLS of term score on the action vector. Returns (slopes b_j,
    intercept, n, residual sd) or None if the term/rows are unusable.

    Repeat rows (zero action) are included: they pin the intercept and add
    residual degrees of freedom without touching the slopes' identification.
    """
    X, y = [], []
    for r in rows:
        if term in r and r[term] is not None and len(r["action"]) == dim:
            X.append(r["action"])
            y.append(float(r[term]))
    n = len(y)
    if n < dim + 2:            # need at least 1 df for the residual
        return None
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    A = np.hstack([X, np.ones((n, 1))])
    coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    if rank < dim + 1:
        return None
    resid = y - A @ coef
    df = n - (dim + 1)
    resid_sd = float(np.sqrt(resid @ resid / df)) if df > 0 else float("nan")
    return coef[:dim], float(coef[dim]), n, resid_sd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="per-dimension SNR (d-prime) table from a raw-row gate file")
    ap.add_argument("gate", help="json written by rl/reward_noise.py --out")
    ap.add_argument("--term", default=None, help="restrict to one reward term")
    ap.add_argument("--pooled", action="store_true",
                    help="also fit one model across all strokes "
                         "(per-stroke intercepts)")
    ap.add_argument("--json", default=None, help="write the table here too")
    args = ap.parse_args()

    report = json.loads(Path(args.gate).read_text())
    strokes = report.get("strokes", {})
    if not strokes:
        sys.exit(f"{args.gate}: no strokes section")
    if not any("probe_rows" in e for e in strokes.values()):
        sys.exit(f"{args.gate} has no raw rows (repeat_rows/probe_rows) -- "
                 "it predates the 2026-08-31 reward_noise.py. Summaries "
                 "cannot support the per-dimension regression; re-run the "
                 "gate.")

    terms = [args.term] if args.term else list(TERMS)
    dim = len(next(iter(
        e["probe_rows"][0]["action"] for e in strokes.values()
        if e.get("probe_rows"))))
    dims = [f"a{j}" for j in range(dim)]
    out = {"gate": args.gate, "dim": dim, "strokes": {}, "pooled": {}}

    for ti, entry in strokes.items():
        rows = entry.get("repeat_rows", []) + entry.get("probe_rows", [])
        print(f"\n{'=' * 66}\n  stroke {ti}   "
              f"({entry.get('n_repeat', '?')} repeats + "
              f"{entry.get('n_probe', '?')} probes)\n{'=' * 66}")
        print(f"  {'term':<12}{'sigma_k':>9} | "
              + " ".join(f"{d:>7}" for d in dims) + "   (SNR_kj = 2|b|/sigma)")
        for t in terms:
            sig = entry.get("repeat", {}).get(t, [None, None])[1]
            f_ = fit(rows, t, dim)
            if f_ is None or sig is None:
                continue
            b, _, n, _ = f_
            if sig <= 1e-9:
                snr = ["inf"] * dim
            else:
                snr = [f"{2 * abs(bj) / sig:.2f}" for bj in b]
            print(f"  {t:<12}{sig:>9.4f} | "
                  + " ".join(f"{s:>7}" for s in snr))
            out["strokes"].setdefault(ti, {})[t] = {
                "sigma": sig, "slopes": [float(x) for x in b],
                "snr": [None if sig <= 1e-9 else 2 * abs(float(bj)) / sig
                        for bj in b], "n": n}
        nrows = len(rows)
        if nrows < 2 * dim + 2:
            print(f"  (only {nrows} rows for {dim + 1} parameters -- "
                  f"estimates are indicative, not evidence; gate with "
                  f"--probes {2 * dim}+ next time)")

    if args.pooled:
        # One slope vector shared across strokes, intercept per stroke:
        # demean action and score within each stroke, then regress the
        # stacked residuals (equivalent to stroke fixed effects).
        print(f"\n{'=' * 66}\n  POOLED (shared slopes, per-stroke "
              f"intercepts)\n{'=' * 66}")
        for t in terms:
            X, y, sigs = [], [], []
            for ti, entry in strokes.items():
                rows = (entry.get("repeat_rows", [])
                        + entry.get("probe_rows", []))
                xs = [r["action"] for r in rows
                      if t in r and r[t] is not None]
                ys = [float(r[t]) for r in rows
                      if t in r and r[t] is not None]
                if len(ys) < 2:
                    continue
                xs, ys = np.asarray(xs, float), np.asarray(ys, float)
                X.append(xs - xs.mean(axis=0))
                y.append(ys - ys.mean())
                s = entry.get("repeat", {}).get(t, [None, None])[1]
                if s is not None:
                    sigs.append(s)
            if not X or not sigs:
                continue
            Xc, yc = np.vstack(X), np.concatenate(y)
            if len(yc) < dim + len(strokes) + 1:
                continue
            b, _, rank, _ = np.linalg.lstsq(Xc, yc, rcond=None)
            if rank < dim:
                continue
            sig = float(np.mean(sigs))     # mean repeat sd across strokes
            snr = [f"{2 * abs(bj) / sig:.2f}" if sig > 1e-9 else "inf"
                   for bj in b]
            print(f"  {t:<12}{sig:>9.4f} | "
                  + " ".join(f"{s:>7}" for s in snr))
            out["pooled"][t] = {
                "sigma_mean": sig, "slopes": [float(x) for x in b],
                "snr": [None if sig <= 1e-9 else 2 * abs(float(bj)) / sig
                        for bj in b], "n": int(len(yc))}

    print(f"\n  dims: a0..a{dim - 1} = the residual action vector in "
          f"training order (3 speed segments, then 3 depth segments for "
          f"the 6-dim piece loop)")
    print(f"  read: SNR_kj >= ~2 means sweeping dimension j across full "
          f"authority moves term k by twice its replay noise -- trainable; "
          f"~1 marginal; <1 invisible.")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
