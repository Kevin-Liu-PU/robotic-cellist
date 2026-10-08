"""
rl/posthoc_heldout.py

One command -> the dual-judge training-gain table for one condition:
each labeled Base/RL pair is scored under BOTH judge checkpoints with the
exact training-time feeding, and the paired differences are printed side
by side with the definition of the number.

  column 1  training judge   quality_cnn_multilength.pt
            (the checkpoint the policy was actually trained against)
  column 2  held-out judge   quality_cnn_human_A1_A5.pt
            (never used in training or checkpoint selection; sibling model
            trained on the same human labels — the reward-hacking check)

THE NUMBER (both columns, identical formula, only the network differs):

  tone(note)      = sum_h w_h(fill) * head_h / sum_h w_h(fill)
                    heads = tone/attack/release/overall quality,
                    w = TONE_HEAD_WEIGHTS (0.45/0.25/0.15/0.15 at full
                    fill, shifted toward attack/release for short notes)
                    — the same mix piece_env._reward uses in training
  mean_tone(take) = mean of tone(note) over the take's notes
  pair delta      = mean_tone(RL take) - mean_tone(Base take)
  TABLE VALUE     = mean of the pair deltas (n = number of pairs)

Feeding is training-identical (rl/posthoc_score.py): audio sliced at the
EXECUTED onsets, un-normalized; speed/torque rebuilt from the 100 Hz state
log. Scores are written as posthoc_<judge>.json in each dir; the existing
posthoc.json is left untouched.

Usage:
    python rl/posthoc_heldout.py Base1=DIR RL1=DIR [Base2=DIR RL2=DIR ...] \
        [--label "vocalise - ALT (8/25)"]

the operator, 2026-08-27; labeled-pair interface by Jeongmin Oh (ver2);
dual-judge table 2026-08-28.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CKPT_DIR = REPO_ROOT / "SoundClassifier" / "checkpoints"
JUDGES = [
    ("training multilength", CKPT_DIR / "quality_cnn_multilength.pt"),
    ("held-out human_A1_A5", CKPT_DIR / "quality_cnn_human_A1_A5.pt"),
]

LEGEND = """
WHAT THE NUMBERS ARE
  Both columns: paired delta of mean_tone (RL take minus its back-to-back
  Base take, averaged over pairs). Same formula, same feeding; only the
  judge network differs.
    tone(note)      = fill-weighted mix of the classifier heads
                      (TONE_HEAD_WEIGHTS 0.45 tone / 0.25 attack /
                      0.15 release / 0.15 overall at full fill, shifted
                      toward attack/release for short notes) -- identical
                      to the judge term inside the training reward
    mean_tone(take) = mean of tone(note) over the take
    table value     = mean over pairs of [mean_tone(RL) - mean_tone(Base)]
  Feeding (training-identical, rl/posthoc_score.py): audio sliced at the
  executed onsets, un-normalized; measured speed/torque from the state log.
  training multilength = the checkpoint the policy was trained against.
  held-out human_A1_A5 = never in training or checkpoint selection; sibling
  model on the same human labels -- the reward-hacking check.
"""


def score_with(ckpt: Path, dirs: list[tuple[str, Path]]) -> dict[str, dict]:
    """Score every dir under one judge; returns label -> result."""
    import rl.piece_env as piece_env
    from rl import posthoc_score
    piece_env.HUMAN_CHECKPOINT = ckpt
    posthoc_score._JUDGE = None          # force rebuild with this checkpoint
    tag = ckpt.stem.replace("quality_cnn_", "")

    out = {}
    for label, d in dirs:
        keep = d / "posthoc.json"
        original = keep.read_text(encoding="utf-8") if keep.exists() else None
        res = posthoc_score.score_dir(d)
        if res is None:
            sys.exit(f"nothing scoreable in {d}")
        (d / "posthoc.json").replace(d / f"posthoc_{tag}.json")
        if original is not None:
            keep.write_text(original, encoding="utf-8")
        out[label] = res
        print(f"  {label:6s} [{res['stem']}]  mean_tone={res['mean_tone']:.4f}"
              f"  (n_notes={res['n']}, skipped={res['skipped']})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="dual-judge training-gain table")
    ap.add_argument("runs", nargs="+", metavar="LABEL=PERFORM_DIR",
                    help="Base1=dir RL1=dir Base2=dir RL2=dir ...")
    ap.add_argument("--label", default=None,
                    help="condition name printed above the table")
    args = ap.parse_args()

    dirs = []
    for raw in args.runs:
        if "=" not in raw:
            sys.exit(f"expected LABEL=DIR, got: {raw}")
        label, raw_dir = raw.split("=", 1)
        d = Path(raw_dir)
        if not d.is_absolute() and not d.exists():
            d = REPO_ROOT / d
        dirs.append((label, d))

    labels = [l for l, _ in dirs]
    pair_ids = [l.replace("Base", "", 1) for l in labels if l.startswith("Base")]
    pairs = [(f"Base{i}", f"RL{i}") for i in pair_ids if f"RL{i}" in labels]
    if not pairs:
        sys.exit("no Base<i>/RL<i> pairs found in the labels")

    per_judge = {}
    for judge_name, ckpt in JUDGES:
        if not ckpt.exists():
            sys.exit(f"checkpoint not found: {ckpt}")
        print(f"\nScoring under {judge_name} ({ckpt.name}):")
        per_judge[judge_name] = score_with(ckpt, dirs)

    condition = args.label or per_judge[JUDGES[0][0]][labels[0]]["stem"].split("_")[0]
    col = [name for name, _ in JUDGES]
    w = max(len(c) for c in col) + 2

    print(f"\ncondition: {condition}")
    header = f"| {'':14s} | " + " | ".join(f"{c:^{w}s}" for c in col) + " |"
    rule = "|" + "-" * 16 + ("|" + "-" * (w + 2)) * len(col) + "|"
    print(rule.replace("-", "="))
    print(header)
    print(rule)
    deltas = {c: [] for c in col}
    for b, r in pairs:
        row = []
        for c in col:
            d = per_judge[c][r]["mean_tone"] - per_judge[c][b]["mean_tone"]
            deltas[c].append(d)
            row.append(f"{d:+.4f}")
        print(f"| pair {b[4:]:>2s} delta  | "
              + " | ".join(f"{v:^{w}s}" for v in row) + " |")
    print(rule)
    summary = []
    for c in col:
        v = np.array(deltas[c])
        sd = f" (SD {np.std(v, ddof=1):.4f})" if len(v) > 1 else ""
        summary.append(f"{v.mean():+.4f}{sd}")
    print(f"| mean, n={len(pairs):<2d}     | "
          + " | ".join(f"{s:^{w}s}" for s in summary) + " |")
    print(rule.replace("-", "="))
    print(LEGEND)


if __name__ == "__main__":
    main()
