"""
rl/run_semantics.py

Re-apply a training run's action/judge semantics at evaluation time.

The ALT loop changes what the ENV means: which notes get envelopes
(CELLO_ENVELOPE_MIN_S 0.40 -> 0.50), what audio the judge hears
(CELLO_PHRASE_SCORING), and whether shaping is priced (CELLO_TIMING_PENALTY).
An ALT-trained checkpoint evaluated under stock semantics -- or a stock one
evaluated in a shell still exporting CELLO_ALT_LOOP=1 from the training that
just finished -- is scored in a different environment than it was trained
in, and notes in [0.40, 0.50) s silently change behaviour between the two.

train_piece_logged records the semantics in the stroke log's header;
select_best and ab_compare call apply_run_semantics(run_dir) to replay them.
Only setdefault: a CELLO_* variable already set in the shell still wins, so
an operator can deliberately override. Runs whose header predates the fields
(before 2026-08-21) are a silent no-op.


"""

from __future__ import annotations

import json
import os
from pathlib import Path

_MAP = (
    ("envelope_min_s", "CELLO_ENVELOPE_MIN_S", lambda v: f"{float(v):g}"),
    ("alt_timing_penalty", "CELLO_TIMING_PENALTY", lambda v: "1" if v else "0"),
    ("alt_phrase_scoring", "CELLO_PHRASE_SCORING", lambda v: "1" if v else "0"),
)


def apply_run_semantics(run_dir) -> dict:
    """setdefault the run's recorded semantics into os.environ.

    Returns {var: value} for what was applied. Must be called BEFORE
    rl.piece_env is imported, since the flags are read at import time.
    """
    log = Path(run_dir) / "stroke_log.jsonl"
    try:
        with open(log) as f:
            header = json.loads(f.readline())
    except (OSError, ValueError):
        return {}
    if not header.get("header"):
        return {}
    applied = {}
    for field, var, fmt in _MAP:
        if field in header and os.environ.get(var) is None:
            os.environ[var] = fmt(header[field])
            applied[var] = os.environ[var]
    if applied:
        print("run semantics re-applied from the run's stroke_log header: "
              + "  ".join(f"{k}={v}" for k, v in sorted(applied.items())))
    return applied


def apply_reward_overrides(run_dir) -> dict:
    """Re-apply the run's --reward-weights overrides to rl.piece_env.

    Call AFTER rl.piece_env is importable (i.e. after apply_run_semantics).
    train_piece_logged records the overrides in the header as
    "reward_overrides"; without this, select_best ranks a run trained under
    e.g. W_QUALITY=2.0 by returns computed with the default 0.5, i.e. by a
    different objective than the one the checkpoints optimised.
    Mirrors train_piece_logged._apply_reward_weights (scalar constants,
    TONE_HEAD_WEIGHTS.<head>, DEFECT_WEIGHTS.<feature>).
    """
    log = Path(run_dir) / "stroke_log.jsonl"
    try:
        with open(log) as f:
            header = json.loads(f.readline())
    except (OSError, ValueError):
        return {}
    changes = header.get("reward_overrides") or {}
    if not changes:
        return {}
    import rl.piece_env as pe
    for key, val in changes.items():
        if key.startswith("TONE_HEAD_WEIGHTS."):
            pe.TONE_HEAD_WEIGHTS[key.split(".", 1)[1]] = float(val)
        elif key.startswith("DEFECT_WEIGHTS."):
            feat = key.split(".", 1)[1]
            for i, row in enumerate(pe.DEFECT_FEATURES):
                if row[0] == feat:
                    pe.DEFECT_FEATURES[i] = (*row[:4], float(val))
                    break
        elif hasattr(pe, key) and isinstance(getattr(pe, key), (int, float)):
            setattr(pe, key, float(val))
        else:
            raise SystemExit(f"reward_overrides in {log}: piece_env has no "
                             f"scalar '{key}'")
    print("reward overrides re-applied from the run's stroke_log header: "
          + "  ".join(f"{k}={v}" for k, v in sorted(changes.items())))
    return changes
