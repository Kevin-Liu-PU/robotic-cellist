"""
rl/pack_listening.py

Package baseline/policy take pairs into a blinded listening set for the
human evaluation (paper Sec. V-B.1, protocol decided 2026-09-01: every
same-session pair of every piece is evaluated, no take is selected by its
score; listeners rate six categories as A / B / tie).

Usage (same Base/RL pair syntax as posthoc_heldout.py):
    .venv/bin/python rl/pack_listening.py --piece twinkle --out listening/twinkle_0902 \
        Base1=rl/checkpoints_piece/perform_X RL1=rl/checkpoints_piece/perform_Y \
        Base2=... RL2=... Base3=... RL3=...

Writes, in --out:
    <piece>_pair1_A.wav, <piece>_pair1_B.wav, ...   blinded copies
    README_listener.md                              instructions + rating sheet
    responses_<piece>.csv                           empty sheet, one row per pair
    KEY_<piece>.json                                which of A/B is the policy
                                                    (PRIVATE: do not send; seed
                                                    recorded so the draw is
                                                    reproducible)
The A/B assignment is drawn per pair from a seeded RNG; the seed defaults
to a hash of the piece name and can be overridden with --seed.


"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CATEGORIES = ["rhythm", "tone", "dynamics", "expressivity", "humanlike", "overall"]


def take_wav(d: Path) -> Path:
    """The take's full recording (not the per-stroke files)."""
    wavs = [w for w in d.glob("*.wav") if "stroke" not in w.name.lower()
            and not w.name.startswith("ep")]
    if not wavs:
        wavs = list(d.glob("*.wav"))
    if not wavs:
        sys.exit(f"no wav in {d}")
    return max(wavs, key=lambda w: w.stat().st_size)


def parse_pairs(items: list[str]) -> list[tuple[Path, Path]]:
    base, rl = {}, {}
    for it in items:
        if "=" not in it:
            sys.exit(f"expected NameN=dir, got {it}")
        k, v = it.split("=", 1)
        if k.lower().startswith("base"):
            base[k[4:]] = Path(v)
        elif k.lower().startswith("rl"):
            rl[k[2:]] = Path(v)
        else:
            sys.exit(f"pair keys must be BaseN / RLN, got {k}")
    if set(base) != set(rl):
        sys.exit(f"unmatched pairs: Base {sorted(base)} vs RL {sorted(rl)}")
    return [(base[i], rl[i]) for i in sorted(base, key=lambda s: int(s))]


README = """# Listening evaluation: {piece}

You will hear {n} pairs of recordings of the same passage. In each pair,
one recording is the robot's baseline performance and the other is the
robot after training. Which is which is randomized and hidden.

For each pair, mark which recording is better in each category, or "tie":

    rhythm, tone, dynamics, expressivity, humanlike, overall

Please judge TONE and the other categories on their own terms. Note
timing is identical at the command level on both sides of every pair
(both play the same written durations), so any difference you hear in
rhythm comes from how notes are shaped and attacked, not from a
different reading of the music.

Listen to each pair as many times as you like, in any order. Fill in
`responses_{piece}.csv` (one row per pair, A / B / tie per category) and
send it back. Thank you.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="blinded listening pack")
    ap.add_argument("pairs", nargs="+", help="Base1=dir RL1=dir Base2=dir ...")
    ap.add_argument("--piece", required=True, help="short piece name for filenames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for the A/B draw (default: hash of --piece)")
    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)
    seed = (args.seed if args.seed is not None
            else int(hashlib.sha256(args.piece.encode()).hexdigest()[:8], 16))
    rng = np.random.default_rng(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    key = {"piece": args.piece, "seed": seed,
           "packed": datetime.now().isoformat(timespec="seconds"), "pairs": []}
    for k, (b, r) in enumerate(pairs, start=1):
        policy_is_A = bool(rng.integers(0, 2))
        a_src, b_src = (r, b) if policy_is_A else (b, r)
        a_dst = out / f"{args.piece}_pair{k}_A.wav"
        b_dst = out / f"{args.piece}_pair{k}_B.wav"
        shutil.copyfile(take_wav(a_src), a_dst)
        shutil.copyfile(take_wav(b_src), b_dst)
        key["pairs"].append({"pair": k, "policy": "A" if policy_is_A else "B",
                             "baseline_dir": str(b), "policy_dir": str(r)})
        print(f"pair {k}: A <- {a_src.name}   B <- {b_src.name}")

    (out / "README_listener.md").write_text(README.format(piece=args.piece,
                                                            n=len(pairs)))
    with open(out / f"responses_{args.piece}.csv", "w") as f:
        f.write("pair," + ",".join(CATEGORIES) + ",comments\n")
        for k in range(1, len(pairs) + 1):
            f.write(f"{k}," + "," * len(CATEGORIES) + "\n")
    (out / f"KEY_{args.piece}.json").write_text(json.dumps(key, indent=2))
    print(f"\nwrote {out}/  ({len(pairs)} pairs, seed {seed})")
    print(f"SEND: the wavs, README_listener.md, responses_{args.piece}.csv")
    print(f"KEEP PRIVATE: KEY_{args.piece}.json")


if __name__ == "__main__":
    main()
