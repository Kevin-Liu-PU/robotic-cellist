"""Play one bundled MIDI score with the mock executor and scorer."""

import argparse
from pathlib import Path


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    from rl.play_piece import main as play_piece

    score = Path(__file__).resolve().parent / "MIDI-Files" / "t1.mid"
    # No hardware, classifier, or policy flags: play the mock baseline.
    play_piece([str(score), "--episodes", "1", "--no-calibrated-dynamics"])


if __name__ == "__main__":
    main()
