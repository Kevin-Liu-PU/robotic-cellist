#!/usr/bin/env zsh
# 2026-09-12 lab day: BIN baseline (train s0-s2 + eval) then the ten-day Sustain brackets (5 seed-0 policies x 2).
# Gain offset measured today +5.02 dB (robust sd 1.75, n 25); file kept at +5.76 so BIN shares the loudness frame
# of FULL (9/3), C1 (9/9), D and G1 (9/10).
# Historical hardware recipe: disabled until its private inputs and fixed protocol are reviewed.
# Restore the exact files in REQUIRED_INPUTS below; do not substitute another run.
# Set CELLO_CONTROL_DIR to an existing absolute directory for STOP_CHAIN/PAUSE_CHAIN.
# Set CELLO_PYTHON to an existing absolute Python executable, or use the repository .venv.
# With on-site coverage and the hardware setup verified, explicitly opt in with
# CELLO_ENABLE_HISTORICAL_RECIPES=1. This opt-in does not replace any downstream approval.
if [[ "${CELLO_ENABLE_HISTORICAL_RECIPES:-}" != 1 ]]; then
  print -u2 -- "Historical recipe disabled. Read the setup and REQUIRED_INPUTS in $0 before setting CELLO_ENABLE_HISTORICAL_RECIPES=1."
  exit 2
fi
ROOT=${0:A:h:h}
[[ -d "$ROOT/rl" ]] || { print -u2 -- "Repository rl directory missing: $ROOT/rl"; exit 2; }
cd -- "$ROOT" || exit 2
SP=${CELLO_CONTROL_DIR:-}
[[ "$SP" == /* && -d "$SP" && -w "$SP" ]] || { print -u2 -- "CELLO_CONTROL_DIR must name an existing writable absolute directory."; exit 2; }
PY=${CELLO_PYTHON:-$ROOT/.venv/bin/python}
[[ "$PY" == /* && -f "$PY" && -x "$PY" ]] || { print -u2 -- "Set CELLO_PYTHON to an existing absolute Python executable; no environment is installed automatically."; exit 2; }
CP=rl/checkpoints_piece
[[ -d "$CP" && -w "$CP" ]] || { print -u2 -- "Restore an existing writable $ROOT/$CP directory before enabling this recipe."; exit 2; }
# These inputs are private or repository-local; this script does not download or replace them.
REQUIRED_INPUTS=(
  "bach_sarabande_m0-7.mxl"
  "maysong_m0-11.mxl"
  "rl/bracket_all5_20260908.sh"
  "rl/chain_BIN_vocalise_20260912.sh"
  "rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final.zip"
  "rl/checkpoints_piece/run_20260902_120353/sac_piece_maysong_m0-11_final.zip"
  "rl/checkpoints_piece/run_20260902_220009/sac_piece_bach_sarabande_m0-7_final.zip"
  "rl/checkpoints_piece/run_20260903_074943/sac_piece_yunpiece_m9-24_final.zip"
  "rl/checkpoints_piece/run_20260903_151543/sac_piece_vocalise_final.zip"
  "rl/checkpoints_piece/run_20260903_155747/sac_piece_vocalise_final.zip"
  "rl/checkpoints_piece/run_20260903_164107/sac_piece_vocalise_final.zip"
  "twinkle-short.mxl"
  "vocalise.mxl"
  "yunpiece_m9-24.mxl"
)
for required_input in "${REQUIRED_INPUTS[@]}"; do
  [[ -f "$required_input" && -r "$required_input" ]] || { print -u2 -- "Missing required historical input: $required_input"; exit 2; }
done
rl/chain_BIN_vocalise_20260912.sh all
echo "=== BIN chain exit $? $(date +%H:%M:%S)"
rl/bracket_all5_20260908.sh day11 2
echo "=== SUSTAIN day11 exit $? $(date +%H:%M:%S)"
echo "=== LAB DAY 20260912 DONE $(date +%H:%M:%S)"
