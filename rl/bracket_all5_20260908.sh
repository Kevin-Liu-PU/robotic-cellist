#!/usr/bin/env zsh
# TEN-DAY SUSTAIN brackets, all five seed-0 policies (historical 9/6 plan: Mon 9/8 = day 5-7, Fri 9/12 = day 9-11).
# Per piece: Base -> seed-0 FINAL -> Base, live through the training executor, same protocol as the
# 16 twinkle brackets of 9/1-9/3. Then posthoc dual judge per bracket. ~15 min of robot time.
# Pre-registration + ledger: the private daylog_20260901_lab.md (9/6). Paper V-D is written as done.
#
# Usage:  rl/bracket_all5_20260908.sh DAYTAG [REPEATS]      e.g.  rl/bracket_all5_20260908.sh day6   (or day10 2)
# Before running: robot connected (192.168.1.100), Scarlett on, caffeinate on, then the session gain offset:
#     "${CELLO_PYTHON:-.venv/bin/python}" rl/calibrate_gain.py --measure
# Control files in $SP: STOP_CHAIN (halt after the current take), PAUSE_CHAIN (wait).
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
DAY=${1:?DAYTAG required, e.g. day6}; REPEATS=${2:-1}
STOP=$SP/STOP_CHAIN; PAUSE=$SP/PAUSE_CHAIN
CP=rl/checkpoints_piece
[[ -d "$CP" && -w "$CP" ]] || { print -u2 -- "Restore an existing writable $ROOT/$CP directory before enabling this recipe."; exit 2; }
# These inputs are private or repository-local; this script does not download or replace them.
REQUIRED_INPUTS=(
  "bach_sarabande_m0-7.mxl"
  "maysong_m0-11.mxl"
  "rl/calibrate_gain.py"
  "rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final.zip"
  "rl/checkpoints_piece/run_20260902_120353/sac_piece_maysong_m0-11_final.zip"
  "rl/checkpoints_piece/run_20260902_220009/sac_piece_bach_sarabande_m0-7_final.zip"
  "rl/checkpoints_piece/run_20260903_074943/sac_piece_yunpiece_m9-24_final.zip"
  "rl/checkpoints_piece/run_20260903_151543/sac_piece_vocalise_final.zip"
  "rl/perform.py"
  "rl/posthoc_heldout.py"
  "twinkle-short.mxl"
  "vocalise.mxl"
  "yunpiece_m9-24.mxl"
)
for required_input in "${REQUIRED_INPUTS[@]}"; do
  [[ -f "$required_input" && -r "$required_input" ]] || { print -u2 -- "Missing required historical input: $required_input"; exit 2; }
done
SEM=(CELLO_ENVELOPE_MIN_S=0.5 CELLO_PHRASE_SCORING=1 CELLO_TIMING_PENALTY=1)
F="perform ->\|contact\|dBFS\|tempo_ratio\|mean_quality_true\|mean_score_ms\|Traceback\|Error\|LOST\|recover\|ckpt\|loaded"

# piece file | seed-0 FINAL checkpoint (the five policies on the listening site)
typeset -A PIECE POL
PIECE=(twinkle twinkle-short.mxl  maysong maysong_m0-11.mxl  bach bach_sarabande_m0-7.mxl  popper yunpiece_m9-24.mxl  vocalise vocalise.mxl)
POL=(twinkle  $CP/run_20260901_185917/sac_piece_twinkle-short_final
     maysong  $CP/run_20260902_120353/sac_piece_maysong_m0-11_final
     bach     $CP/run_20260902_220009/sac_piece_bach_sarabande_m0-7_final
     popper   $CP/run_20260903_074943/sac_piece_yunpiece_m9-24_final
     vocalise $CP/run_20260903_151543/sac_piece_vocalise_final)
ORDER=(twinkle maysong bach popper vocalise)

stage() { echo "=== $1 $(date +%H:%M:%S)"; }
check_stop() {
  while [ -f "$PAUSE" ]; do stage "PAUSE file present, waiting"; sleep 60; done
  if [ -f "$STOP" ]; then stage "STOP file found, halted"; exit 0; fi
}
take() {
  local LOG=$1; shift; local out attempt
  for attempt in 1 2; do
    out=$(env $SEM "$PY" -u rl/perform.py --real --tempo-scale 1.0 "$@" 2>&1 | grep -v -i warning | grep -i "$F")
    echo "$out" | tee -a $LOG
    if echo "$out" | grep -q "perform ->"; then return 0; fi
    echo "take FAILED (attempt $attempt): $*  retry in 20 s" | tee -a $LOG; sleep 20
  done
  return 1
}
takedirs() { sed -n 's/^.*perform ->[[:space:]]*//p' "$1"; }
posthoc() { "$PY" rl/posthoc_heldout.py "$@" 2>&1 | grep -v -i warning | grep "mean_tone=\|delta\|mean, n\|condition"; }

for p in $ORDER; do
  if [ ! -f "${POL[$p]}.zip" ]; then stage "MISSING checkpoint for $p: ${POL[$p]}.zip"; exit 1; fi
done
SUMMARY=$CP/bracket_all5_${DAY}_summary.txt; : > $SUMMARY
stage "TEN-DAY SUSTAIN $DAY: 5 pieces x $REPEATS bracket(s), Base -> FINAL -> Base"
for rep in $(seq 1 $REPEATS); do
  for p in $ORDER; do
    check_stop
    LOG=$CP/bracket_${p}_${DAY}_r${rep}.log; : > $LOG
    stage "BRACKET $p $DAY r$rep"
    echo "=== B1 $(date +%T)" | tee -a $LOG; take $LOG --baseline ${PIECE[$p]}
    echo "=== POL $(date +%T)" | tee -a $LOG; take $LOG ${POL[$p]} ${PIECE[$p]}
    echo "=== B2 $(date +%T)" | tee -a $LOG; take $LOG --baseline ${PIECE[$p]}
    d=("${(@f)$(takedirs "$LOG")}"); echo "takes: ${#d[@]}"
    if [ ${#d[@]} -eq 3 ]; then
      { echo "## $p $DAY r$rep  B1=${d[1]:t} POL=${d[2]:t} B2=${d[3]:t}";
        posthoc --label "bracket $p $DAY r$rep (RL = s0 final; pair1 vs pre-Base, pair2 vs post-Base)" Base1=${d[1]} RL1=${d[2]} Base2=${d[3]} RL2=${d[2]}; } | tee -a $SUMMARY
    else echo "bracket $p $DAY r$rep incomplete (${#d[@]}/3)" | tee -a $SUMMARY; fi
  done
done
stage "DONE $DAY; summary: $SUMMARY"
