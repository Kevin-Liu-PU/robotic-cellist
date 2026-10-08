#!/usr/bin/env zsh
# AUDIT NEGATIVE ARM: yunpiece_m9-24 (Popper opening, 32 strokes, swept 09:36 -> slope ~0, PREDICTED Dtone~0) seeds 0/1/2.
# Confirms the predicted-FAILURE reproduces across seeds (matches the 3-seed positive arm). Budget 3200 = 100 eps x 32. NOT auto-launched; start only after explicit operator approval.
# bach_sarabande_m0-7 seeds 1,2 (historical request to continue at 23:0x); seed 0 = run_20260902_220009 done. Same recipe.
# bach_sarabande_m0-7 seed 0, 9/2 21:1x (historical request to start training). Same recipe as maysong: ALT2, 100 eps x 42 strokes = 4200, ref-every 5, ckpt-every 10 -> select_best --last 7 -> live A/B x3 -> posthoc -> C1 -> twinkle bracket.
# maysong_m0-11 seed 1 RESUME (from run_20260902_153317, 84 eps banked) + seed 2 full, 9/2 16:4x. Original 15:3x chain halted at 16:34 (robot drops).
# Historical request, 9/2 15:3x: finish maysong_m0-11 seeds 1 and 2 to test whether the observed effect reproduces.
# Same recipe as seed 0 (run_20260902_120353): ALT2 env, 4400 strokes = 100 eps x 44, ref-every 5, ckpt-every 10.
# Per seed: train -> select_best --last 7 -> live A/B x3 (Base / FINAL, live, no --compile) -> posthoc dual judge -> C1 -> twinkle bracket.
# Helpers copied from rl/chain_positive_20260902.sh. Control files in $SP: STOP_CHAIN, PAUSE_CHAIN, SKIP_SWEEP.
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
STOP=$SP/STOP_CHAIN; PAUSE=$SP/PAUSE_CHAIN
CP=rl/checkpoints_piece
[[ -d "$CP" && -w "$CP" ]] || { print -u2 -- "Restore an existing writable $ROOT/$CP directory before enabling this recipe."; exit 2; }
# These inputs are private or repository-local; this script does not download or replace them.
REQUIRED_INPUTS=(
  "maysong_m0-11.mxl"
  "rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final.zip"
  "rl/ckpt_entropy.py"
  "rl/perform.py"
  "rl/posthoc_heldout.py"
  "rl/select_best.py"
  "rl/train_supervisor.py"
  "twinkle-short.mxl"
  "yunpiece_m9-24.mxl"
)
for required_input in "${REQUIRED_INPUTS[@]}"; do
  [[ -f "$required_input" && -r "$required_input" ]] || { print -u2 -- "Missing required historical input: $required_input"; exit 2; }
done
POL=rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final   # twinkle seed-0 FINAL, fixed all campaign
SEM=(CELLO_ENVELOPE_MIN_S=0.5 CELLO_PHRASE_SCORING=1 CELLO_TIMING_PENALTY=1)
ALT2=(CELLO_ALT_LOOP=1 CELLO_ENVELOPE_MIN_S=0.5 CELLO_TIMING_PENALTY=1 CELLO_PHRASE_SCORING=1 CELLO_ZERO_INIT=1 CELLO_TARGET_ENTROPY_SCALE=0.5 CELLO_ENT_COEF_INIT=0.005)
F="perform ->\|contact\|dBFS\|tempo_ratio\|mean_quality_true\|mean_score_ms\|Traceback\|Error\|LOST\|recover\|ckpt\|loaded"

stage() { echo "=== $1 $(date +%H:%M:%S)"; }
check_stop() {
  while [ -f "$PAUSE" ]; do stage "PAUSE file present, waiting"; sleep 60; done
  if [ -f "$STOP" ]; then stage "STOP file found, chain halted"; exit 0; fi
}
newest_run() { local d; for d in $(ls -td $CP/run_2026*/); do case "$1" in *"$d"*) ;; *) echo ${d%/}; return;; esac; done; }
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

bracket() {
  check_stop
  local LOG=$CP/bracket_twinkle_$1.log; : > $LOG
  stage "BRACKET twinkle $1 [Base -> s0 final -> Base]"
  echo "=== B1 $(date +%T)" | tee -a $LOG; take $LOG --baseline twinkle-short.mxl
  echo "=== POL $(date +%T)" | tee -a $LOG; take $LOG $POL twinkle-short.mxl
  echo "=== B2 $(date +%T)" | tee -a $LOG; take $LOG --baseline twinkle-short.mxl
  local -a d; d=("${(@f)$(takedirs "$LOG")}"); echo "bracket takes: ${#d[@]}"
  if [ ${#d[@]} -eq 3 ]; then
    posthoc --label "bracket twinkle $1 (RL = s0 final; pair1 vs pre-Base, pair2 vs post-Base)" Base1=${d[1]} RL1=${d[2]} Base2=${d[3]} RL2=${d[2]}
  else echo "bracket $1 incomplete (${#d[@]}/3), continuing"; fi
}

finish_seed() {  # finish_seed RUN PIECE STEM SEED SWL TAG  (everything after training)
  local RUN=$1 PIECE=$2 STEM=$3 SEED=$4 SWL=$5 TAG=$6
  local FINAL=$RUN/sac_piece_${STEM}_final
  if [ ! -f "$FINAL.zip" ]; then stage "NO FINAL CKPT for $TAG seed=$SEED, chain halted"; exit 1; fi
  echo "final ckpt: $FINAL.zip  ckpts: $(ls $RUN | grep -c '^ckpt_ep')  strokes: $(wc -l < $RUN/stroke_log.jsonl)"
  check_stop
  if [ -f "$SP/SKIP_SWEEP" ]; then stage "SWEEP $TAG seed=$SEED skipped (SKIP_SWEEP)"; else
    stage "SWEEP $TAG seed=$SEED (select_best --last $SWL)"
    "$PY" -u rl/select_best.py $RUN --piece $PIECE --real --last $SWL > $CP/select_${TAG}_alt2_s${SEED}.log 2>&1
    echo "sweep rc=$?"; grep -i "return\|winner\|best" $CP/select_${TAG}_alt2_s${SEED}.log | grep -v -i warning | tail -12
  fi
  check_stop
  stage "AB $TAG seed=$SEED live Base vs FINAL x3"
  local AB=$CP/ab_${TAG}_alt2_s${SEED}.log; : > $AB
  local k; for k in 1 2 3; do
    echo "=== Base$k $(date +%T)" | tee -a $AB; take $AB --baseline $PIECE
    echo "=== RL$k $(date +%T)" | tee -a $AB;   take $AB $FINAL $PIECE
  done
  local -a d; d=("${(@f)$(takedirs "$AB")}"); echo "takes: ${#d[@]}"
  if [ ${#d[@]} -ne 6 ]; then stage "A/B produced ${#d[@]} takes (expected 6), chain halted"; exit 1; fi
  stage "POSTHOC $TAG seed=$SEED dual judge"
  posthoc --label "ALT2 $TAG seed $SEED FINAL vs Base, same session" Base1=${d[1]} RL1=${d[2]} Base2=${d[3]} RL2=${d[4]} Base3=${d[5]} RL3=${d[6]}
  stage "C1 $TAG seed=$SEED ckpt_entropy (offline)"
  nice "$PY" rl/ckpt_entropy.py $RUN 2>&1 | grep -v -i warning
  stage "DONE $TAG seed=$SEED"
  bracket "after_${TAG}_s${SEED}"
}

run_seed() {  # run_seed PIECE STEM BUDGET REFEVERY SEED SWEEPLAST TAG
  local PIECE=$1 STEM=$2 BUDGET=$3 REFEV=$4 SEED=$5 SWL=$6 TAG=$7
  check_stop
  local before=$(ls -d $CP/run_2026*/ 2>/dev/null)
  stage "RUN $TAG seed=$SEED launch ($PIECE, $BUDGET strokes, ref-every $REFEV)"
  env $ALT2 "$PY" rl/train_supervisor.py -- $PIECE --real --timesteps $BUDGET --ckpt-every 10 \
      --save-stroke-audio --no-calibrated-dynamics --reference-every $REFEV --seed $SEED \
      > $CP/${TAG}_alt2_s${SEED}_20260902.log 2>&1
  stage "RUN $TAG seed=$SEED supervisor exited rc=$?"
  local RUN=$(newest_run "$before"); echo "run dir: $RUN"
  [ -z "$RUN" ] && { stage "NO RUN DIR for $TAG seed=$SEED, chain halted"; exit 1; }
  finish_seed $RUN $PIECE $STEM $SEED $SWL $TAG
}

resume_seed1() {  # seed 1: 84/100 episodes banked in run_20260902_153317 (ckpt_ep0010..0080 + latest); 704 strokes left
  local SRC=$CP/run_20260902_153317
  check_stop
  local before=$(ls -d $CP/run_2026*/ 2>/dev/null)
  stage "RESUME maysong_m0-11 seed=1 from $SRC (704 strokes = 16 eps left)"
  env $ALT2 "$PY" rl/train_supervisor.py -- maysong_m0-11.mxl --real --timesteps 704 --ckpt-every 10 \
      --save-stroke-audio --no-calibrated-dynamics --reference-every 5 --seed 1 --resume-run $SRC \
      > $CP/maysong_m0-11_alt2_s1_resume_20260902.log 2>&1
  stage "RESUME seed=1 supervisor exited rc=$?"
  local RUN=$(newest_run "$before"); echo "run dir: $RUN"
  [ -z "$RUN" ] && { stage "NO RUN DIR for resumed seed 1, chain halted"; exit 1; }
  # select_best / C1 read one dir: bring the banked numbered ckpts alongside the new ones (copy, never move)
  local f; for f in $SRC/ckpt_ep*.zip; do cp -n $f $RUN/; done
  echo "ckpts in $RUN after merge: $(ls $RUN | grep '^ckpt_ep' | tr '\n' ' ')"
  finish_seed $RUN maysong_m0-11.mxl maysong_m0-11 1 7 maysong_m0-11
}

check_stop
stage "YUNPIECE_M9-24 (AUDIT NEGATIVE) SEEDS 0,1,2 CHAIN start"
for SEED in 0 1 2; do run_seed yunpiece_m9-24.mxl yunpiece_m9-24 3200 5 $SEED 7 yunpiece_m9-24; done
stage "YUNPIECE_M9-24 SEEDS 0,1,2 CHAIN DONE"
