#!/usr/bin/env zsh
# ABLATION D (historical 9/6 protocol): "no learned reward" on VOCALISE, seeds 0/1/2, judged later by the audio-only tone
# model C and by both classifiers. Answers: does the learned human-rating reward add tone gain beyond the
# hand-designed terms alone?
#
# FIXED CHOICES (historical 9/6 protocol, do not drift):
#   * only the classifier reward coefficient is zeroed: CELLO_W_QUALITY=0 (rl/piece_env.py W_QUALITY override);
#     the classifier is still called every stroke, its score stays in the observation and in the logs,
#     every other reward weight, the budget, the ALT2 recipe and the executor are unchanged; no renormalisation.
#   * same recipe/budget as the full-reward vocalise runs of 9/3: 1400 strokes = 100 eps x 14, ref-every 5,
#     ckpt-every 10, FINAL checkpoint used (no selection), --save-stroke-audio, same recording + timeline files.
#   * SAME-DAY CONTROL (review 9/6): one extra FULL-reward training, seed 0, same budget, scheduled right after
#     ABL seed 0, so one FULL/ABL pair shares the training day. The three 9/3 FULLs stay in the planned evaluation;
#     the new pair is a matched supplement (one pair, not three).
#   * evaluation per seed in ONE session, interleaved x3: Base then the policy arms in rotating order
#     (seed 0: FULL / FULLNEW / ABL, each arm once per position; seeds 1, 2: FULL / ABL alternating).
#     posthoc dual judge: FULL-Base, ABL-Base, FULL-ABL (+ FULLNEW-Base, FULLNEW-ABL, FULLNEW-FULL for seed 0).
#     C scores the same take dirs offline afterwards (the private tone_model_C/score_takes.py --d-tsv).
#
# Usage:  rl/chain_D_vocalise_20260908.sh train   # ABL s0, FULLNEW s0, ABL s1, ABL s2 (~45 min each, ~3 h)
#         rl/chain_D_vocalise_20260908.sh eval    # s0: 12 live takes (~9 min); s1, s2: 9 takes each (~7 min)
#         rl/chain_D_vocalise_20260908.sh all
# Before running: robot connected (192.168.1.100), Scarlett on, caffeinate on, then the session gain offset:
#     "${CELLO_PYTHON:-.venv/bin/python}" rl/calibrate_gain.py --measure
# Control files in $SP: STOP_CHAIN (halt after the current stage), PAUSE_CHAIN (wait).
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
MODE=${1:-all}
STOP=$SP/STOP_CHAIN; PAUSE=$SP/PAUSE_CHAIN
CP=rl/checkpoints_piece
[[ -d "$CP" && -w "$CP" ]] || { print -u2 -- "Restore an existing writable $ROOT/$CP directory before enabling this recipe."; exit 2; }
# These inputs are private or repository-local; this script does not download or replace them.
REQUIRED_INPUTS=(
  "rl/calibrate_gain.py"
  "rl/checkpoints_piece/run_20260903_151543/sac_piece_vocalise_final.zip"
  "rl/checkpoints_piece/run_20260903_155747/sac_piece_vocalise_final.zip"
  "rl/checkpoints_piece/run_20260903_164107/sac_piece_vocalise_final.zip"
  "rl/ckpt_entropy.py"
  "rl/perform.py"
  "rl/piece_env.py"
  "rl/posthoc_heldout.py"
  "rl/train_supervisor.py"
  "vocalise.mxl"
)
for required_input in "${REQUIRED_INPUTS[@]}"; do
  [[ -f "$required_input" && -r "$required_input" ]] || { print -u2 -- "Missing required historical input: $required_input"; exit 2; }
done
PIECE=vocalise.mxl; STEM=vocalise; BUDGET=1400; REFEV=5
SEM=(CELLO_ENVELOPE_MIN_S=0.5 CELLO_PHRASE_SCORING=1 CELLO_TIMING_PENALTY=1)
ALT2=(CELLO_ALT_LOOP=1 CELLO_ENVELOPE_MIN_S=0.5 CELLO_TIMING_PENALTY=1 CELLO_PHRASE_SCORING=1 CELLO_ZERO_INIT=1 CELLO_TARGET_ENTROPY_SCALE=0.5 CELLO_ENT_COEF_INIT=0.005)
F="perform ->\|contact\|dBFS\|tempo_ratio\|mean_quality_true\|mean_score_ms\|Traceback\|Error\|LOST\|recover\|ckpt\|loaded"
typeset -A FULLRUN; FULLRUN=(0 run_20260903_151543 1 run_20260903_155747 2 run_20260903_164107)   # 9/3 full-reward FINALs (Table II)
RUNS=$CP/ablD_vocalise_runs.txt        # "ARM<TAB>seed<TAB>run_dir", ARM = ABL | FULLNEW; appended by train, read by eval
TSV=$CP/abD_vocalise_takes_20260908.tsv

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
posthoc() { "$PY" rl/posthoc_heldout.py "$@" 2>&1 | grep -v -i warning | grep "mean_tone=\|delta\|mean, n\|condition"; }
recorded_run() { awk -F '\t' -v a="$1" -v s="$2" '$1==a && $2==s {r=$3} END{print r}' "$RUNS" 2>/dev/null; }

train_one() {  # train_one ARM SEED   (ARM = ABL -> CELLO_W_QUALITY=0 ; FULLNEW -> default 0.50)
  local ARM=$1 SEED=$2; check_stop
  local -a EXTRA; [ "$ARM" = ABL ] && EXTRA=(CELLO_W_QUALITY=0) || EXTRA=()
  local before=$(ls -d $CP/run_2026*/ 2>/dev/null); local LOG=$CP/vocalise_${ARM}_s${SEED}_20260908.log
  stage "D TRAIN $ARM vocalise seed=$SEED launch (${EXTRA:-default W_QUALITY}, $BUDGET strokes, ref-every $REFEV)"
  env $ALT2 $EXTRA "$PY" rl/train_supervisor.py -- $PIECE --real --timesteps $BUDGET --ckpt-every 10 \
      --save-stroke-audio --no-calibrated-dynamics --reference-every $REFEV --seed $SEED > $LOG 2>&1
  stage "D TRAIN $ARM seed=$SEED supervisor exited rc=$?"
  local RUN=$(newest_run "$before"); echo "run dir: $RUN"
  [ -z "$RUN" ] && { stage "NO RUN DIR for $ARM seed $SEED, chain halted"; exit 1; }
  [ -f "$RUN/sac_piece_${STEM}_final.zip" ] || { stage "NO FINAL CKPT in $RUN, chain halted"; exit 1; }
  echo "final ckpt: $RUN/sac_piece_${STEM}_final.zip  strokes: $(wc -l < $RUN/stroke_log.jsonl)  arm $ARM (${EXTRA:-W_QUALITY 0.50})"
  printf "%s\t%s\t%s\n" $ARM $SEED $RUN >> $RUNS
  stage "C1 ckpt_entropy $ARM seed=$SEED (offline)"; nice "$PY" rl/ckpt_entropy.py $RUN 2>&1 | grep -v -i warning | tail -3
}

eval_seed() {  # eval_seed SEED : Base + policy arms, interleaved x3 in one session
  local SEED=$1; check_stop
  local ABLRUN=$(recorded_run ABL $SEED); local NEWRUN=$(recorded_run FULLNEW $SEED)
  [ -z "$ABLRUN" ] && { stage "no ABL run recorded for seed $SEED in $RUNS, skipping"; return; }
  typeset -A CK; CK=(FULL $CP/${FULLRUN[$SEED]}/sac_piece_${STEM}_final ABL $ABLRUN/sac_piece_${STEM}_final)
  local -a ARMS; ARMS=(FULL ABL)
  if [ -n "$NEWRUN" ]; then CK[FULLNEW]=$NEWRUN/sac_piece_${STEM}_final; ARMS=(FULL FULLNEW ABL); fi
  local a; for a in $ARMS; do [ -f "${CK[$a]}.zip" ] || { stage "MISSING ckpt for $a: ${CK[$a]}.zip"; exit 1; }; done
  local LOG=$CP/abD_vocalise_s${SEED}_20260908.log; : > $LOG
  stage "D EVAL vocalise seed=$SEED arms ${ARMS[*]} x3 interleaved (FULL=${FULLRUN[$SEED]} ABL=$(basename $ABLRUN)${NEWRUN:+ FULLNEW=$(basename $NEWRUN)})"
  local k n=${#ARMS[@]} i order
  for k in 1 2 3; do
    echo "=== Base$k $(date +%T)" | tee -a $LOG; take $LOG --baseline $PIECE
    for i in $(seq 0 $((n-1))); do            # rotate: round k starts at arm index (k-1), so every arm sits once at every position
      a=${ARMS[$(( (i + k - 1) % n + 1 ))]}
      echo "=== ${a}$k $(date +%T)" | tee -a $LOG; take $LOG ${CK[$a]} $PIECE
    done
  done
  # map takes to arms from the log order (stage line, then its perform -> line)
  typeset -A D; local cur line
  while IFS= read -r line; do
    case "$line" in "=== "*) cur=${${line#=== }%% *};; *"perform ->"*) D[$cur]=${line#*perform -> };; esac
  done < $LOG
  local expected=$((3 * (n + 1))); echo "takes mapped: ${#D[@]} (expected $expected)" | tee -a $LOG
  [ ${#D[@]} -eq $expected ] || { stage "eval incomplete (${#D[@]}/$expected), posthoc skipped; see $LOG"; return; }
  stage "POSTHOC seed=$SEED dual judge"
  local cmp; local -a CMPS; CMPS=("FULL Base" "ABL Base" "FULL ABL"); [ -n "$NEWRUN" ] && CMPS+=("FULLNEW Base" "FULLNEW ABL" "FULLNEW FULL")
  for cmp in $CMPS; do
    local X=${cmp%% *} Y=${cmp##* }
    posthoc --label "D s$SEED $X vs $Y" Base1=${D[${Y}1]} RL1=${D[${X}1]} Base2=${D[${Y}2]} RL2=${D[${X}2]} Base3=${D[${Y}3]} RL3=${D[${X}3]} | tee -a $LOG
  done
  local rec="seed $SEED"; for a in Base $ARMS; do rec="$rec\t$a ${D[${a}1]} ${D[${a}2]} ${D[${a}3]}"; done; printf "$rec\n" >> $TSV
  stage "D EVAL seed=$SEED DONE"
}

# Evaluation-only mode needs the persisted tab-separated training ledger and every primary arm.
# Ledger columns: ARM, seed (0/1/2), run directory relative to ROOT or absolute.
if [[ "$MODE" == eval ]]; then
  [[ -f "$RUNS" && -r "$RUNS" ]] || { print -u2 -- "Missing evaluation ledger: $RUNS"; exit 2; }
  for s in 0 1 2; do
    recorded=$(recorded_run ABL $s)
    [[ -n "$recorded" && -f "$recorded/sac_piece_${STEM}_final.zip" ]] || { print -u2 -- "Missing ABL final checkpoint for seed $s in $RUNS"; exit 2; }
    recorded=$(recorded_run FULLNEW $s)
    [[ -z "$recorded" || -f "$recorded/sac_piece_${STEM}_final.zip" ]] || { print -u2 -- "Missing recorded FULLNEW final checkpoint for seed $s in $RUNS"; exit 2; }
  done
fi

for s in 0 1 2; do [ -f "$CP/${FULLRUN[$s]}/sac_piece_${STEM}_final.zip" ] || { stage "MISSING 9/3 full-reward final for seed $s"; exit 1; }; done
case $MODE in
  train) train_one ABL 0; train_one FULLNEW 0; train_one ABL 1; train_one ABL 2 ;;
  eval)  for s in 0 1 2; do eval_seed $s; done ;;
  all)   train_one ABL 0; train_one FULLNEW 0; train_one ABL 1; train_one ABL 2; for s in 0 1 2; do eval_seed $s; done ;;
  *) echo "usage: $0 [train|eval|all]"; exit 2 ;;
esac
stage "D CHAIN ($MODE) DONE"
