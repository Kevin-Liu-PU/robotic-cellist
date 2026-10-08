#!/usr/bin/env zsh
# Night chain 9/2. Historical request at 23:0x on 9/1: run the full sequence with on-site coverage and operator handovers after 05:00.
# Waits for the twinkle ALT2 chain (pid $WAITPID) to exit, then runs WITHOUT gaps:
#   bracket#1 -> A8 twinkle compiled-vs-live x2 -> A11 reduced-guard (train 2100 + live A/B x2)
#   -> bracket#2 -> Popper gate -> Popper seed 0/1/2 [train 18200 -> sweep (--last 3, 1 ep, 1 confirm)
#   -> live A/B x3 Base vs FINAL -> posthoc dual judge -> C1 -> twinkle bracket]
#   -> Vocalise seed 0/1/2 (same block, budget 1700, ref-every 5, sweep --last 7) ONLY if $SP/VOCALISE_OK
#      exists; its content = piece file to use (the 8/25 17-stroke version, verified before the marker is written).
# Control files in $SP: STOP_CHAIN = exit at next boundary; PAUSE_CHAIN = wait at next boundary until removed;
#   SKIP_SWEEP = skip checkpoint sweeps. A6 omitted from the historical schedule. Nothing here uses --compile except A8.
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
  "rl/checkpoints_piece/chain_alt2_s1s2_20260901.log"
  "rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final.zip"
  "rl/ckpt_entropy.py"
  "rl/gate_regression.py"
  "rl/perform.py"
  "rl/posthoc_heldout.py"
  "rl/posthoc_score.py"
  "rl/reward_noise.py"
  "rl/select_best.py"
  "rl/train_supervisor.py"
  "twinkle-short.mxl"
  "yunpiece.mxl"
)
for required_input in "${REQUIRED_INPUTS[@]}"; do
  [[ -f "$required_input" && -r "$required_input" ]] || { print -u2 -- "Missing required historical input: $required_input"; exit 2; }
done
# An optional marker must contain a single readable score path relative to ROOT, or an absolute path.
if [[ -f "$SP/VOCALISE_OK" ]]; then
  VOC=$(cat "$SP/VOCALISE_OK")
  [[ -n "$VOC" && -f "$VOC" && -r "$VOC" ]] || { print -u2 -- "VOCALISE_OK must contain the verified readable score path."; exit 2; }
fi
POL=rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final   # twinkle seed-0 FINAL, fixed all campaign
SEM=(CELLO_ENVELOPE_MIN_S=0.5 CELLO_PHRASE_SCORING=1 CELLO_TIMING_PENALTY=1)   # take/gate semantics
ALT2=(CELLO_ALT_LOOP=1 CELLO_ENVELOPE_MIN_S=0.5 CELLO_TIMING_PENALTY=1 CELLO_PHRASE_SCORING=1 CELLO_ZERO_INIT=1 CELLO_TARGET_ENTROPY_SCALE=0.5 CELLO_ENT_COEF_INIT=0.005)
F="perform ->\|contact\|dBFS\|tempo_ratio\|mean_quality_true\|mean_score_ms\|Traceback\|Error\|LOST\|recover\|ckpt\|loaded"
WAITPID=${1:?Pass the current predecessor chain PID as the first argument.}
[[ "$WAITPID" == <1-> ]] || { print -u2 -- "The predecessor PID must be a positive integer."; exit 2; }

stage() { echo "=== $1 $(date +%H:%M:%S)"; }
check_stop() {
  while [ -f "$PAUSE" ]; do stage "PAUSE file present, waiting"; sleep 60; done
  if [ -f "$STOP" ]; then stage "STOP file found, chain halted"; exit 0; fi
}
newest_run() {  # $1 = "before" listing; prints the newest run dir not in it
  local d; for d in $(ls -td $CP/run_2026*/); do case "$1" in *"$d"*) ;; *) echo ${d%/}; return;; esac; done
}
take() {  # take LOG perform-args... ; one retry if no "perform ->" line
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

bracket() {  # twinkle drift bracket: Base -> seed-0 FINAL -> Base ; $1 = tag
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
  local FINAL=$RUN/sac_piece_${STEM}_final
  if [ -z "$RUN" ] || [ ! -f "$FINAL.zip" ]; then stage "NO FINAL CKPT for $TAG seed=$SEED, chain halted"; exit 1; fi
  echo "final ckpt: $FINAL.zip  ckpts: $(ls $RUN | grep -c '^ckpt_ep')  strokes: $(wc -l < $RUN/stroke_log.jsonl)"

  check_stop
  if [ -f "$SP/SKIP_SWEEP" ]; then stage "SWEEP $TAG seed=$SEED skipped (SKIP_SWEEP)"; else
    stage "SWEEP $TAG seed=$SEED (diagnostic, --last $SWL)"
    if [ "$SWL" -le 3 ]; then EXTRA=(--episodes 1 --confirm-episodes 1); else EXTRA=(); fi
    "$PY" -u rl/select_best.py $RUN --piece $PIECE --real --last $SWL $EXTRA > $CP/select_${TAG}_alt2_s${SEED}.log 2>&1
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

# ---------------------------------------------------------------- 0. wait for the twinkle chain
stage "NIGHT CHAIN start, waiting for pid $WAITPID (twinkle ALT2 chain)"
while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
stage "twinkle chain exited"; tail -3 $CP/chain_alt2_s1s2_20260901.log

# ---------------------------------------------------------------- 1. bracket #1 (closing point of 9/1)
bracket "0902_after_twinkle_s2"

# ---------------------------------------------------------------- 2. A8 twinkle compiled-vs-live x2 (seed-0 FINAL)
check_stop
stage "A8 twinkle compiled-vs-live x2 (policy = $POL)"
A8=$CP/a8_twinkle_20260902.log; : > $A8
for k in 1 2; do
  echo "=== COMPILED$k $(date +%T)" | tee -a $A8; take $A8 --compile $POL twinkle-short.mxl
  echo "=== LIVE$k $(date +%T)"     | tee -a $A8; take $A8 $POL twinkle-short.mxl
done
d=("${(@f)$(takedirs "$A8")}"); echo "A8 takes: ${#d[@]}"
if [ ${#d[@]} -eq 4 ]; then
  posthoc --label "A8 twinkle: Base=compiled, RL=live, same policy (s0 final)" Base1=${d[1]} RL1=${d[2]} Base2=${d[3]} RL2=${d[4]}
  for i in 1 3; do j=$((i+1)); echo "--- posthoc_score pair ${d[$i]:t} vs ${d[$j]:t}"
    "$PY" rl/posthoc_score.py ${d[$i]} ${d[$j]} 2>&1 | grep -v -i warning | grep -i "tempo\|tone\|mean\|delta" | head -12; done
else echo "A8 incomplete (${#d[@]}/4), continuing"; fi

# ---------------------------------------------------------------- 3. A11 reduced-guard twinkle (2100, seed 0) + live A/B x2
check_stop
before=$(ls -d $CP/run_2026*/ 2>/dev/null)
stage "A11 reduced-guard twinkle train 2100 seed 0 (TIMING_PENALTY=0 PHRASE_SCORING=0, else = guarded s0)"
CELLO_ALT_LOOP=1 CELLO_TIMING_PENALTY=0 CELLO_PHRASE_SCORING=0 CELLO_TARGET_ENTROPY_SCALE=0.5 CELLO_ENT_COEF_INIT=0.005 \
  "$PY" rl/train_supervisor.py -- twinkle-short.mxl --real --timesteps 2100 --ckpt-every 10 \
  --save-stroke-audio --no-calibrated-dynamics --reference-every 5 --seed 0 > $CP/twinkle_a11_20260902.log 2>&1
stage "A11 supervisor exited rc=$?"
grep -i -m3 "timing_penalty\|phrase\|reference" $CP/twinkle_a11_20260902.log
RUN=$(newest_run "$before"); echo "A11 run dir: $RUN"; A11FINAL=$RUN/sac_piece_twinkle-short_final
if [ -n "$RUN" ] && [ -f "$A11FINAL.zip" ]; then
  echo "A11 final: $A11FINAL.zip  ckpts: $(ls $RUN | grep -c '^ckpt_ep')  strokes: $(wc -l < $RUN/stroke_log.jsonl)"
  check_stop
  stage "A11 live A/B x2 Base vs reduced-guard FINAL"
  AB=$CP/ab_twinkle_a11_20260902.log; : > $AB
  for k in 1 2; do
    echo "=== Base$k $(date +%T)" | tee -a $AB; take $AB --baseline twinkle-short.mxl
    echo "=== RL$k $(date +%T)"   | tee -a $AB; take $AB $A11FINAL twinkle-short.mxl
  done
  d=("${(@f)$(takedirs "$AB")}"); echo "A11 takes: ${#d[@]}"
  if [ ${#d[@]} -eq 4 ]; then posthoc --label "A11 reduced-guard twinkle FINAL vs Base, same session" Base1=${d[1]} RL1=${d[2]} Base2=${d[3]} RL2=${d[4]}
  else echo "A11 A/B incomplete (${#d[@]}/4), continuing"; fi
  stage "C1 A11 ckpt_entropy"; nice "$PY" rl/ckpt_entropy.py $RUN 2>&1 | grep -v -i warning
else stage "A11 produced no final ckpt, continuing to Popper"; fi
stage "DONE A11"

# ---------------------------------------------------------------- 4. bracket #2, then Popper gate
bracket "0902_before_popper"
check_stop
stage "GATE Popper (reward_noise 6 repeats x 12 probes, 3 targets, raw rows)"
env $SEM "$PY" -u rl/reward_noise.py --piece yunpiece.mxl --real --repeats 6 --probes 12 \
  --no-calibrated-dynamics --out rl/gate_yunpiece_20260902.json > $CP/gate_yunpiece_20260902.log 2>&1
echo "gate rc=$?"; grep -i "wrote\|verdict\|GO\|Traceback" $CP/gate_yunpiece_20260902.log | grep -v -i warning | tail -8
"$PY" rl/gate_regression.py rl/gate_yunpiece_20260902.json --pooled --json rl/gate_yunpiece_20260902_snr.json \
  > $CP/gate_yunpiece_20260902_snr.log 2>&1
echo "gate_regression rc=$?"; grep -A 14 -i "POOLED" $CP/gate_yunpiece_20260902_snr.log | head -20
stage "GATE Popper done (pre-registered: train regardless; NO-GO = predicted-failure arm, logged in the day log)"

# ---------------------------------------------------------------- 5. Popper seeds 0, 1, 2
for SEED in 0 1 2; do run_seed yunpiece.mxl yunpiece 18200 10 $SEED 3 popper; done
stage "POPPER ALL SEEDS DONE"

# ---------------------------------------------------------------- 6. Vocalise seeds 0, 1, 2 (only with the verified file)
check_stop
if [ -f "$SP/VOCALISE_OK" ]; then
  VOC=$(cat "$SP/VOCALISE_OK"); VSTEM=${VOC:t:r}
  stage "GATE Vocalise ($VOC)"
  env $SEM "$PY" -u rl/reward_noise.py --piece $VOC --real --repeats 6 --probes 12 \
    --no-calibrated-dynamics --out rl/gate_${VSTEM}_20260902.json > $CP/gate_${VSTEM}_20260902.log 2>&1
  echo "gate rc=$?"; grep -i "wrote\|verdict\|GO\|Traceback" $CP/gate_${VSTEM}_20260902.log | grep -v -i warning | tail -8
  "$PY" rl/gate_regression.py rl/gate_${VSTEM}_20260902.json --pooled --json rl/gate_${VSTEM}_20260902_snr.json \
    > $CP/gate_${VSTEM}_20260902_snr.log 2>&1; grep -A 14 -i "POOLED" $CP/gate_${VSTEM}_20260902_snr.log | head -20
  for SEED in 0 1 2; do run_seed $VOC $VSTEM 1700 5 $SEED 7 vocalise; done
  stage "VOCALISE ALL SEEDS DONE"
else
  stage "VOCALISE_OK absent (8/25 file not verified): chain halted before Vocalise"
fi
stage "NIGHT CHAIN DONE"
