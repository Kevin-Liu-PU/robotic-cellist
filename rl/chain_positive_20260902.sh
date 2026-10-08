#!/usr/bin/env zsh
# Positive-case chain 9/2 08:xx (historical schedule): high-fill pieces first. vocalise (fill 0.98) then maysong (fill 0.86).
# Each piece: gate (reward_noise, raw rows) -> seed 0/1/2 [train 100 episodes -> sweep -> live A/B x3 -> posthoc dual judge -> C1 -> twinkle bracket].
# Starts immediately (robot idle since 07:30; Popper s1 invalid, s2 deferred).
# Control files in $SP: STOP_CHAIN = exit at next boundary; PAUSE_CHAIN = wait at next boundary until removed;
#   SKIP_SWEEP = skip checkpoint sweeps. A6 dropped (historical decision at 23:0x). Nothing here uses --compile except A8.
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
  "maysong.mxl"
  "rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final.zip"
  "rl/ckpt_entropy.py"
  "rl/gate_regression.py"
  "rl/perform.py"
  "rl/posthoc_heldout.py"
  "rl/reward_noise.py"
  "rl/select_best.py"
  "rl/train_supervisor.py"
  "twinkle-short.mxl"
  "vocalise.mxl"
)
for required_input in "${REQUIRED_INPUTS[@]}"; do
  [[ -f "$required_input" && -r "$required_input" ]] || { print -u2 -- "Missing required historical input: $required_input"; exit 2; }
done
POL=rl/checkpoints_piece/run_20260901_185917/sac_piece_twinkle-short_final   # twinkle seed-0 FINAL, fixed all campaign
SEM=(CELLO_ENVELOPE_MIN_S=0.5 CELLO_PHRASE_SCORING=1 CELLO_TIMING_PENALTY=1)   # take/gate semantics
ALT2=(CELLO_ALT_LOOP=1 CELLO_ENVELOPE_MIN_S=0.5 CELLO_TIMING_PENALTY=1 CELLO_PHRASE_SCORING=1 CELLO_ZERO_INIT=1 CELLO_TARGET_ENTROPY_SCALE=0.5 CELLO_ENT_COEF_INIT=0.005)
F="perform ->\|contact\|dBFS\|tempo_ratio\|mean_quality_true\|mean_score_ms\|Traceback\|Error\|LOST\|recover\|ckpt\|loaded"
# (no wait: nothing else owns the robot)
NIGHTLOG=$CP/chain_night_20260902.log

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

# ---------------------------------------------------------------- 0. wait for the night chain (Popper) to exit
# ---------------------------------------------------------------- 1. Vocalise gate
check_stop
stage "POSITIVE CHAIN start (vocalise -> maysong)"
stage "GATE Vocalise (vocalise.mxl, 14 strokes; 6 repeats x 12 probes, 3 targets, raw rows)"
env $SEM "$PY" -u rl/reward_noise.py --piece vocalise.mxl --real --repeats 6 --probes 12 \
  --no-calibrated-dynamics --out rl/gate_vocalise_20260902.json > $CP/gate_vocalise_20260902.log 2>&1
echo "gate rc=$?"; grep -i "wrote\|verdict\|GO\|Traceback" $CP/gate_vocalise_20260902.log | grep -v -i warning | tail -8
"$PY" rl/gate_regression.py rl/gate_vocalise_20260902.json --pooled --json rl/gate_vocalise_20260902_snr.json \
  > $CP/gate_vocalise_20260902_snr.log 2>&1
echo "gate_regression rc=$?"; grep -A 14 -i "POOLED" $CP/gate_vocalise_20260902_snr.log | head -20
stage "GATE Vocalise done (pre-registered: train regardless)"
# ---------------------------------------------------------------- 2. Vocalise seeds 0, 1, 2 (budget 1400 = 100 eps x 14, ref-every 5, sweep --last 7)
for SEED in 0 1 2; do run_seed vocalise.mxl vocalise 1400 5 $SEED 7 vocalise; done
stage "VOCALISE ALL SEEDS DONE"
# ---------------------------------------------------------------- 3. Maysong gate
check_stop
stage "GATE Maysong (maysong.mxl, 88 strokes; 6 repeats x 12 probes, 3 targets, raw rows)"
env $SEM "$PY" -u rl/reward_noise.py --piece maysong.mxl --real --repeats 6 --probes 12 \
  --no-calibrated-dynamics --out rl/gate_maysong_20260902.json > $CP/gate_maysong_20260902.log 2>&1
echo "gate rc=$?"; grep -i "wrote\|verdict\|GO\|Traceback" $CP/gate_maysong_20260902.log | grep -v -i warning | tail -8
"$PY" rl/gate_regression.py rl/gate_maysong_20260902.json --pooled --json rl/gate_maysong_20260902_snr.json \
  > $CP/gate_maysong_20260902_snr.log 2>&1
echo "gate_regression rc=$?"; grep -A 14 -i "POOLED" $CP/gate_maysong_20260902_snr.log | head -20
stage "GATE Maysong done (pre-registered: train regardless)"
# ---------------------------------------------------------------- 4. Maysong seeds 0, 1, 2 (budget 8800 = 100 eps x 88, ref-every 10, sweep --last 3 fast)
for SEED in 0 1 2; do run_seed maysong.mxl maysong 8800 10 $SEED 3 maysong; done
stage "MAYSONG ALL SEEDS DONE"
stage "POSITIVE CHAIN DONE"
