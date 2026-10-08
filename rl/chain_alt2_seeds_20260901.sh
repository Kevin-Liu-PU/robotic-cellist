#!/usr/bin/env zsh
# ALT2 twinkle convergence chain (historical approval at 21:1x): for seed in 1 2:
#   train (supervisor, identical to run 1 + --seed) -> sweep (select_best, diagnostic)
#   -> live A/B x3 Base vs FINAL -> posthoc dual judge -> C1 entropy check.
# Halts between stages if $STOP exists; halts on missing final ckpt or <6 takes.
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
STOP=$SP/STOP_CHAIN
CP=rl/checkpoints_piece
[[ -d "$CP" && -w "$CP" ]] || { print -u2 -- "Restore an existing writable $ROOT/$CP directory before enabling this recipe."; exit 2; }
# These inputs are private or repository-local; this script does not download or replace them.
REQUIRED_INPUTS=(
  "$SP/ckpt_entropy.py"
  "rl/perform.py"
  "rl/posthoc_heldout.py"
  "rl/select_best.py"
  "rl/train_supervisor.py"
  "twinkle-short.mxl"
)
for required_input in "${REQUIRED_INPUTS[@]}"; do
  [[ -f "$required_input" && -r "$required_input" ]] || { print -u2 -- "Missing required historical input: $required_input"; exit 2; }
done
command -v caffeinate >/dev/null 2>&1 || { print -u2 -- "This historical recipe requires the macOS caffeinate command."; exit 2; }
F="perform ->\|contact\|dBFS\|tempo_ratio\|mean_quality_true\|mean_score_ms\|Traceback\|Error\|LOST\|recover\|ckpt\|loaded"
stage() { echo "=== $1 $(date +%H:%M:%S)"; }
check_stop() { if [ -f "$STOP" ]; then stage "STOP file found, chain halted"; exit 0; fi }

for SEED in 1 2; do
  check_stop
  before=$(ls -d $CP/run_2026*/ 2>/dev/null)
  stage "RUN seed=$SEED launch"
  CELLO_ALT_LOOP=1 CELLO_ENVELOPE_MIN_S=0.5 CELLO_TIMING_PENALTY=1 CELLO_PHRASE_SCORING=1 CELLO_ZERO_INIT=1 CELLO_TARGET_ENTROPY_SCALE=0.5 CELLO_ENT_COEF_INIT=0.005 \
    caffeinate -dims "$PY" rl/train_supervisor.py -- twinkle-short.mxl --real --timesteps 4200 --ckpt-every 10 --save-stroke-audio --no-calibrated-dynamics --reference-every 5 --seed $SEED \
    > $CP/twinkle_alt2_s${SEED}_20260901.log 2>&1
  stage "RUN seed=$SEED supervisor exited rc=$?"
  RUN=""
  for d in $(ls -td $CP/run_2026*/); do
    case "$before" in *"$d"*) ;; *) RUN=${d%/}; break;; esac
  done
  echo "run dir: $RUN"
  FINAL=$RUN/sac_piece_twinkle-short_final
  if [ -z "$RUN" ] || [ ! -f "$FINAL.zip" ]; then stage "NO FINAL CKPT for seed=$SEED, chain halted"; exit 1; fi
  echo "final ckpt: $FINAL.zip  ckpts: $(ls $RUN | grep -c '^ckpt_ep')  strokes: $(wc -l < $RUN/stroke_log.jsonl)"

  check_stop
  stage "SWEEP seed=$SEED (diagnostic select_best, --last 7)"
  "$PY" -u rl/select_best.py $RUN --piece twinkle-short.mxl --real --last 7 > $CP/select_twinkle_alt2_s${SEED}.log 2>&1
  echo "sweep rc=$?"; grep -i "return\|winner\|best" $CP/select_twinkle_alt2_s${SEED}.log | grep -v -i warning | tail -12

  check_stop
  stage "AB seed=$SEED live Base vs FINAL x3"
  export CELLO_ENVELOPE_MIN_S=0.5 CELLO_PHRASE_SCORING=1 CELLO_TIMING_PENALTY=1
  AB=$CP/ab_twinkle_alt2_s${SEED}.log; : > $AB
  for k in 1 2 3; do
    echo "=== Base$k $(date +%H:%M:%S)" | tee -a $AB
    "$PY" -u rl/perform.py --real --tempo-scale 1.0 --baseline twinkle-short.mxl 2>&1 | grep -v -i warning | grep -i "$F" | tee -a $AB
    echo "=== RL$k $(date +%H:%M:%S)" | tee -a $AB
    "$PY" -u rl/perform.py --real --tempo-scale 1.0 $FINAL twinkle-short.mxl 2>&1 | grep -v -i warning | grep -i "$F" | tee -a $AB
  done
  unset CELLO_ENVELOPE_MIN_S CELLO_PHRASE_SCORING CELLO_TIMING_PENALTY
  dirs=("${(@f)$(sed -n 's/^.*perform ->[[:space:]]*//p' "$AB")}")
  echo "takes: ${#dirs[@]}"
  if [ ${#dirs[@]} -ne 6 ]; then stage "A/B produced ${#dirs[@]} takes (expected 6), chain halted"; exit 1; fi

  stage "POSTHOC seed=$SEED dual judge"
  "$PY" rl/posthoc_heldout.py --label "ALT2 seed $SEED FINAL vs Base, same session" \
    Base1=${dirs[1]} RL1=${dirs[2]} Base2=${dirs[3]} RL2=${dirs[4]} Base3=${dirs[5]} RL3=${dirs[6]} 2>&1 | grep -v -i warning | grep "mean_tone=\|delta\|mean, n\|condition"

  stage "C1 seed=$SEED ckpt_entropy (offline)"
  nice "$PY" "$SP/ckpt_entropy.py" $RUN 2>&1 | grep -v -i warning
  stage "DONE seed=$SEED"
done
stage "CHAIN DONE"
