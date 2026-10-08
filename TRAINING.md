# Running a training run

Run commands from the repository root in the configured project environment.
The examples use the retained root-level score `yunpiece.mxl`; replace it with
an explicit path to another available score when needed. Confirm that the score
exists before starting any hardware command. MusicXML (`.mxl`, `.musicxml`,
`.xml`) preserves written bowings, slurs and dynamics; MIDI has a different
score representation.

## Required inputs

| Mode | Required inputs |
| --- | --- |
| Mock | A score and the Python planning/training dependencies. The default mock executor and scorer do not require a robot, microphone or classifier checkpoint. |
| Real | The score, hardware dependencies, configured controller and taught poses, working audio input, a compatible classifier checkpoint, and loudness calibration appropriate to the microphone chain. |
| Resume or evaluation | An explicitly selected private run directory, its model and any available replay buffer, and the settings used for that run. These artifacts are not example inputs supplied with the code. |

Mock training still requires the planning and training libraries, including
PyTorch/Stable-Baselines3 and the audio libraries imported by the planner. The
examples disable calibrated dynamics; enabling it also requires
`SoundClassifier/checkpoints/loudness_model.json`. Gain calibration requires this
JSON even in mock mode.

Mock execution checks software wiring; it does not establish physical safety,
audio quality or hardware readiness. Before real execution, review controller
limits, calibration, recovery procedures and the emergency stop with an operator
present. Do not fabricate or bypass required session inputs.

Use `python` from the activated project environment. The commands below do not
depend on a machine-specific virtual-environment path or a platform-specific
sleep-prevention utility. Uppercase arguments such as `STROKE_BUDGET` and
`RUN_DIR` are placeholders to replace before running.

Training defaults to mock mode. `calibrate_gain.py --measure`, `reward_noise.py`
and `select_best.py` default to real hardware unless explicitly given `--mock`;
`ab_compare.py` only supports real hardware. Keep the mode explicit in commands.

## 1. Check the mock path

```bash
python rl/train_piece_logged.py yunpiece.mxl --mock --timesteps 10 --no-calibrated-dynamics
```

This short example still creates a private run directory and logs. Adding
`--real-scorer` requires the classifier and its dependencies even with a mock
executor; it is a separate scoring check, not a microphone or robot test.

## 2. Prepare the real setup

Verify the configured robot endpoint, installation, taught poses and audio input.
Check that the selected classifier checkpoint is available and compatible.
Training accepts `--classifier-checkpoint CHECKPOINT_PATH`; helper tools may use
the classifier's configured default, so verify that they resolve the intended
model as well.

Microphone-chain gain calibration measures the difference between predicted and
recorded loudness. The following command **moves the robot and records audio**:

```bash
python rl/calibrate_gain.py --measure --real --piece yunpiece.mxl
```

Inspect the estimate before using `--write`, which updates
`SoundClassifier/checkpoints/loudness_model.json`. Recheck calibration when the
microphone, gain or interface changes. Mock calibration is only a software check
and must not be written as a hardware calibration.

## 3. Check reward repeatability

```bash
python rl/reward_noise.py --piece yunpiece.mxl --real --no-calibrated-dynamics --out rl/gate.json
```

This command also operates the robot. Inspect reward variability for repeated
actions, differences across actions and the per-term diagnostics before training.
Unexpected or constant scores require investigating audio, calibration, model
compatibility and the reward configuration. This report is not a hardware safety
certificate or an automatic authorization to proceed.

## 4. Train and save

```bash
python -u rl/train_piece_logged.py yunpiece.mxl --real --timesteps STROKE_BUDGET --ckpt-every 10 --no-calibrated-dynamics
```

`--timesteps` specifies a training-step budget, with one step per planned stroke.
The number of strokes per episode depends on the score and planning settings.
Inspect the run header for the resolved model, dynamics mode, gain offset and
environment options. Stop and investigate unexpectedly zero dynamics scores,
out-of-zone readings or stalled progress rather than assuming they are normal
training behaviour.

Two save settings serve different purposes:

- `--ckpt-every` controls numbered model checkpoints in episodes.
- `--save-every` controls latest-model and replay-buffer saves in training steps;
  the implementation defers these saves to an episode boundary.

A handled Ctrl-C requests saving the model and replay buffer. An abrupt process
termination or connection failure may leave only the last completed save. Verify
the resulting artifacts before resuming.

## 5. Resume an explicitly selected run

After a hardware interruption, follow the lab recovery procedure and verify the
robot and audio state before restarting. Select the intended run directory from
the run output instead of assuming that the newest directory is the correct one.

```bash
python -u rl/train_piece_logged.py yunpiece.mxl --real --timesteps REMAINING_STROKES --ckpt-every 10 --no-calibrated-dynamics --resume-run RUN_DIR
```

`--resume-run` derives the checkpoint and episode offset from that directory.
Do not combine it with manually supplied `--resume` or `--episode-offset`.
Inspect the printed checkpoint and whether its replay buffer exists. Reapply the
original `CELLO_*` settings and the same score/planning options.

Compute the remaining budget from completed training strokes across all run
segments. `stroke_log.jsonl` contains a header and can contain reference summary
rows; its line count minus one is not a reliable training-step count. Checkpoint
progress counters restart for a resumed process. Count records containing
`stroke_seq` across the selected segments, and verify the saved checkpoint and
buffer separately: a logged stroke does not guarantee that its transition was
persisted. Each resume creates a new run directory.

## 6. Select and compare

These examples require an existing private run and the reviewed hardware setup:

```bash
python rl/select_best.py RUN_DIR --piece yunpiece.mxl --real --episodes 2
python rl/ab_compare.py RUN_DIR --piece yunpiece.mxl --gap 8 --no-calibrated-dynamics
```

`ab_compare.py` uses real hardware and has no mock mode. Keep its dynamics mode
consistent with training: use `--no-calibrated-dynamics` for an uncalibrated run,
or the corresponding calibrated setting when that was used.

`--tempo-scale` above 1 slows the score down.
`--no-calibrated-dynamics` uses the planner's linear dynamic-to-speed rule instead
of inverting the loudness model to aim each note at a target dB level.

## Environment options

| Variable | Purpose |
| --- | --- |
| `CELLO_ACCEL_MAX` | Acceleration ceiling; default 5.5. |
| `CELLO_BOW_CONFIG` | Bow configuration: standard, board, bridge, topangle or botangle. |
| `CELLO_DYNAMIC_MAP` | Rewrite dynamics, for example `p=f`. |
| `CELLO_ARTICULATION_REF` | Staccato crispness; lower values create larger gaps. |
| `CELLO_SPEED_TRIM` / `CELLO_DEPTH_TRIM_MM` | Scale planned speed / add depth. |

## Alternate loop

Set `CELLO_ALT_LOOP=1` in the process environment before launching the same
training command. In a POSIX shell use `export CELLO_ALT_LOOP=1`; in PowerShell
use `$env:CELLO_ALT_LOOP = "1"`. Unset it when returning to the stock loop.

| Option | Behaviour |
| --- | --- |
| Timing penalty | Accounts for the dispatch cost of shaping a stroke. |
| Zero initialization | Zeroes the actor mean, initializes exploration through `log_std`, and sets the initial entropy coefficient to 0.1. |
| Reference episodes | Records zero-residual reference takes at the configured episode interval, marked with `"reference": true`. |
| Phrase scoring | Uses a full classifier window rather than a note-length fragment. |

The alternate default raises `ENVELOPE_MIN_DURATION` from 0.40 to 0.50 seconds
to match the classifier window. This is a configuration choice, not a claim of
measured perceptual performance.

Check that the header prints `ALT loop: timing_penalty=True zero_init=True
phrase_scoring=True`. Individual settings are available through
`CELLO_TIMING_PENALTY`, `CELLO_ZERO_INIT`, `CELLO_PHRASE_SCORING`,
`CELLO_REFERENCE_EVERY` and `CELLO_ENVELOPE_MIN_S`.

Keep generated recordings, logs, reports and checkpoints in authorized private
storage. The default generated directories are not release artifacts.
