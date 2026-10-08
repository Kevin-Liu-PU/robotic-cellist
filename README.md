# Multimodal Robot Learning with Action-Space Expansion and Cross-String Adaptation

This code snapshot implements score-based planning and residual reinforcement
learning for robotic cello bowing. The research direction is to combine audio,
force/torque, vision and robot state in policies, expand the action space, and
support adaptation across strings.

## Implemented scope

The baseline uses residual SAC for planned bow strokes. `PieceResidualEnv` has
22 observations and 6 residual actions; the optional driver adds five scalar
loudness and torque observations, giving 27 observations with the same action
space. Audio classification, loudness and physical measurements contribute to
scoring and reward.

An integrated audio/force/vision/state policy, a validated expanded policy action
space and learned cross-string adaptation remain research work. Multimodal reward
inputs do not establish a multimodal policy. Existing control code and calibration
files do not establish hardware readiness.

## Setup

Use 64-bit Python 3.11 or 3.12; the project declares `>=3.11,<3.13`.
From the repository root:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[dev]"
```

The optional `hardware` and `labeling` extras add robot and annotation dependencies.
The platform setup scripts also install dependencies and run tests; inspect their
actions before using them.

## Entry points

| Path | Purpose |
| --- | --- |
| `rl/train_piece.py` | Whole-piece residual SAC training; supports `--mock`. |
| `rl/play_piece.py` | Score playback through the environment. |
| `rl/perform.py` | Policy performance and hardware-session integration. |
| `rl/piece_env.py` | Observations, residual actions and reward wiring. |
| `reward/` | Audio-window scoring and reward components. |
| `SoundClassifier/` | Classifier training, inference and collection tools. |
| `BaselineControls/` | Score planning, execution and motion primitives. |

See [TRAINING.md](TRAINING.md) for training and resume options,
[PIPELINE.md](PIPELINE.md) for interfaces and historical schematics, and
[RL_METHOD.md](RL_METHOD.md) for historical methods notes.

## Inputs and local configuration

The snapshot includes eight model/scaler files and 29 MIDI/MXL scores; see
[Models and scores](assets/README.md) for their locations and provenance.
Raw datasets, ratings, run outputs, replay buffers and trained RL policy ZIP
archives are not included. Included classifier weights are not RL policies.

Commands require an available score and, where applicable, a compatible
classifier checkpoint and calibration files. Check the actual files and model
compatibility before running; example paths do not grant redistribution
permission. The default mock executor and scorer do not need
classifier weights, but calibrated dynamics requires the loudness model.

Analysis workflows require separately supplied recordings, logs and manifests.
Keep sensor traces, ratings, recordings, reports and generated checkpoints in
authorized private storage. Historical evaluation recipes retain fixed experiment
identifiers and pair definitions; configuring paths does not define a new
evaluation protocol.

Offline analysis defaults to this checkout. Relative environment paths resolve
from the calling working directory; missing required inputs raise configuration
errors.

| Setting | Use |
| --- | --- |
| `CELLO_REPO_ROOT` | Optional source checkout override. |
| `CELLO_DATA_ROOT` | Private logs and `perform_*` takes; defaults to the selected checkout's `rl/checkpoints_piece`. |
| `CELLO_REFERENCE_ROOT` | Explicit reference code/model root for reward reweighting; no implicit reference model. |
| `CELLO_PAM_ROOT` | Local PAM checkout, required when running that judge. |
| `CELLO_PAIR_MANIFEST` | Private pair manifest for external-evaluation preparation. |
| `CELLO_CACHE_SCORES` / `CELLO_CACHE_MANIFEST` | Private cached scores and their manifest. |
| `CELLO_PAPER_ROOT` | Optional manuscript checkout whose Git head is recorded only when configured. |
| `CELLO_HF_HUB_CACHE` / `CELLO_AUDIOBOX_SOURCE` | Existing model cache and local AudioBox source for provenance inventory; these settings do not download assets. |

`python analysis/external_judges/score_takes.py --help` requires no model libraries
or private logs. Scoring and feature extraction require their dependencies and
inputs. Logged take paths are relocated under `CELLO_DATA_ROOT` by `perform_*` ID.
Moving locked evaluation inputs requires a new reviewed manifest. The combined
external probe requires all PAM source pins before either judge runs; prepare a
new lock with `CELLO_PAM_ROOT` set if these pins are absent.

## Hardware boundary

Review the controller configuration, taught poses, force limits, calibration,
recovery procedure and emergency stop before hardware commands. Offline checks
and mock execution do not validate contact control or physical safety. Required
gate files, review hashes and session inputs must not be fabricated or bypassed.

`BaselineControls/robot_runner_simple.py` is a legacy standalone script with
import-time hardware connection and motion side effects. It is not safe to
import or run without local hardware review. Its four `/path/to/local/` input
paths require explicit configuration; the required RTDE client, bowing input
and URScript template must be supplied and reviewed locally. It is not a
library module and was not executed during validation.

Historical `rl/chain*.sh` and `rl/bracket*.sh` recipes derive the checkout from
their location and accept `CELLO_PYTHON`. Except for the G1 wrapper, they require an existing absolute
`CELLO_CONTROL_DIR` and remain disabled unless
`CELLO_ENABLE_HISTORICAL_RECIPES=1` is explicitly set after reviewing the recipe
and supplying its inputs. This switch does not replace downstream checks.

The G1 wrapper forwards arguments to `rl/g1_session.py` and defaults to a dry run.
Controls use the session directory unless `--control-dir` is supplied. Execution
of this fixed historical protocol requires `--execute`, `--attended`, runtime
pins, `rl/gate.json`, preregistration,
the review lock and historical budget acknowledgement. Its review JSON must
provide `full_checkpoints` keys `"0"`, `"1"` and `"2"`, each with a selected
`path` and `sha256`. Review paths, including `prereg_path`, are repository-relative
unless absolute. Checkpoint hashes are rechecked before execution.

## Validation and limitations

Limited checks used Python 3.12.10:

| Inspected test scope | Passed | Skipped | Boundary-blocked errors |
| --- | ---: | ---: | ---: |
| Five helper suites | 59 | 2 | 2 |
| `test_perform_audit.py` alone, also included above | 15 | 2 | 2 |

Static parsing passed for 106 Python files and 6 JSON files.

The two snapshot-test errors occur when `write_snapshot` attempts to import
`rl.piece_env`. The validation guard blocks environment, hardware and model
imports, network access and subprocess execution. The real environment was not
replaced with a stub. The two skips require unavailable private C1 fixtures.
No assertion failures were observed; the blocked cases remain unvalidated.

The full pytest suite was not run: pytest, Gymnasium, soundfile, sounddevice,
mido, PyTorch, Stable-Baselines3 and librosa were absent. Python 3.11 was
unavailable.

Robot drivers, hardware motion, mock or hardware training, model inference and full
environment integration were not exercised. Historical zsh recipes received static inspection;
their runtime behaviour was not validated.
The legacy standalone runner's connection and motion behaviour is also unvalidated.

## Release boundaries

Code availability does not establish redistribution rights for models, scores or
third-party material. No repository-wide open-source license is declared.
Preserve scientific citations, embedded attribution and third-party notices.
This code snapshot does not establish anonymity.
