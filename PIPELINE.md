# Classifier interfaces and historical RL loop schematics

Reference for the sound classifier, scoring windows and residual-learning interfaces.

The current `PieceResidualEnv` uses **22 observation dimensions and 6 action
dimensions** (`rl/piece_env.py`, `OBS_DIM` and the action space). The optional
driver in `rl/driver_piece.py` adds **5 observation dimensions**, giving
**27 observations**, with the same 6-dimensional action space.

The 18-dimensional observation and two-component action diagram below documents
a **historical schema**. It is not the current policy input/output contract.
Use the source files and root [README.md](README.md) for current runtime settings.
Experimental recordings, ratings and evaluation outputs are separate private inputs.

---

## 0. The one-paragraph version

A frozen CNN listens to the cello and predicts what a human annotator would
rate the tone, 0–1. A reinforcement-learning policy plays a piece on the robot
and is rewarded by that score plus how accurately it hits the written dynamics.
The policy never drives the robot directly — a rule-based planner decides the
bowings, stroke lengths and timing, and the policy applies bounded speed and
depth residuals. The older two-component illustration is retained below as history.

---

## 1. Three different rates, constantly confused

Most of the disagreement in this project comes from collapsing three distinct
numbers into one. They are independent:

| Rate | What it means | Current value |
|---|---|---|
| **Analysis window** | How much audio the classifier scores at once | 500 ms |
| **Policy / setpoint rate** | How often the commanded (speed, depth) changes | once per stroke |
| **Motion control rate** | How often the robot controller gets a command | one `moveL` per stroke |

"Use 500 ms instead of 50 ms" is a statement about the **analysis window**.
"Don't re-command the robot every 50 ms" is a statement about the **setpoint
rate**. They are not the same argument and can be answered separately.

---

## 2. The classifier

`SoundClassifier/` — trained offline, then **frozen**. The RL never updates it;
no gradients flow through it. It is a reward function, not a learner.

### 2.1 Inputs

Three parallel inputs per scored window:

| Input | Shape | Source |
|---|---|---|
| Log-mel spectrogram | `(1, 64 mels, 87 frames)` | 500 ms of audio @ 44.1 kHz |
| Scalar audio features | `(40,)` | `audio_features.extract_scalar_features()` |
| Physical features | `(6,)` | robot state during the stroke |
| `window_pos` | scalar `[0,1]` | where the window sits in the stroke (0 = onset, 1 = release) |

The 40 scalar features are:

```
hnr_db_mean, hnr_db_std                      harmonic-to-noise (scratchiness)
flatness_mean, flatness_std                  tonal vs noisy
f0_mean_hz, f0_stability_cents, voiced_fraction
rms_mean, rms_std, envelope_cv,
envelope_trend, envelope_outlier_rate        loudness shape over the window
attack_time_s, attack_overshoot              onset behaviour
mfcc0..12_mean, mfcc0..12_std                timbre (13 MFCCs x mean/std = 26)
```

The 6 physical features, in `dataset.PHYSICAL_FEATURE_NAMES` order:

```
depth_or_force, force_deviation_or_zero, bow_speed,
bow_position, torque_or_lateral, torque_max_or_torque
```

Audio is **peak-normalised** before the model sees it, so absolute loudness is
deliberately invisible to the tone judgement. (Loudness is handled separately —
see §5.)

### 2.2 Architecture

`SoundClassifier/quality_classifier.py::MelQualityCNN`

```
mel (B,1,64,87) ──> conv stack ──> AdaptiveAvgPool2d((1, None))   # pool freq, KEEP time
                                        │
                                        └─> concat[ mean(t), std(t) ]  ->  (B,128)
                                                    │
scalar (B,40) ──> normalise ────────────────────────┤
physical (B,6) ─> normalise ────────────────────────┤──> fusion ──> heads ──> sigmoid
window_pos ─────────────────────────────────────────┘
```

**The trunk is length-agnostic.** It pools the time axis with mean+std, so it
produces a fixed 128-dim vector for *any* input length. `TARGET_FRAMES = 87` is
only used by `audio_to_mel_tensor()` to resize inputs for batching — it is a
convenience, not an architectural requirement. Variable-length windows are
therefore cheap on the model side (the expensive part is labels, see §2.5).

### 2.3 Outputs

- `predict(...)` -> single `overall` score in `[0, 1]`
- `predict_detailed(...)` -> all Tier-1 dimensions:
  `overall, tone_quality, attack_quality, release_quality, bow_control, dynamic_accuracy`

**What the number means.** The model regresses human ratings mapped by
`annotation_score_01 = (mean(overall) - 1) / 3`, where `overall` is a 1–4 scale.
So the score is a *predicted mean human rating*:

| score | ≈ rating |
|---|---|
| 0.20 | 1.6 / 4 |
| 0.50 | 2.5 / 4 |
| 0.80 | 3.4 / 4 |

### 2.4 Checkpoints

| file | label source |
|---|---|
| `quality_cnn.pt` | pseudo-label teacher (`pseudo_heuristic_v1`) |
| `quality_cnn_human_current.pt` | human annotations |
| `quality_cnn_human_A1_A5.pt` | human annotations |
| `quality_cnn_multilength.pt` | human annotations; training-objective scorer |

`classifier.py` defines the default checkpoint. `rl/piece_env.py::RealScorer`
prefers a human-labelled checkpoint and warns if a pseudo-label checkpoint is
loaded. Agreement with a pseudo-label teacher does not measure agreement with
human judgement. Retaining model weights does not include their recordings,
participant ratings, split assignments or evaluation reports.

### 2.5 Training

`train_classifier.py --meta <metadata.jsonl> --audio-dir <audio/>`

- Supply an authorized private metadata and audio directory.
- Each recording is sliced into 500 ms windows hopping 250 ms
  (`dataset.py: WINDOW_SEC = 0.5, TRAIN_HOP_SEC = 0.25`)
- Every window inherits its parent recording's human rating
- Annotations are imported with `label_studio_bridge.py import`, which **mutates
  `--meta` in place** — always copy the pristine `metadata.jsonl` first

**Distribution limit.** Isolated sustained strokes do not by themselves validate
short notes in musical context; assess the intended use separately (§4.4).

---

## 3. RL loops

The retained loops use different setpoint and analysis rates.

| Loop | Analysis window | Action rate | Status |
|---|---|---|---|
| `rl/piece_env.py` + `rl/piece_hardware.py` | **500 ms** | **one per stroke** | current, runs on the robot |
| `rl/env.py` + `constants.py` + `reward/` | 500 ms (buffered) | 50 ms tick | earlier single-string design |

### 3.1 Why the "50 ms" confusion happens

`constants.py` says, in as many words:

```python
TIMESTEP_SEC     = 0.05    # 50ms per RL step
STEPS_PER_STROKE = 150
```

and `rl/env.py::_record_audio()` records exactly `TIMESTEP_SEC * 44100` = 2205
samples. Read on its own, that looks like a 50 ms analysis window.

It is not. `reward/sliding_window.py` buffers **ten** 50 ms chunks and only then
scores:

```python
SlidingWindowClassifier(window_sec=0.5, hop_sec=0.05)
# "Receives 50ms chunks, scores on 500ms windows"
```

So in that loop 50 ms is the *action* rate; the analysis window was already
500 ms.

Archived models and scores are included under `assets/legacy_models/` and
`assets/legacy_scores/`; see [assets/README.md](assets/README.md) for their
locations and provenance. The archived DeepMLP architecture/loader implementation
is not included and those weights are not integrated into the current pipeline.

`constants.py`'s `TIMESTEP_SEC` is unused by the current full-song pipeline and
is worth quarantining, because it keeps causing this misreading.

---

## 4. Historical full-song RL loop schematic

### 4.1 The three layers — only the middle one learns

| Layer | Role | Trained? |
|---|---|---|
| `BowPlanner` (`play_midi_pieces.py`) | bowings, stroke lengths, timing, bow budget | **No** — rule-based |
| **SAC policy** | shades speed & depth per stroke | **Yes** |
| Classifier | judges the resulting sound | **No** — frozen |

### 4.2 Historical loop, one iteration = one bow stroke

This diagram shows the earlier 18-observation, two-component action formulation.
The current contract is 22 observations and 6 actions, or 27 observations with
the optional driver. Do not use the historical dimensions for a current policy.

```
observe (18-dim historical schema)
   note duration, written dynamic, bow direction, current u,
   bow remaining, next note's dynamic/duration/direction,
   last stroke's quality, EMA quality, progress, previous action
        │
        ▼
actor samples a ~ pi(.|s)          a = (speed residual, depth residual), tanh -> [-1,1]
        │
        ▼
apply residual to the PLANNED stroke
   mean_speed x (1 + a0 * 0.20)        +/-20% ~ +/-1.6 dB
   depth      + (a1 * 0.5 mm)          clamped to [-1.5, +2.0] mm
   length clamped to remaining bow in [U_MIN=0.04, U_MAX=0.96]
        │
        ▼
robot plays it            one moveL (or blended path for a swell)
        │
        ├──> StateLogger @100 Hz  -> measured speed, torque, bow position
        └──> microphone           -> 500 ms window, peak-normalised
        │
        ▼
reward  (see 4.3)
        │
        ▼
store (s, a, r, s') in replay buffer;  4 gradient updates (real-robot setting)
```

Episode = one pass through the piece. `reset()` re-places the bow and clears
`u`, the quality EMA and the previous action.

### 4.3 Reward

```
total = 0.50 * quality_eff     tone
      + 0.25 * r_dynamic       dynamics
      + 0.15 * r_bow           bow budget (<= 0, penalty)
      + 0.10 * r_smooth        action jerk (<= 0, penalty)
```

with

```
quality_eff = 0.5 + fill * (quality - 0.5),   fill = min(1, duration / 0.5 s)

r_dynamic   = zone reward on MEASURED dBFS      (see 5)
              or, without a loudness model, clip(1 - |err_dB| / 3, 0, 1)
              where the target speed is capped at accel_max * T / 4

r_bow       = -clip(3 * shortfall, 0, 1) - edge penalty near U_MIN/U_MAX
r_smooth    = -0.5 * mean|action - previous_action|
```

Two relevant reward-design considerations:

- **`fill`** — the classifier scores a fixed 500 ms window, but most notes are
  shorter, so most of the window is neighbouring notes. The tone term is shrunk
  toward neutral in proportion to how much of the window the note occupies. The
  raw `quality` is still logged; only the gradient is discounted.
- **achievable dynamic target** — covering length L in time T needs
  `accel >= 4L/T^2`, so the fastest possible mean speed is `accel_max * T / 4`
  (~0.08 m/s for an 0.08 s note). A target beyond that cap cannot be reached
  by a residual, so the reward must account for the planner's feasibility limit.

### 4.4 Why short notes are a problem

Not because short windows are inherently bad — because the window is **fixed at
500 ms while notes vary 18-fold** (0.08–1.5 s). For a note shorter than the
window, most of what is scored is not that note.

The analysis-window duration and note duration determine how much neighbouring
audio can enter a score. Evaluate this effect on separately authorized data.

A **note-matched window**, `min(note_duration, 500 ms)`, limits neighbouring audio.
The model can take variable lengths (§2.2), but short-note quality still requires
appropriate human labels and feature normalisers fitted to the intended inputs.

### 4.5 SAC configuration

| | mock | real robot |
|---|---|---|
| `learning_starts` | 1000 | **200** |
| `gradient_steps` | 1 | **4** |
| default save chunk | 5000 training steps | **250 training steps** |

Saving may extend to the current episode boundary. Numbered model checkpoints
from `train_piece_logged.py --ckpt-every` use an episode interval instead.

Shared: `lr 3e-4`, `gamma 0.99`, `buffer 100k`, `batch 256`, `tau 0.005`,
`train_freq 1`, `ent_coef "auto"`.

Off-policy replay reuses recorded transitions for multiple gradient updates.
Replay buffers are experiment data and must remain in private storage even
when trained model weights are retained.

---

## 5. Closed-loop dynamics

`rl/loudness.py` + `SoundClassifier/checkpoints/loudness_model.json`

Originally the dynamics reward compared measured **bow speed** to a target
speed — speed as a proxy for loudness. Now it measures loudness directly.

The retained model uses these calibration coefficients:

```
dBFS = 1.131 * 20log10(bow_speed) + 0.421 * depth_mm - 5.29
```

The model uses speed and depth; it omits bow position. Its coefficients do not
establish validity for a different position distribution, string, microphone or
bow configuration. Recalibrate and validate when those conditions change.

### Dynamic zones

Four equal 2.5 dB slices of the reachable ~10 dB span:

| zone | dBFS | speed at centre |
|---|---|---|
| p | -28.7 … -26.2 | 0.102 m/s |
| mp | -26.2 … -23.7 | 0.132 m/s |
| mf | -23.7 … -21.2 | 0.170 m/s |
| f | -21.2 … -18.7 | 0.220 m/s |

A stroke inside its zone scores 1.0; outside, reward falls linearly to 0 over
3 dB. The zone interior is flat rather than peaked at its centre.

`measure_dbfs()` must be given the **raw** capture: the classifier window is
peak-normalised, and normalising destroys exactly the level being measured.

**Caveat: dBFS is absolute**, so the zones are tied to this microphone, gain and
placement. Change any of them and the zones must be refitted or they are
silently wrong.

### Feed-forward calibration

`--calibrated-dynamics` inverts the model to target the centre of each written
dynamic zone. Depth remains tied to the original written volume.

This option changes planned bow travel and can move the trajectory toward
`U_MIN`. Check drift, bow margins and the resulting path before enabling it;
calibration does not replace the existing hardware safety procedures.

---

## 6. Experiment storage

Keep recordings, sensor arrays, participant ratings, run logs, replay buffers
and derived evaluation reports in separate private experiment storage. Retaining
model weights does not authorize sharing their inputs or evaluation records.

---

## 7. Validation and safety boundaries

1. **Note-matched windows.** Validate short-window scores and feature
   normalization on separately authorized recordings before relying on them.
2. **Intra-stroke control.** Streaming requires sensing and inference that run
   asynchronously from the motion-control loop. Account for analysis and
   inference latency; blocking inference can stall the control loop.
3. **Streaming safety.** `servoL` bypasses `safe_moveL` fault checking. Equivalent
   guards are needed: clamp `u` at every tick, check faults and define safe
   behaviour for a stalled or disconnected control loop.
4. **Repertoire.** Score dynamics and note durations affect achievable actions
   and judge reliability. Validate each intended repertoire and setup.
