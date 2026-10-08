# Residual RL for expressive cello bowing — historical methods notes

These notes retain the earlier method formulation and parameter examples.
The current `PieceResidualEnv` uses **22 observation dimensions and 6 action
dimensions** (`rl/piece_env.py`, `OBS_DIM` and the action space). The optional
driver appends **5 observation dimensions**, giving **27 observations**
(`rl/driver_piece.py`); it does not add action dimensions.

The 18-dimensional table below is a **historical schema**, not the current
observation contract. See the root [README.md](README.md), current source and
[PIPELINE.md](PIPELINE.md) for implementation context. Keep experimental results
separate from this code documentation.

Primary sources: `rl/piece_env.py`, `rl/piece_hardware.py`, `rl/loudness.py`,
`rl/train_piece.py`, `BaselineControls/play_midi_pieces.py`.

---

## 1. Problem formulation

A musical score is executed on a UR5e bowing the open A string of a cello. A
rule-based planner converts the score into a sequence of bow strokes; a learned
policy applies bounded corrections to each stroke; a frozen neural sound-quality
model supplies the reward.

The task is a finite-horizon MDP, one **episode per piece** and one **timestep
per bow stroke**:

- **Historical state schema** `s_t ∈ ℝ^18` — score context, bow state, recent quality (§3); the current base environment uses 22 observations
- **Action** `a_t ∈ [-1,1]^6` — a per-segment bowing envelope (§4)
- **Transition** — the robot physically executes the stroke; the next state
  follows from the score and the achieved bow position (§5)
- **Reward** `r_t ∈ ℝ` — weighted sum of tone, dynamic accuracy, bow-budget
  and smoothness terms (§6)
- **Horizon** — the number of strokes the planner produces (14 for
  `string_crossings`, 25 for `t1`, 182 for `challengepiece`)

The formulation is **residual**: the policy never generates a trajectory. The
planner fixes bow directions, note onsets and note durations; the policy shades
speed and depth within bounded limits. These bounds do not by themselves establish
physical safety or musical quality; hardware use requires the configured controller
limits and an attended lab session.

### 1.1 Three layers, one learned

| Layer | Responsibility | Learned? |
|---|---|---|
| `BowPlanner` | bowings, stroke lengths, bow budget, timing | No — rule-based |
| **SAC policy** | per-segment speed and depth residuals | **Yes** |
| `BowingQualityClassifier` | tone judgement (reward) | No — frozen |

No gradient flows through the classifier; SAC treats it as a black-box reward.

---

## 2. Physical parameterisation

Bow position is a scalar `u ∈ [0,1]`, the fraction along the taught frog→tip
line. Poses are

```
pose(u, d) = apply_depth( lerp(FROG, TIP, u), CFG, d )
```

with translation a straight lerp and rotation a lerp **after canonicalising the
tip's axis-angle vector into the frog's hemisphere**. The taught frog and tip
rotation vectors sit in opposite hemispheres of the axis-angle double cover
(|r| ≈ 3.136 and 3.097, both just under π); a naive lerp passes through r ≈ 0
and produces a 178.5° wrist flip at the stroke midpoint. Canonicalised, the
lerp deviates < 0.0021° from the geodesic.

Configuration and model parameters (verify runtime settings and calibrate for
the intended hardware before use):

| Symbol | Value | Meaning |
|---|---|---|
| `BOW_LENGTH` | 0.5249 m | frog→tip distance |
| `U_MIN, U_MAX` | 0.04, 0.96 | usable bow, hard margins |
| `SPEED_MIN, SPEED_MAX` | 0.09, 0.25 m/s | calibrated bow-speed grid |
| `MOVE_ACCEL` | 1.2 m/s² | pendant-verified nominal |
| `ACCEL_MAX` | 4.0 m/s² | planner ceiling |
| depth envelope | −1.5 … +2.0 mm | signed, − is lighter |

The retained loudness model maps bow speed and depth to acoustic level.
Its coefficients and dynamic zones depend on the calibrated hardware setup.

### 2.1 Feasibility constraint

A stroke starting and ending at rest follows at best a triangular velocity
profile, so covering length `L` in time `T` requires

```
a_required = 4L / T²
```

and the fastest achievable **mean** speed for a note of duration `T` is

```
v_max(T) = a_max · T / 4
```

For `a_max = 4.0`, a 0.08 s note tops out near 0.08 m/s regardless of the
dynamic marked. `solve_stroke()` shortens the stroke rather than missing the
beat, and §6.2 grades against `v_max(T)` rather than the written dynamic — a
distinction worth ~8 dB of spurious penalty on short notes.

---

## 3. Historical state schema (18 dimensions)

The table records the earlier 18-dimensional formulation. It is retained for
method history; use the current 22-dimensional base schema (27 with the optional
driver) when constructing observations or loading a compatible policy.
Values in this historical table were normalised to approximately [−1, 1]:

| idx | quantity | normalisation |
|---|---|---|
| 0 | planned note duration | ÷ 2 s |
| 1 | planned mean speed | ÷ `SPEED_MAX` |
| 2 | planned depth | ÷ `DEPTH_HI` |
| 3 | bow direction | +1 down, −1 up |
| 4 | current bow position `u` | already [0,1] |
| 5 | available bow in this direction | fraction of usable range |
| 6 | written dynamic (volume target) | [0,1] |
| 7 | volume at note end (swell) | [0,1] |
| 8 | stroke begins with a retake | binary |
| 9 | stroke is part of a split note | binary |
| 10 | next stroke's volume | [0,1] |
| 11 | next stroke's duration | ÷ 2 s |
| 12 | next stroke's direction | +1/−1/0 |
| 13 | previous stroke's tone score | [0,1] |
| 14 | EMA of tone this episode | α = 0.7 |
| 15 | progress through the piece | [0,1] |
| 16–17 | previous action (mean speed, mean depth) | [−1,1] |

Indices 10–12 provide one-step lookahead; 13–15 provide history. Both are
needed because bow budget and phrasing are sequential: bow consumed now
constrains what is playable several notes later.

---

## 4. Action space — a per-note bowing envelope

`a_t ∈ [-1,1]^6`, interpreted as three segments × (speed, depth):

```
a = [ σ₁, σ₂, σ₃ , δ₁, δ₂, δ₃ ]
```

### 4.1 Rendering

For a note of duration `T ≥ 0.40 s` (`ENVELOPE_MIN_DURATION`):

1. **Stroke length** is set by the *mean* speed residual, because bow budget is
   a property of the whole note:
   ```
   scale = 1 + mean(σ) · 0.35            (SPEED_RESIDUAL_FRAC)
   L_desired = L_planned · scale
   L = clip(L_desired, 1e-4, bow available in this direction)
   ```
2. **Time is split equally** across the three segments, `T/3` each.
3. **Bow is allocated in proportion to segment speed weights**:
   ```
   w_i = clip(1 + σ_i · 0.35, 0.25, 4.0)
   span_i = (u_end − u_start) · w_i / Σw
   ```
   Equal time with unequal distance is what makes `σ_i` read as "speed during
   this part of the note".
4. **Depth is set per segment outright**:
   ```
   d_i = clip( d_base + δ_i · 0.001 m , −0.0015 , +0.002 )
   ```
5. Each segment is solved with `solve_stroke(span_i, T/3)` for its commanded
   speed and acceleration, and the segments are emitted as one blended
   `moveL` path.

> **Renderer requirement.** Solving each envelope segment independently as a
> rest-to-rest trajectory can produce unintended internal stops. A continuous
> envelope requires non-zero interior boundary velocities; only the first and
> last boundaries should be at rest. Check this independently of reward scores.

**Total length and total duration are preserved exactly.** The envelope decides
only how travel is distributed in time; it cannot acquire extra bow or alter
the rhythm, which remain planner decisions.

For `T < 0.40 s` the note is executed as a single segment using `σ₁, δ₁`. Three
segments of a 0.25 s note are ~12 mm of bow each, and the `moveL` blend radius
(≤ 5 mm) would smear them together. Dimension 0 therefore means "the attack"
consistently: for a short note the attack *is* the note.

### 4.2 Feed-forward and streaming control

`moveL` executes a pre-specified path; the envelope is selected before the
stroke sounds. Reacting during a stroke requires streaming control, such as
`BaselineControls/servo_player.py`, plus sensing and inference that complete
within the stroke. Account for analysis-window, inference and command latency
before interpreting the policy as feedback within a note.

---

## 5. Transition — execution and measurement

Per stroke (`rl/piece_hardware.py`):

1. Sleep the notated gap, or perform a planned retake (lift → traverse → set
   down), keeping the bowing intact.
2. Execute the blended `moveL` path through `safe_moveL`, which raises on a
   failed move or a stopped control script.
3. `StateLogger` samples RTDE at 100 Hz; `get_summary(t_start, t_end)` returns
   the achieved mean bow speed and torque statistics over exactly that window.
4. Slice the analysis window from a continuously running microphone stream
   (§5.1), peak-normalise it, and measure its raw RMS level in dBFS **before**
   normalisation.
5. Score with the frozen classifier and compute the reward.

### 5.1 Note-matched analysis windows

The classifier uses 0.5 s windows, while score notes can be shorter. A fixed
0.5 s window centred on an 80 ms note contains 80 ms of that note and 420 ms
of its neighbours, so it can score the surrounding passage.

The window is therefore cut to the note:

```
if T > WINDOW + 2·PRE_ROLL:      # long note
    centre on the steady middle, past the onset transient; width = 0.5 s
else:                            # short note
    window = [t_start, t_start + max(T, 0.08 s)]
```

`MIN_ANALYSIS_SEC = 0.08` is a signal-processing floor: at 50 ms an open A
yields only ~11 pitch periods, insufficient for F0 stability or HNR. A note
below the floor is extended into its own release, never into the next note.

Short-window quality requires separate validation for the intended repertoire.

The trunk pools time with mean+std, so padded frames contribute a constant
while the real frames drive the statistics.

---

## 6. Reward function

```
r =  0.50 · quality_eff        tone
   + 0.25 · r_dynamic          dynamic accuracy
   + 0.15 · r_bow              bow budget          (≤ 0)
   + 0.10 · r_smooth           inter-stroke jerk   (≤ 0)
   − 0.15 · r_defect           objective defects   (≥ 0, subtracted)
   + 0.05 · r_onset            onset acceleration  (≤ 0)
   + 0.05 · r_envelope         intra-stroke jerk   (≤ 0)
```

### 6.1 Tone

The classifier exposes six human-rated heads. The reward is a length-dependent
weighted mean, because **a short note is nearly all attack and release** — there
is little sustain to judge:

```
base weights:  tone_quality 0.45, attack 0.25, release 0.15, overall 0.15
fill = min(1, T / 0.5)
if fill < 1:  shift 0.30·(1−fill) from tone_quality to attack
              shift 0.10·(1−fill) from overall     to release
```

Effective weights:

| note | tone_quality | attack | release | overall |
|---|---|---|---|---|
| ≥ 0.50 s | 0.45 | 0.25 | 0.15 | 0.15 |
| 0.25 s | 0.30 | 0.40 | 0.20 | 0.10 |
| 0.07 s | 0.19 | **0.51** | 0.24 | 0.06 |

`attack_quality` and `release_quality` are human judgements about events short
enough to fit inside a brief note, which makes them the only sound estimator
available at that length — no additional annotation is required.

A light length discount remains:

```
quality_eff = 0.5 + (0.85 + 0.15·fill) · (tone − 0.5)
```

The floor in this formulation is 0.85. The discount represents a design choice
for short-window reliability; it does not establish validation on short notes.

### 6.2 Dynamic accuracy

Closed loop on **measured acoustic level**, not on commanded bow speed. The
level model uses retained calibration coefficients:

```
dBFS = 1.131 · 20log₁₀(v) + 0.421 · depth_mm − 5.295
```

**Bow position is excluded from this model.** Predictions for partial strokes
or a changed bow configuration require independent calibration; the retained
coefficients do not establish validity outside the calibration conditions.

Written dynamics are mapped to four **zones**, equal 2.5 dB slices of the
~10 dB reachable span:

| zone | dBFS (model frame) | speed at centre |
|---|---|---|
| p | −28.7 … −26.2 | 0.102 m/s |
| mp | −26.2 … −23.7 | 0.132 m/s |
| mf | −23.7 … −21.2 | 0.170 m/s |
| f | −21.2 … −18.7 | 0.220 m/s |

```
r_dynamic = 1                                  inside the zone
          = max(0, 1 − distance / 3 dB)        outside
```

The reward is flat inside each target zone and falls with distance outside it.

**Gain calibration.** dBFS is absolute. `gain_offset_db` corrects measurements
into the model frame and must be re-measured whenever the microphone, gain,
placement or audio path changes.

Where no level model or microphone is available the term falls back to a
speed proxy, `r_dynamic = clip(1 − |err_dB| / 3, 0, 1)` with the target capped
at `v_max(T)` from §2.1.

### 6.3 Bow budget, smoothness, defects, onset

```
r_bow      = −clip(3 · shortfall, 0, 1) − 0.5·(1 − edge/0.03) if edge < 0.03
r_smooth   = −0.5 · mean|a_t − a_{t−1}|
r_envelope = −0.5 · ( mean|Δσ| + mean|Δδ| )        envelope only
r_onset    = −clip( (accel − 1.2) / (4.0 − 1.2), 0, 1 )
```

`shortfall` is the fraction of the desired stroke length that had to be cut to
stay inside `[U_MIN, U_MAX]`; `edge` is the distance from the final bow position
to the nearer hard limit. `r_onset` prices attack harshness explicitly, since
`solve_stroke` otherwise raises acceleration freely to fit short notes and
higher acceleration *is* a harder attack.

**Objective defect penalty.** Four hand-engineered features, monotonic and
computable at any window length, scored linearly between a good and a bad value
and averaged:

| feature | direction | good | bad | defect |
|---|---|---|---|---|
| `hnr_db_mean` | higher better | 12.0 | 3.0 | scratchy |
| `voiced_fraction` | higher better | 0.98 | 0.70 | string not speaking |
| `attack_overshoot` | lower better | 0.85 | 1.40 | harsh attack |
| `f0_stability_cents` | lower better | 1.0 | 20.0 | pitch wobble |

Spectral flatness and `envelope_cv` are **deliberately excluded** although the
codebase's heuristic scorer uses them: some noise is intrinsic to cello timbre
(driving flatness to zero yields a sterile tone), and a swell has high envelope
variation by design. Because these features need no labels and no minimum
window, they remain valid exactly where the CNN is weakest.

---

## 7. Reward model (frozen classifier)

`SoundClassifier/quality_classifier.py`, `MelQualityCNN`.

**Inputs** — log-mel spectrogram (64 mels × T frames, 0.5 s → 87 frames);
40 hand-engineered scalar features (HNR, spectral flatness, F0 stability,
voiced fraction, RMS envelope statistics, attack time and overshoot, 13 MFCC
means and standard deviations); a 6-dim physical vector (commanded depth, force
deviation, measured bow speed, bow position, mean and peak torque); and
`window_pos ∈ [0,1]`, the window's position within the stroke.

**Trunk** — convolutional stack → `AdaptiveAvgPool2d((1, None))` (pools
frequency fully, retains time) → concatenation of mean and standard deviation
over time → 128-dim. **This is length-agnostic**: any frame count produces the
same vector across frame counts, permitting note-matched windows.

**Outputs** — six sigmoid heads over `TIER1_FIELDS`: `overall`, `tone_quality`,
`attack_quality`, `release_quality`, `bow_control`, `dynamic_accuracy`.

**Targets** — mean annotator rating on a 1–4 scale mapped to [0,1] by
`(mean − 1)/3`, so a score is a *predicted mean human rating*.

**Training representation** — recordings are cut into 0.5 s windows hopping
0.25 s. Source recordings, participant ratings and split assignments remain
private inputs.
`attack_quality` is supervised only on early windows and `release_quality` only
on late ones (`position_weight`). Loss is MSE on `overall` + 0.5 × in-batch
pairwise ranking hinge + 0.3 × position-weighted auxiliary MSE.



**Checkpoint provenance matters.** `classifier.py`'s `DEFAULT_CHECKPOINT`
(`quality_cnn.pt`) uses `pseudo_heuristic_v1` **pseudo-labels**. Validation against
that teacher measures agreement with a deterministic heuristic, not agreement
with human judgement. Optimising a policy against it would maximise a
hand-written formula. `RealScorer` therefore prefers human-labelled checkpoints
and warns on pseudo-label ones.

---

## 8. Algorithm and hyperparameters

Soft Actor-Critic (Stable-Baselines3), `MlpPolicy` (two 256-unit hidden layers),
twin Q-critics, automatic entropy tuning.

| | mock | **real robot** |
|---|---|---|
| learning rate | 3e-4 | 3e-4 |
| discount γ | 0.99 | 0.99 |
| replay buffer | 100,000 | 100,000 |
| batch size | 256 | 256 |
| τ (soft update) | 0.005 | 0.005 |
| `train_freq` | 1 | 1 |
| `learning_starts` | 1000 | **200** |
| `gradient_steps` | 1 | **4** |
| default save chunk | 5000 training steps | **250 training steps** |

Saving may extend to the current episode boundary. Numbered model checkpoints
from `train_piece_logged.py --ckpt-every` use an episode interval instead.

Off-policy replay reuses transitions for multiple gradient updates. Keep replay
buffers in private experiment storage even when model weights are retained.

γ = 0.99 over a 14–182 step episode gives a discount horizon spanning the whole
piece, which the critic needs because bow consumed now constrains later strokes.

Latest-model saves also save the replay buffer, which contains collected
transitions. An abrupt interruption can lose transitions since the last completed
save. A changed reward invalidates stored transitions, so reward modifications
require a fresh run rather than `--resume`.

---

## 9. Evaluation records

Keep measured outcomes, recordings, participant ratings, run logs, replay buffers
and derived reports in separate private experiment storage. They are outside the
code-sharing boundary. Any distribution requires a completed data-exclusion check.

---

## 10. Limitations

1. **String and configuration coverage.** Validate the policy and classifier
   separately for each intended string and bow configuration.
2. **Reward and evaluation independence.** A score from the training reward
   model is not independent evidence of audible improvement. Use separate
   evaluation and listening evidence for performance claims.
3. **Short-note supervision.** Whole-stroke labels and attack/release heads are
   proxies for short-note quality; window matching alone does not validate them.
4. **Musical context.** Isolated-stroke inputs and notes embedded in passages
   can have different distributions.
5. **Loudness model scope.** The retained model omits bow position. Its absolute
   dBFS zones depend on the microphone, gain and hardware setup.
6. **Renderer continuity.** Inspect velocity continuity at envelope boundaries.
   Reward terms do not by themselves guarantee that the renderer avoids
   unintended stops or amplitude dips within a note.
7. **Mock validation.** An authored mock reward tests learnability under that
   model; it does not establish hardware performance or safety.

---

## 11. Running the supplied implementation

These commands illustrate execution; they do not reproduce an experiment without
its original inputs and configuration. Run from the repository root, replace
`CHECKPOINT_PATH` with an authorized trained RL policy, and use private storage
for new run artifacts. Policy ZIP archives and replay buffers are not supplied.
All commands below use real hardware and require the setup and safety review
described in [TRAINING.md](TRAINING.md).

```bash
# baseline (zero residual)
python rl/play_piece.py MIDI-Files/string_crossings.mid --real --episodes 3

# train
python rl/train_piece.py MIDI-Files/string_crossings.mid --real \
    --timesteps 1500 --save-stroke-audio --save-dir /path/to/private_run/checkpoints

# evaluate a policy
python rl/play_piece.py MIDI-Files/string_crossings.mid --real --episodes 3 \
    --model CHECKPOINT_PATH
```

Verify the configured robot endpoint and check the audio channel at startup.
An empty input can return a valid stream containing only its noise floor. Verify
that the selected input carries the intended microphone.

The hardware path requires a compatible `ur_rtde` installation. Binary-wheel and
source-build requirements depend on the platform and package version; verify
them for the selected environment rather than assuming a fixed Boost version.
