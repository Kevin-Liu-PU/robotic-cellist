# VIE01 interface audit

Use model and source provenance alongside the applicable interface below.
Experiment cohorts, recording identities, cache manifests and measured scores
remain in separate private storage.

## Internal classifiers

- `SoundClassifier/classifier.py:BowingQualityClassifier.predict_detailed(audio_chunk, physical_params, string='A', window_pos=None)` returns `overall`, `tone_quality`, `attack_quality`, `release_quality`, `bow_control`, `dynamic_accuracy` from both schema-v2 checkpoints.
- Training-objective checkpoint: `quality_cnn_multilength.pt`; held-out-from-objective checkpoint: `quality_cnn_human_A1_A5.pt`. This distinction does not establish data independence or external-domain validity.
- Source architecture: 128-dimensional temporal mean/std CNN embedding, 40 z-scored scalar acoustic features, six physical features, and window position; sigmoid heads. Human annotation targets are averaged and mapped from 1-4 to 0-1. Not audio-only models: every head sees the shared physical input.
- Actual LEGACY caller: `rl/posthoc_score.py:score_stem`. It uses first-channel 44.1 kHz audio without peak normalization, executed audio-relative onsets and full executed bow durations. The classifier fits each window to 22,050 samples using a central crop for long inputs and reflection padding for short inputs. Window position is 0.5. Do not interpret a central crop's attack/release prediction as an independently measured complete-note attack/release property.
- Actual physical vector: commanded depth (m), constant zero, mean absolute measured bow speed, midpoint of commanded normalized bow positions, mean measured torque magnitude, maximum measured torque magnitude. The classifier clips checkpoint-standardized physical values to +/-2.5 standard deviations.
- Training/live scoring peak-normalizes audio, whereas the offline LEGACY path does not. Account for this normalization difference when interpreting outputs.
- Inclusion requires at least 50 ms of audio and at least one state row within the executed bow interval.
- Composite is the normalized four-head sum. At full fill: tone 0.45, attack 0.25, release 0.15, overall 0.15. With `shift=1-min(duration/0.5,1)`, move `0.30*shift` from tone to attack and `0.10*shift` from overall to release, flooring donor weights at zero. Take score is the arithmetic bow mean. This is not total RL reward and includes no invented guard information.
- Cached per-bow outputs can differ in precision and available heads. Reuse them only after checking source-input, model and implementation identity; do not label reused outputs as fresh inference.
- `score_dir`/`score_with` write into recording directories. Use authorized working copies; `reuse_cache.py` reads existing results and writes derived outputs in this analysis directory.

### External applicability gate

Public marker trajectories do not automatically provide calibrated UR wrist torques, this project's penetration-depth reference, or normalized bow coordinates. Do not pass `physical_params=None`: the wrapper silently substitutes zeros. A plausible output after that substitution is not a validated external quality measurement. Do not treat marker distance as force/torque without a justified calibration.

## Public model outputs

The adapters expose PAM and Audiobox PQ/PC. Their training-corpus coverage and calibration on any external cello dataset require separate verification; the presence of an adapter does not establish domain validity.

### PAM

- Existing adapter: `external_judges/score_takes.py:load_pam`, returning `(PAMFixed, DS)`; inference: `DS.collate([DS[index]])`, then `pam.evaluate(audio, indices)`.
- Source repository HEAD: `ead6e7f9d907268e30c73b29526f50523a34b4aa`.
- Weights: Microsoft MS-CLAP 2023, cached revision `c47d441165daa21986ead0850660917636a81775`; GPT-2 cached revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Prompts: clear/clean versus noisy/artifacts. Output is the softmax probability of the former, 0-1, not a calibrated preference probability for robot music.
- Adapter uses soundfile, first channel, requires 44.1 kHz. Native 7-second chunks; inputs shorter than 7 seconds repeat to fill 7 seconds. A long final partial chunk is replaced by the last complete 7 seconds, creating overlap. Final score is the arithmetic chunk mean. No custom level normalization.
- Existing tokenizer compatibility subclass retained without changing prompts, text length, weights or model. CPU inference, two Torch threads. Native ability to tile a short bow does not validate that repeated bow as a musical-quality stimulus.

### Audiobox Aesthetics

- Adapter package interface: `audiobox-aesthetics==0.0.4`. Existing adapter: `external_judges/score_takes.py:load_audiobox`, using `initialize_predictor()` and `.forward([{'path': WAV}])`.
- Cached model revision `9b1dd8e5df9af7216e836a98974fe3b82c56ded6`.
- Channel mean, torchaudio resampling to 16 kHz. Ten-second non-overlapping windows; final window zero-padded, with a valid-sample mask. Scores are averaged with real-duration weights.
- PQ is production quality; PC is production complexity. Both are model outputs on the reported 1-10 scale. PC is not automatically higher-is-better. They share a network, not independent judges.
- CPU inference, two Torch threads, offline Hugging Face/Transformers flags. No credentials, new downloads of model weights, external API uploads, or model training.

## Runtime requirements

Record dependency versions and model revisions for each run. The adapters use
CPU inference with bounded thread counts and offline model caches. Verify that
the required packages and weights are already available before scoring; do not
assume an environment from a previous run is present or compatible.

## Segmentation requirements

Record source identities and exact event boundaries before scoring. Keep
full-recording overviews separate from matched bow/event comparisons. Freeze
input and segmentation choices before inspecting model outputs. Report counts
of independently acquired recordings separately from derived analysis windows.
