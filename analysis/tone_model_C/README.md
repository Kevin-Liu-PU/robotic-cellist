# Audio-only tone model and classifier checks

These tools compare classifier heads with human labels and fit an audio-only tone
model for scoring separately supplied recordings.

- **B**: compare classifier heads with human ratings (`check_b.py`).
- **C**: select an audio-only model using labeled recordings, freeze it, then score
  separately supplied policy and control recordings (`score_takes.py`).

Split by original recording session so a session cannot appear on both sides.
The outer GroupKFold reports Spearman/MAE; an inner GroupKFold on each outer
training partition selects the model and hyperparameters. Fit imputation and
standardization only on training rows. Then repeat inner selection on all labeled
training data, refit the selected model and freeze it. Default fold counts are
five outer and four inner, subject to the available session count. Policy takes
must not be read during model training or selection.

## Private inputs

- Metadata records with `annotations` containing the required rating fields,
  or raw metadata plus authorized Label Studio exports. Confirm export field
  mappings before using the best-effort importer.
- An `audio/` directory containing files named by `audio_file` (44.1 kHz WAV).
- Original checkpoint train/validation assignments when comparing held-out
  performance. `check_b.py --split-file split.json` accepts
  `{"val_group_ids": [...]}` or `{"val_stroke_ids": [...]}`. Without that file,
  it reconstructs a split with `dataset.group_train_val_split` (seed 0, 20%)
  and labels it accordingly. The used IDs are saved to `split_B_used.json`;
  all-row statistics remain descriptive.

Recordings, individual ratings, split assignments and generated outputs remain
private. They are not bundled with the code.

## Example workflow

Create a private run directory and use it as the working directory. Invoke these
scripts by absolute path with an existing compatible Python environment; replace
all placeholder paths. Do not run expensive scoring alongside hardware recording.

```sh
cd "/path/to/private_run"
python "/path/to/checkout/analysis/tone_model_C/build_manifest.py" --meta /path/to/private_dataset/metadata.jsonl --audio-dir /path/to/private_dataset/audio --out /path/to/private_run/manifest.csv
python "/path/to/checkout/analysis/tone_model_C/check_b.py" --manifest /path/to/private_run/manifest.csv --meta /path/to/private_dataset/metadata.jsonl --split-file /path/to/private_run/split.json
python "/path/to/checkout/analysis/tone_model_C/train_c.py" --manifest /path/to/private_run/manifest.csv --meta /path/to/private_dataset/metadata.jsonl --target tone_quality --cache /path/to/private_run/feature_cache --out /path/to/private_run/model_C.joblib
python "/path/to/checkout/analysis/tone_model_C/score_takes.py" --model /path/to/private_run/model_C.joblib --d-tsv /path/to/private_run/takes.tsv
```

Configure source and recording roots as described in the repository
[README](../../README.md#inputs-and-local-configuration). `check_b.py` writes
split and prediction files to the working directory. `train_c.py --cache` controls
its feature-cache location; its model metadata and out-of-fold predictions are
written beside `--out`. Scoring can write into supplied take directories, so use
authorized private working copies and keep those outputs outside the code tree.

## Files

- `features_c.py`: audio-only features per segment (0.5 s windows, 0.25 s hop,
  peak-normalized; scalar acoustic features and log-mel band summaries,
  aggregated into the implementation's 210-dimensional feature vector).
  Notes are sliced using the same logic as `rl/posthoc_score.py`.
- `build_manifest.py`: maps labels from 1-4 to [0,1], averages annotators,
  records session/condition/group IDs and computes inter-annotator agreement.
- `train_c.py`: ridge (four alphas), gradient boosting (two depths) and random
  forest candidates with imputation and scaling; nested GroupKFold by session;
  pooled outer Spearman/MAE and per-fold selections; frozen final model with
  associated model-selection metadata.
- `score_takes.py`: frozen-model scoring of takes and paired control/ablation
  comparisons.
- `check_b.py`: tone, overall, attack and release heads compared with human
  labels on all rows and on the supplied or reconstructed validation split.
