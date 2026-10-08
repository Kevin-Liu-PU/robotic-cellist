# Models and scores

The release includes eight model/scaler files and 29 MIDI/MXL score files.
Classifier weights are under `SoundClassifier/checkpoints/`; archived assets
are under `assets/legacy_models/` and `assets/legacy_scores/`. Other scores are
in `MIDI-Files/` and the repository root. Duplicate files remain separate assets.

## Model inventory

Paths below are relative to the repository root.

| File | Role |
| --- | --- |
| `SoundClassifier/checkpoints/quality_cnn.pt` | CNN checkpoint at the legacy default path. |
| `SoundClassifier/checkpoints/quality_cnn_human_current.pt` | Alternative CNN checkpoint. |
| `SoundClassifier/checkpoints/quality_cnn_human_A1_A5.pt` | Human-annotated CNN checkpoint used by held-out-objective scoring tools. |
| `SoundClassifier/checkpoints/quality_cnn_multilength.pt` | CNN checkpoint used by the training-objective scorer. |
| `assets/legacy_models/quality_cnn.pt` | Archived CNN checkpoint. |
| `assets/legacy_models/quality_cnn_human_current.pt` | Archived human-annotated CNN checkpoint. |
| `assets/legacy_models/deep_mlp.pt` | Archived DeepMLP weights. |
| `assets/legacy_models/scaler.pkl` | Fitted scaler accompanying the archived DeepMLP. |

`assets/legacy_models/selected_features.json` describes the six selected features
for the archived DeepMLP input. Its former wrapper also appended pitch as a
seventh input. The source wrappers described the DeepMLP and scaler as supplied
by the providing team. Their architecture/loader implementation is not included;
these assets are not integrated into the current pipeline.

The five scores under `assets/legacy_scores/` are `happy-farmer.mxl`,
`maysong.mxl`, `paganini_variations.mxl`, `t1.mxl` and `twinkle-twinkle.mxl`.
Score notation, embedded attribution and serialized model content are unchanged.

## Use and provenance

A checkpoint's label source and role do not establish independent evaluation,
external-domain validity or hardware readiness. Confirm the expected feature
schema and loader before using a model.

Raw recordings, sensor traces, individual ratings, experiment reports, replay
buffers and trained RL policy ZIP archives are not included. Supply authorized
private inputs separately for training, evaluation and policy playback.

No new license is granted by including these files. Preserve existing scientific
attribution, embedded score credits and third-party notices; provenance alone
does not establish redistribution permission.
