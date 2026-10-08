# Offline evaluation and diagnostics

This directory contains offline evaluation tools. Supply recordings, sensor arrays,
annotations and experiment manifests separately from authorized private storage.
Generated scores, caches, figures and reports are experiment outputs and must not
be committed to the code-sharing repository.

| Folder | Purpose |
|---|---|
| external_judges/ | Scoring adapters for external audio models and acoustic comparisons |
| external_eval/ | Source, interface and segmentation checks for external recordings |
| tone_model_C/ | Audio-only features, grouped model selection and frozen-model scoring |
| acoustic_diagnostics_20260906/ | Envelope, transition and sustained-note acoustic diagnostics |
| reward_reweight_analysis/ | Offline reward-term reweighting tools |

Some scripts retain workflow-specific input selectors. Inspect their paths and
input contracts before use; a code checkout alone does not reproduce an experiment.
Keep source records immutable and store new outputs in a separate private run
directory. Some tools write beside their scripts or into supplied take directories;
use private working copies and follow each workflow's output-location instructions.
Configure source and data paths using the repository
[README](../README.md#inputs-and-local-configuration).

Third-party dependency not vendored: PAM (https://github.com/soham97/PAM), referenced
at commit `ead6e7f9d907268e30c73b29526f50523a34b4aa`. Its source, model weights and
applicable licenses must be supplied separately when using the PAM adapters.
No performance claim or redistribution permission follows from this reference.
