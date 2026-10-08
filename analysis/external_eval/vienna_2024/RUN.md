# External-recording evaluation workflow

These scripts require separately supplied recordings, model weights and private
experiment manifests. A code checkout alone does not reproduce a prior comparison.
Inspect the external README, audio and motion-capture schema before defining
segments or scoring adapters.

Do not run CPU scoring alongside hardware recording. Check relevant processes
first and follow the lab's hardware safety procedures independently.

## Environment and input preparation

Use an existing compatible Python environment. The public-model adapter enables
offline Hugging Face/Transformers mode and caps Torch to two CPU threads. Supply
authorized cached weights and record runtime versions and model/source hashes.
See [INTERFACES.md](INTERFACES.md) for input semantics and applicability limits.

Configure the input paths listed in the repository
[README](../../../README.md#inputs-and-local-configuration). The scripts retain
workflow-specific input selectors and write generated files beside the scripts.
Changing the current working directory does not relocate those outputs. Use a
private working copy of the checkout for each analysis run, and run the commands
below from its `analysis/external_eval/vienna_2024/` directory. Preserve original
records and earlier runs; do not silently replace input hashes or results.

`prepare.py` reads a supplied pair index, hashes raw inputs and model/interface
files, checks WAV format/level/pitch, and writes a protocol lock. It refuses to
overwrite a directory containing `SCORES.csv`. Do not remove that guard or delete
results just to relock a changed cohort.

## Cached analysis and public inference

With the required private inputs and paths configured:

```sh
python reuse_cache.py
python summarize.py
```

These commands verify the available cache and regenerate scores, descriptive
tables and figures. The cached condition is LEGACY; PEAK is excluded. An input
change requires a new audit. Cache reuse requires matching input, model and
implementation provenance.

To score the supplied recording manifest with the public-model adapters:

```sh
python public_probe.py --all-our
```

Probe interfaces first and fix probe identities in the private protocol.
The implementation saves partial outputs after inputs; inspect restart behaviour
before resuming. Do not label reused cache values as fresh inference.

## External data and segmentation

Source retrieval is a separate authorized operation. `fetch_public.py` uses a
persistent ledger and limits attempts to three per URL (initial plus two retries).
Preserve the ledger, original filenames, attribution and applicable license terms.

1. Read the actual source README. Inspect channels, sample rate, synchronization,
   silence/clipping, score notation and motion-capture columns.
2. Define included and excluded event or pitch-group segments before consulting
   model outputs. Record exact sample indices, string/pitch, boundary source and
   reasons for manual corrections.
3. Freeze the external protocol and input hashes for the new run. Keep full-file
   overviews separate from matched-pitch or matched-bow analyses.
4. Use the PAM and Audiobox adapters described in `INTERFACES.md`. Segment
   inference requires materializing the locked slices in a derived directory and
   preserving native channel/resampling/window handling. Document any justified
   normalization or pitch transformation before scoring.
5. If required physical inputs are unavailable, report the internal classifier
   output as `not_computable`. Do not substitute its zero-input fallback or an
   untrained audio-only branch.

The code's presence does not establish that any external data or segment adapter
has been validated. Complete that validation on the actual authorized inputs.

## Private outputs

Generated source manifests, protocol locks, pair/take indices, segment tables,
scores, summary tables, validation records, caches and figures are experiment
outputs. Keep them in authorized private storage and exclude them from the
code-sharing repository. Preserve prior runs and their provenance when preparing
a new analysis.
