# Annotation package workflow

The collection tools can create browser annotation packages from separately
supplied recordings. Annotators open a local HTML app and return a JSON export;
they do not need a Label Studio account or a Python environment.

Recording packages, task assignments and returned ratings are not included in
this code release. Confirm the permitted recipients and use of the recordings
before distributing packages. Keep source datasets, packages, assignment manifests
and returned ratings in authorized private storage outside the repository.
The word `public` in the packaging script's filename does not grant permission
to publish the recordings.

## Prepare local inputs

Run the commands below from the repository root using the configured Python
environment. Replace every `/path/to/...` with an absolute path on your machine;
the quoted placeholders also accommodate paths containing spaces. `RATER_1`,
`RATER_2` and `RATER_3` are example IDs, not predefined assignments.

Use a working copy of a dataset containing `metadata.jsonl`, `audio/` and
`states/`. Preserve the original records. To inspect duplicate metadata and
missing or unreferenced files without modifying the working copy:

```sh
python SoundClassifier/Data_Collection/prepare_annotation_dataset.py "/path/to/private_dataset" --dry-run
```

Without `--dry-run`, this tool rewrites metadata and creates a backup by default.
The separate `--remove-unreferenced-files` option deletes unreferenced audio/state
files; it is not part of this example. Review the reported changes before applying
them to a working copy.

## Export task assignments

Choose the annotator IDs, repetition count and assignment seed for the intended
study. The example requests two ratings per recording from three annotators:

```sh
python SoundClassifier/Data_Collection/label_studio_bridge.py export --meta "/path/to/private_dataset/metadata.jsonl" --audio-dir "/path/to/private_dataset/audio" --out-dir "/path/to/private_run/tasks" --annotators RATER_1 RATER_2 RATER_3 --annotations-per-sample 2 --seed 42 --hide-condition
```

`--annotations-per-sample` cannot exceed the annotator count. The tool writes
`tasks_RATER_ID.json`, `label_config.xml`, instructions and `manifest.json`.
Inspect the generated per-annotator and shared-pair counts; no fixed sample count
or exact balance is promised here. Use a new task directory for a new assignment
rather than replacing a prior study's manifest.

`--hide-condition` controls the displayed condition labels. It does not establish
that task metadata or audio filenames are anonymous; inspect the package before
distributing it.

## Build browser packages

Use the same annotator IDs and the task files generated above:

```sh
python SoundClassifier/Data_Collection/package_public_annotation.py --tasks-dir "/path/to/private_run/tasks" --audio-dir "/path/to/private_dataset/audio" --out-dir "/path/to/private_run/packages" --annotators RATER_1 RATER_2 RATER_3
```

The packager creates one folder per ID containing `index.html`, assigned WAV
files in `audio/`, and `README.txt`. It also writes a package manifest. It replaces
an existing folder for the same annotator under `--out-dir`, so use a fresh output
directory and preserve previously distributed packages.

Check that a package opens and its assigned audio is available before sending it.
To create an archive for one annotator:

```sh
python -m zipfile -c "/path/to/private_run/packages/RATER_1_annotation_package.zip" "/path/to/private_run/packages/RATER_1"
```

Repeat with the other assigned IDs as needed. Send each package only to its
intended recipient through the approved study channel.

## Annotator instructions

1. Extract the package and open `index.html` in a desktop browser.
2. Listen to each assigned recording and complete the ratings. Use headphones
   when available.
3. The app stores progress in that browser's local storage. Keep the same package
   and browser profile, and do not clear browser data before exporting results.
4. Choose **Download Results** and return the JSON file through the agreed
   private channel. A downloaded export is separate from browser-local progress.

## Annotation form

The browser package and `label_studio_bridge.py` define these fields:

- Required technical quality: `overall`, `tone_quality`, `bow_control`,
  `attack_quality`, `release_quality` and `dynamic_accuracy`, each rated
  1 (poor), 2 (fair), 3 (good) or 4 (great).
- Required consistency: `dynamic_consistency` (loudness change),
  `tone_consistency` (where tone is clearest), and `noise_consistency`
  (where surface noise is most noticeable), using the displayed ordered choices.
- Optional tonal character: `grainy_smooth`, `harsh_sweet`, `thin_rich`,
  `airy_clear` and `dry_resonant`, each using the displayed 1-5 scale.
- Problem flags and free-text notes. The importer records `audio_problem` and
  `robot_problem` flags without importing quality scores for those annotations.

These are collection fields; their presence does not imply that every field is
used by the classifier's training objective. Preserve the form and instructions
used for a study alongside its private assignment manifest.

## Import returned ratings

Store returned exports in a private run directory and make a working copy of
the source metadata. First inspect the import without changing that copy:

```sh
python SoundClassifier/Data_Collection/label_studio_bridge.py import --meta "/path/to/private_dataset/metadata.jsonl" --exports "RATER_1=/path/to/private_run/returned/annotations_RATER_1.json" "RATER_2=/path/to/private_run/returned/annotations_RATER_2.json" "RATER_3=/path/to/private_run/returned/annotations_RATER_3.json" --dry-run
```

The browser app exports the Label-Studio-shaped JSON accepted by this importer.
Check the imported, flagged and skipped counts and the mapping of annotator IDs.
Remove `--dry-run` to write the annotations into the working metadata file.
Existing annotations for the same annotator and recording are skipped, not
silently replaced. Preserve the original exports and source metadata separately.

## Other annotation interfaces

Label Studio can use the exported task JSON and `label_config.xml`. Its
local-files configuration must match the dataset root. When exporting tasks
for that interface, add `--document-root "/path/to/private_dataset"`; the audio
directory must lie beneath that root. The legacy `start_label_studio.sh` fixes
its environment and document root to repository-local paths, so inspect and
configure it before using external private inputs. It is not required for the
browser packages above.

The terminal annotator collects a single `overall` rating and can use the same
assignment file:

```sh
python SoundClassifier/Data_Collection/annotate.py --meta "/path/to/private_dataset/metadata.jsonl" --audio-dir "/path/to/private_dataset/audio" --annotator RATER_1 --task-file "/path/to/private_run/tasks/tasks_RATER_1.json" --hide-condition
```

It saves ratings into the supplied metadata file. Use a working copy and keep
all generated annotations outside the public repository.
