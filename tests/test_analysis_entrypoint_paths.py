"""Offline path contracts using temporary logs and blocked model/audio imports."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
UNAVAILABLE_DEPENDENCIES = {
    name: None for name in
    ("numpy", "torch", "torchaudio", "soundfile", "scipy", "librosa",
     "audiobox_aesthetics", "PAM", "classifier")
}


def load_script(relative):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location("_analysis_path_contract", path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(path.parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


class AnalysisEntrypointPathTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.dependencies = patch.dict(sys.modules, UNAVAILABLE_DEPENDENCIES)
        self.environment.start()
        self.dependencies.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.dependencies.stop)

    def test_external_judge_import_and_help_need_no_logs_or_models(self):
        script = load_script("analysis/external_judges/score_takes.py")
        with patch.object(sys, "argv", ["score_takes.py", "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    script.main()
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--only", output.getvalue())

    def test_external_judge_relocates_logged_paths_without_changing_pairs(self):
        script = load_script("analysis/external_judges/score_takes.py")
        with tempfile.TemporaryDirectory(prefix="private inputs ") as directory:
            root = Path(directory) / "recordings with spaces"
            root.mkdir()
            (root / "pairs.log").write_text(
                "=== Base1 01:02:03\n"
                "perform -> /old machine/private/perform_000001   \n"
                "=== RL1 01:02:04\n"
                "perform -> C:\\old machine\\private\\perform_000002\n",
                encoding="utf-8",
            )
            with patch.object(script, "CP", root):
                takes = script.parse_log("pairs.log")
                self.assertEqual(takes, [
                    ("Base1", root / "perform_000001"),
                    ("RL1", root / "perform_000002"),
                ])
                self.assertEqual(script.pairs_from(takes), [
                    (root / "perform_000001", root / "perform_000002"),
                ])
                self.assertEqual(script.P("000001"), root / "perform_000001")

    def test_missing_historical_log_fails_before_model_or_output_access(self):
        script = load_script("analysis/external_judges/score_takes.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(script, "CP", root), patch.object(script, "OUT", root / "results"):
                with patch.object(sys, "argv", ["score_takes.py"]):
                    with self.assertRaisesRegex(FileNotFoundError, "CELLO_DATA_ROOT"):
                        script.main()
            self.assertFalse((root / "results").exists())

    def test_pam_requires_explicit_source_before_importing_models(self):
        script = load_script("analysis/external_judges/score_takes.py")
        with self.assertRaisesRegex(FileNotFoundError, "CELLO_PAM_ROOT"):
            script.load_pam()

    def test_missing_wav_has_actionable_error(self):
        script = load_script("analysis/external_judges/score_takes.py")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Expected one WAV.*CELLO_DATA_ROOT"):
                script.wav_of(Path(directory))

    def test_reweight_help_needs_no_reference_or_dependencies(self):
        script = load_script("analysis/reward_reweight_analysis/reweight_analysis.py")
        with patch.object(sys, "argv", ["reweight_analysis.py", "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    script.main()
        self.assertEqual(result.exception.code, 0)
        self.assertIn("CELLO_REFERENCE_ROOT", output.getvalue())

    def test_reweight_requires_explicit_reference(self):
        script = load_script("analysis/reward_reweight_analysis/reweight_analysis.py")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["reweight_analysis.py", "--wav-dir", directory]):
                with self.assertRaisesRegex(FileNotFoundError, "CELLO_REFERENCE_ROOT"):
                    script.main()

    def test_prepare_requires_private_manifest_before_audio_import(self):
        script = load_script("analysis/external_eval/vienna_2024/prepare.py")
        with self.assertRaisesRegex(FileNotFoundError, "CELLO_PAIR_MANIFEST"):
            script.main()

    def test_prepare_does_not_query_unconfigured_paper_repository(self):
        script = load_script("analysis/external_eval/vienna_2024/prepare.py")
        with patch.object(script.subprocess, "run", side_effect=AssertionError("unexpected git call")):
            self.assertEqual(script.git_head(script.PAPER_ROOT), "not_configured")

    @staticmethod
    def pam_fixture(script, root):
        models = {}
        for name, relative in {
            "pam_source": "PAM/PAM.py",
            "pam_config": "PAM/config.yml",
            "pam_dataset": "dataset.py",
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test input; never executed\n", encoding="utf-8")
            models[name] = {"path": str(path), "sha256": script.sha256(path)}
        return {"models": models}

    def test_probe_accepts_locked_pam_sources_but_rejects_other_checkout(self):
        script = load_script("analysis/external_eval/vienna_2024/public_probe.py")
        with tempfile.TemporaryDirectory(prefix="pam sources ") as directory:
            root = Path(directory)
            original, other = root / "source A", root / "source B"
            protocol = self.pam_fixture(script, original)
            self.pam_fixture(script, other)  # Identical bytes do not imply the same selected source.
            with patch.dict(os.environ, {"CELLO_PAM_ROOT": str(original)}):
                self.assertEqual(script.validate_pam_selection(protocol), original.resolve())
            (root / "PROTOCOL_LOCK.json").write_text(json.dumps(protocol), encoding="utf-8")
            with patch.dict(os.environ, {"CELLO_PAM_ROOT": str(other)}), patch.object(script, "ROOT", root):
                with patch.object(sys, "argv", ["public_probe.py"]):
                    with self.assertRaisesRegex(ValueError, "CELLO_PAM_ROOT differs"):
                        script.main()  # Reject before blocked numpy/torch imports or take reads.
            for entry in protocol["models"].values():
                self.assertEqual(script.sha256(Path(entry["path"])), entry["sha256"])

    def test_probe_requires_each_complete_pam_source_pin(self):
        script = load_script("analysis/external_eval/vienna_2024/public_probe.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self.pam_fixture(script, root)
            for name in protocol["models"]:
                for missing in ("entry", "path", "sha256"):
                    with self.subTest(name=name, missing=missing):
                        incomplete = json.loads(json.dumps(protocol))
                        if missing == "entry":
                            del incomplete["models"][name]
                        else:
                            incomplete["models"][name][missing] = ""
                        with self.assertRaisesRegex(ValueError, "Missing PAM source pin.*fresh protocol lock"):
                            script.validate_pam_selection(incomplete)

    def test_probe_rechecks_each_pam_source_hash_before_reuse(self):
        script = load_script("analysis/external_eval/vienna_2024/public_probe.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self.pam_fixture(script, root)
            with patch.dict(os.environ, {"CELLO_PAM_ROOT": str(root)}):
                for name, entry in protocol["models"].items():
                    with self.subTest(name=name):
                        path = Path(entry["path"])
                        original = path.read_bytes()
                        script.validate_pam_selection(protocol)
                        path.write_bytes(original + b"changed")
                        with self.assertRaisesRegex(ValueError, "Model/interface changed"):
                            script.validate_pam_selection(protocol)
                        path.write_bytes(original)

    def test_pam_loader_rejects_conflicting_cached_source_without_model_imports(self):
        script = load_script("analysis/external_judges/score_takes.py")
        probe = load_script("analysis/external_eval/vienna_2024/public_probe.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pam_fixture(probe, root / "selected")
            for name in ("PAM", "PAM.PAM", "dataset"):
                with self.subTest(name=name):
                    cached = ModuleType(name)
                    cached.__file__ = str(root / "other" / (name + ".py"))
                    with patch.dict(sys.modules, {name: cached}):
                        with self.assertRaisesRegex(RuntimeError, "outside the selected PAM checkout"):
                            script.load_pam(pam_root=root / "selected")
                        self.assertIs(sys.modules[name], cached)

    def test_public_probe_help_needs_no_protocol_or_model_imports(self):
        script = load_script("analysis/external_eval/vienna_2024/public_probe.py")
        with patch.object(sys, "argv", ["public_probe.py", "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    script.main()
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--all-our", output.getvalue())

    def test_cache_reuse_rejects_other_provenance_path_with_identical_bytes(self):
        script = load_script("analysis/external_eval/vienna_2024/reuse_cache.py")
        with tempfile.TemporaryDirectory(prefix="cache inputs ") as directory:
            root = Path(directory)
            original, other = root / "cohort A.json", root / "cohort B.json"
            original.write_text("{}", encoding="utf-8")
            other.write_bytes(original.read_bytes())
            protocol = {"cache": {"provenance_path": str(original), "provenance_sha256": script.digest(original)}}
            with patch.object(script, "CACHE_MANIFEST", original):
                script.validate_cache_provenance(protocol)
            (root / "PROTOCOL_LOCK.json").write_text(json.dumps(protocol), encoding="utf-8")
            source = root / "scores.csv"
            source.write_text("", encoding="utf-8")
            with patch.object(script, "CACHE_MANIFEST", other), patch.object(script, "CACHE_SOURCE", source):
                with patch.object(script, "ROOT", root):
                    with self.assertRaisesRegex(ValueError, "CELLO_CACHE_MANIFEST differs"):
                        script.main()  # Reject before reading any cohort or score tables.
            self.assertEqual(script.digest(original), protocol["cache"]["provenance_sha256"])

    def test_cache_reuse_rejects_changed_or_unpinned_provenance(self):
        script = load_script("analysis/external_eval/vienna_2024/reuse_cache.py")
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "cohort.json"
            manifest.write_text("{}", encoding="utf-8")
            protocol = {"cache": {"provenance_path": str(manifest), "provenance_sha256": script.digest(manifest)}}
            with patch.object(script, "CACHE_MANIFEST", manifest):
                for field in ("provenance_path", "provenance_sha256"):
                    with self.subTest(field=field):
                        incomplete = {"cache": dict(protocol["cache"], **{field: ""})}
                        with self.assertRaisesRegex(ValueError, "Missing cache provenance pin.*fresh protocol lock"):
                            script.validate_cache_provenance(incomplete)
                manifest.write_text('{"changed": true}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Cache provenance manifest changed"):
                    script.validate_cache_provenance(protocol)


if __name__ == "__main__":
    unittest.main()
