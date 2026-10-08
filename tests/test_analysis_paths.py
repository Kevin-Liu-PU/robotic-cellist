"""Offline input-path contracts; no model, audio or hardware imports."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.analysis_paths import (
    configured_path, data_root, recorded_take_path, repo_root,
    require_directory, require_file,
)


class AnalysisPathTests(unittest.TestCase):
    def test_default_repo_is_checkout_and_data_is_below_it(self):
        with patch.dict(os.environ, {}, clear=True):
            root = Path(__file__).resolve().parents[1]
            self.assertEqual(repo_root(), root)
            self.assertEqual(data_root(), root / "rl" / "checkpoints_piece")

    def test_data_root_override_is_independent_of_repo_root(self):
        with tempfile.TemporaryDirectory(prefix="analysis inputs ") as directory:
            root = Path(directory)
            with patch.dict(os.environ, {
                "CELLO_REPO_ROOT": str(root / "code"),
                "CELLO_DATA_ROOT": str(root / "private takes"),
            }, clear=True):
                self.assertEqual(repo_root(), (root / "code").resolve())
                self.assertEqual(data_root(), (root / "private takes").resolve())

    def test_empty_setting_is_rejected_instead_of_falling_back(self):
        with patch.dict(os.environ, {"CELLO_DATA_ROOT": " "}, clear=True):
            with self.assertRaisesRegex(ValueError, "CELLO_DATA_ROOT is empty"):
                data_root()

    def test_optional_external_path_has_no_implicit_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configured_path("CELLO_REFERENCE_ROOT"))
            with self.assertRaisesRegex(FileNotFoundError, "CELLO_REFERENCE_ROOT"):
                require_directory(None, "CELLO_REFERENCE_ROOT")

    def test_missing_or_wrong_kind_input_identifies_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "PAIR_INPUT"):
                require_file(root / "absent.json", "PAIR_INPUT")
            with self.assertRaisesRegex(FileNotFoundError, "PAIR_INPUT"):
                require_file(root, "PAIR_INPUT")
            file = root / "pairs.json"
            file.write_text("{}", encoding="utf-8")
            self.assertEqual(require_file(file, "PAIR_INPUT"), file)
            with self.assertRaisesRegex(FileNotFoundError, "DATA_INPUT"):
                require_directory(file, "DATA_INPUT")

    def test_logged_take_relocation_keeps_id_for_both_path_styles(self):
        root = Path("configured-data")
        for logged in ["/old-machine/private/perform_000001", r"C:\old-machine\private\perform_000001"]:
            with self.subTest(logged=logged):
                self.assertEqual(recorded_take_path(logged, root), root / "perform_000001")

    def test_log_entry_cannot_select_a_non_take_directory(self):
        for logged in ["/tmp/..", "/tmp/recordings", ""]:
            with self.subTest(logged=logged):
                with self.assertRaisesRegex(ValueError, "perform_"):
                    recorded_take_path(logged, Path("configured-data"))


if __name__ == "__main__":
    unittest.main()
