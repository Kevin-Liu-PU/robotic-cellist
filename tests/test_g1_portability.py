"""File-only checks for reviewed G1 inputs; no models or robot modules are loaded."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl import g1_session


class ReviewedInputsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repository with spaces"
        self.root.mkdir()
        self.root_patch = patch.object(g1_session, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.no_spawn = patch.object(g1_session.subprocess, "run", side_effect=AssertionError("must not spawn"))
        self.no_spawn.start()
        self.addCleanup(self.no_spawn.stop)
        self.checkpoints = {}
        for seed in range(3):
            relative = f"private inputs/seed {seed}.zip"
            path = self.root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"opaque checkpoint fixture {seed}".encode())
            self.checkpoints[str(seed)] = {"path": relative, "sha256": g1_session.digest(path)}

    def review_fixture(self):
        pins = {}
        for relative in g1_session.REQUIRED_RUNTIME_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            pins[relative] = g1_session.digest(path)
        prereg = self.root / "private inputs/preregistration.txt"
        prereg.write_text("reviewed protocol fixture", encoding="utf-8")
        return {"status": "G1_REVIEW_PASS", "guard_off_nonzero_terms_removed": 9,
                "historical_full_seed2_timesteps": 1318, "runtime_files": pins,
                "prereg_path": "private inputs/preregistration.txt", "prereg_sha256": g1_session.digest(prereg),
                "full_checkpoints": self.checkpoints}

    def session_fixture(self):
        review = self.review_fixture()
        path = self.root / "review.json"
        path.write_text(json.dumps(review), encoding="utf-8")
        options = SimpleNamespace(session=str(self.root / "session"), control_dir=None,
                                  review=str(path), attended=True)
        return g1_session.Session(options), review, path

    def test_relative_paths_resolve_against_repository_and_absolute_paths_are_supported(self):
        external = Path(self.temporary.name) / "external checkpoint.zip"
        external.write_bytes(b"external fixture")
        self.checkpoints["2"] = {"path": str(external), "sha256": g1_session.digest(external).upper()}
        actual = g1_session.reviewed_full_checkpoints({"full_checkpoints": self.checkpoints})
        self.assertEqual(actual[0], (self.root / "private inputs/seed 0.zip").resolve())
        self.assertEqual(actual[2], external.resolve())

    def test_all_three_explicit_seeds_are_required_without_historical_fallback(self):
        for entries in (None, {}, {"0": self.checkpoints["0"]}, {**self.checkpoints, "3": self.checkpoints["0"]}):
            with self.subTest(entries=entries), self.assertRaisesRegex(ValueError, "seeds 0, 1, and 2"):
                g1_session.reviewed_full_checkpoints({"full_checkpoints": entries})

    def test_missing_and_changed_checkpoint_are_rejected(self):
        path = self.root / self.checkpoints["1"]["path"]
        path.write_bytes(b"changed fixture")
        with self.assertRaisesRegex(ValueError, "changed for seed 1"):
            g1_session.reviewed_full_checkpoints({"full_checkpoints": self.checkpoints})
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing reviewed FULL checkpoint for seed 1"):
            g1_session.reviewed_full_checkpoints({"full_checkpoints": self.checkpoints})

    def test_invalid_checkpoint_pin_and_empty_path_are_rejected(self):
        self.checkpoints["0"]["sha256"] = "not a digest"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            g1_session.reviewed_full_checkpoints({"full_checkpoints": self.checkpoints})
        self.checkpoints["0"]["sha256"] = "0" * 64
        self.checkpoints["0"]["path"] = ""
        with self.assertRaisesRegex(ValueError, "nonempty string"):
            g1_session.reviewed_full_checkpoints({"full_checkpoints": self.checkpoints})

    def test_review_rechecks_checkpoint_bytes_without_loading_models(self):
        session, _, _ = self.session_fixture()
        session.verify_review()
        session.full_checkpoints[2].write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "changed for seed 2"):
            session.verify_review()
        self.assertFalse(session.directory.exists())

    def test_review_cannot_refresh_pins_during_a_session(self):
        session, review, path = self.session_fixture()
        session.verify_review()
        checkpoint = session.full_checkpoints[0]
        checkpoint.write_bytes(b"replacement")
        review["full_checkpoints"]["0"]["sha256"] = g1_session.digest(checkpoint)
        path.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "session review changed"):
            session.verify_review()

    def test_historical_gate_runtime_and_attended_requirements_remain(self):
        session, review, path = self.session_fixture()
        session.options.attended = False
        with self.assertRaisesRegex(ValueError, "attended"):
            session.verify_review()
        session.options.attended = True
        del review["runtime_files"]["rl/gate.json"]
        path.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source/model/gate pins"):
            session.verify_review()

    def test_review_rechecks_gate_and_preregistration(self):
        session, _, _ = self.session_fixture()
        session.verify_review()
        gate = self.root / "rl/gate.json"
        original_gate = gate.read_bytes()
        gate.write_bytes(b"changed gate")
        with self.assertRaisesRegex(ValueError, "runtime/input changed: rl/gate.json"):
            session.verify_review()
        gate.write_bytes(original_gate)
        (self.root / "private inputs/preregistration.txt").write_bytes(b"changed protocol")
        with self.assertRaisesRegex(ValueError, "preregistration changed"):
            session.verify_review()

    def test_dry_run_has_explicit_unconfigured_full_inputs_and_creates_no_session(self):
        session = self.root / "unused session"
        output = io.StringIO()
        with patch("sys.argv", ["g1_session.py", "eval", "--session", str(session)]), redirect_stdout(output):
            g1_session.main()
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 30)
        for row in rows:
            if row["arm"] == "FULL":
                self.assertIn(f"<review full_checkpoints[{row['seed']}]>", row["argv"])
        self.assertFalse(session.exists())


if __name__ == "__main__":
    unittest.main()
