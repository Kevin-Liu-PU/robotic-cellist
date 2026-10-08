import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl.guard_config import GUARD_WEIGHTS, REWARD_WEIGHTS, apply_guard_mode, guard_configuration, parse_guard_mode
from rl.g1_session import ORDERS, child_environment, evaluation_command, mapped_path, training_command, Session


class GuardConfigurationTests(unittest.TestCase):
    def weights(self):
        return {name: .5 if name == "W_QUALITY" else .25 for name in REWARD_WEIGHTS}

    def test_default_and_explicit_full(self):
        self.assertEqual(parse_guard_mode(), "full")
        self.assertEqual(parse_guard_mode("full"), "full")
        self.assertEqual(parse_guard_mode("off"), "off")

    def test_reject_unknown(self):
        for value in ("", "0", "1", "false", "OFF", "full ", "none", 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_guard_mode(value)

    def test_full_changes_nothing(self):
        weights = self.weights()
        original = weights.copy()
        apply_guard_mode(weights, "full")
        self.assertEqual(weights, original)

    def test_off_removes_nine_terms_and_keeps_classifier(self):
        weights = self.weights()
        apply_guard_mode(weights, "off")
        self.assertEqual(len(GUARD_WEIGHTS), 9)
        self.assertEqual(weights["W_QUALITY"], .5)
        self.assertTrue(all(weights[name] == 0 for name in (*GUARD_WEIGHTS, "W_SPEAK_MEAN")))
        self.assertEqual(guard_configuration(weights, "off")["effective_weights"], weights)

    def test_off_refuses_d_and_nonfinite_classifier(self):
        for value in (0.0, .25, float("nan"), float("inf")):
            weights = self.weights()
            weights["W_QUALITY"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_guard_mode(weights, "off")

    def test_off_detects_later_override(self):
        for name in REWARD_WEIGHTS:
            weights = self.weights()
            apply_guard_mode(weights, "off")
            weights[name] = .1
            with self.subTest(name=name), self.assertRaises(ValueError):
                guard_configuration(weights, "off")

    def test_nonfinite_metadata_refused(self):
        weights = self.weights()
        weights["W_DYNAMIC"] = float("nan")
        with self.assertRaises(ValueError):
            guard_configuration(weights, "full")

    def test_environment_isolated_from_d_and_training_overrides(self):
        with patch.dict(os.environ, {"CELLO_W_QUALITY": "0", "CELLO_ALT_LOOP": "1", "CELLO_TRAINER_OVERRIDE": "bad", "CELLO_GUARD_MODE": "off"}):
            environment = child_environment(False, "FULL")
        self.assertEqual(environment["CELLO_GUARD_MODE"], "full")
        self.assertEqual(environment["CELLO_W_QUALITY"], "0.50")
        self.assertNotIn("CELLO_ALT_LOOP", environment)
        self.assertNotIn("CELLO_TRAINER_OVERRIDE", environment)

    def test_training_recipe_and_no_auto_restart(self):
        command = training_command(2, Path("/tmp/unused"))
        for flag, value in (("--seed", "2"), ("--timesteps", "1400"), ("--reference-every", "5"), ("--ckpt-every", "10"), ("--max-restarts", "0")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--no-calibrated-dynamics", command)
        self.assertIn("--save-stroke-audio", command)
        self.assertEqual(child_environment(True, "OFF")["CELLO_ALT_LOOP"], "1")

    def test_evaluation_balanced_30_takes(self):
        self.assertEqual(ORDERS[0], ("Base", "FULL", "Base", "OFF", "Base"))
        self.assertEqual(ORDERS[1], ("Base", "OFF", "Base", "FULL", "Base"))
        self.assertEqual(3 * sum(map(len, ORDERS)), 30)
        for arm in ("Base", "FULL", "OFF"):
            command = evaluation_command(1, arm, "checkpoint.zip")
            self.assertNotIn("--async-score", command)
            self.assertEqual("--baseline" in command, arm == "Base")
            self.assertEqual(child_environment(arm=arm)["CELLO_GUARD_MODE"], "off" if arm == "OFF" else "full")

    def test_mapping_uses_complete_path_not_newest(self):
        self.assertEqual(mapped_path("perform -> /tmp/perform_20260910_000001\n", "perform -> "), Path("/tmp/perform_20260910_000001").resolve())
        for output in ("nothing", "perform -> /tmp/a\nperform -> /tmp/b\n"):
            with self.assertRaises(ValueError):
                mapped_path(output, "perform -> ")

    def test_stop_wins_over_pause(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            session = Session(SimpleNamespace(session=temporary, control_dir=None))
            (Path(temporary) / "STOP_CHAIN").touch()
            (Path(temporary) / "PAUSE_CHAIN").touch()
            with self.assertRaises(RuntimeError):
                session.check_controls()


if __name__ == "__main__":
    unittest.main()
