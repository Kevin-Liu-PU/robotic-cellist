import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from rl.g1_analyze import baseline_adjusted, score_take, task_measures
from rl.g1_session import Session, validate_take


def fixture(directory):
    duration = .1
    timeline = [{"note_index": index, "actual_onset": index * duration, "scheduled_onset": index * duration,
                 "actual_duration": duration, "planned_duration": duration, "depth_m": .001,
                 "u_start": .2, "u_end": .3, "dynamic": "mf", "drift": 0} for index in range(14)]
    (directory / "fixture_performance.json").write_text(json.dumps({"timeline": timeline, "audio_lead_in_s": 0}))
    audio = .1 * np.sin(2 * np.pi * 220 * np.arange(66150) / 44100)
    sf.write(directory / "fixture.wav", audio, 44100)
    state = np.asarray([{"t": 1_800_000_000 + index / 100, "bow_speed": .15,
                         "ft_tx": .1, "ft_ty": .1, "ft_tz": .1} for index in range(151)], dtype=object)
    np.save(directory / "fixture_state.npy", state)
    stroke = {"commanded_segments": [{"speed": .1, "depth_m": .001}, {"speed": .2, "depth_m": .001}],
              "r_envelope": -.4, "r_dynamic": 1.0, "zone": "mf", "dbfs": -20., "shortfall": 0.0}
    (directory / "ep01_strokes.json").write_text(json.dumps({"status": "complete", "records": [
        {"event": "result", "terminated": index == 13, "truncated": False, "info": {"stroke": stroke}}
        for index in range(14)]}))
    (directory / "runtime_snapshot.json").write_text(json.dumps({"guard_reward": {"mode": "off"},
        "argv": {"mock": False}, "scoring_mode": "synchronous",
        "calibration_file": {"text": json.dumps({"zones": {"mf": {"db_lo": -22, "db_hi": -18}}, "gain_offset_db": 0})},
        "runtime_constants": {"rl.piece_env": {"values": {"DEPTH_LO": -.0015, "DEPTH_HI": .002}}}}))


class SessionTests(unittest.TestCase):
    def test_baseline_interpolation(self):
        result = baseline_adjusted({"midpoint": 2, "value": 8}, {"midpoint": 0, "value": 3}, {"midpoint": 4, "value": 5})
        self.assertEqual(result["reference"], 4)
        self.assertEqual(result["delta"], 4)
        for midpoint in (-1, 0, 4, 5):
            with self.assertRaises(ValueError):
                baseline_adjusted({"midpoint": midpoint, "value": 8}, {"midpoint": 0, "value": 3}, {"midpoint": 4, "value": 5})

    def test_valid_take_and_task_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture(directory)
            self.assertEqual(len(validate_take(directory, "OFF")), 5)
            midpoint, measures = task_measures({"take_dir": temporary})
            self.assertGreater(midpoint, 1e9)
            self.assertAlmostEqual(measures["speed_total_variation_mps"], .1)
            self.assertAlmostEqual(measures["envelope_cost_unweighted"], .4)
            self.assertEqual(measures["absolute_onset_error_s"], 0)
            self.assertEqual(measures["loudness_zone_fraction_runtime"], 1)
            self.assertEqual(measures["commanded_depth_excursion_fraction"], 0)
            with self.assertRaises(ValueError):
                validate_take(directory, "FULL")

    def test_incomplete_corrupt_or_disconnected_take_refused(self):
        for corruption in ("timeline", "audio", "state", "audit", "mode"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                fixture(directory)
                if corruption == "timeline":
                    path = directory / "fixture_performance.json"
                    document = json.loads(path.read_text())
                    document["timeline"].pop()
                    path.write_text(json.dumps(document))
                elif corruption == "audio":
                    sf.write(directory / "fixture.wav", np.zeros(66150), 44100)
                elif corruption == "state":
                    np.save(directory / "fixture_state.npy", np.asarray([{"t": 2}, {"t": 1}], dtype=object))
                elif corruption == "audit":
                    path = directory / "ep01_strokes.json"
                    document = json.loads(path.read_text())
                    document["records"][5]["truncated"] = True
                    path.write_text(json.dumps(document))
                else:
                    path = directory / "runtime_snapshot.json"
                    document = json.loads(path.read_text())
                    document["argv"]["mock"] = True
                    path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    validate_take(directory, "OFF")

    def test_six_heads_use_actual_legacy_slicer_without_raw_writes(self):
        class MockJudge:
            def __init__(self):
                self.calls = []

            def score_detailed(self, audio, physical):
                self.calls.append((audio.copy(), physical.copy()))
                return dict(overall=.71, tone_quality=.82, attack_quality=.63, release_quality=.54,
                            bow_control=.45, dynamic_accuracy=.36)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture(directory)
            before = {path.name: path.read_bytes() for path in directory.iterdir()}
            judge = MockJudge()
            result, heads = score_take({"take_dir": temporary}, judge)
            self.assertEqual(result["n"], 14)
            self.assertEqual(len(heads), 14)
            self.assertEqual(len(heads[0]), 6)
            self.assertTrue(all(np.max(np.abs(audio)) < .11 for audio, physical in judge.calls))
            self.assertTrue(all(physical[0] == np.float32(.001) and physical[1] == 0 for audio, physical in judge.calls))
            self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})

    def test_failed_launch_never_automatically_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Session(SimpleNamespace(session=temporary, control_dir=None))
            with patch.object(session, "verify_review"), patch("rl.g1_session.subprocess.run", return_value=SimpleNamespace(returncode=3)) as process:
                with self.assertRaises(RuntimeError):
                    session.launch("failure", ["mock-command-not-executed"], {})
                with self.assertRaises(RuntimeError):
                    session.launch("failure", ["mock-command-not-executed"], {})
                self.assertEqual(process.call_count, 1)

    def test_completed_launch_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Session(SimpleNamespace(session=temporary, control_dir=None))
            session.append({"id": "complete", "status": "valid"})
            with patch.object(session, "verify_review"), patch("rl.g1_session.subprocess.run") as process:
                self.assertEqual(session.launch("complete", ["mock-command-not-executed"], {})["status"], "valid")
                process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
