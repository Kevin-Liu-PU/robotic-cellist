import ast
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


CANDIDATE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("c1_perform_audit", CANDIDATE / "rl/perform_audit.py")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


@dataclasses.dataclass
class Record:
    duration: float = 1.0


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.piece = self.folder / "score.mxl"
        self.piece.write_bytes(b"synthetic-score-identity")

    def args(self, **changes):
        values = dict(policy_input_mode="normal", baseline=False, replay_actions=None,
                      checkpoint="policy.zip", compile=False, async_score=False,
                      frozen_references=None, frozen_reference_id=None, c1_cohort=None,
                      replay_preflight_hash=None, piece=str(self.piece))
        values.update(changes)
        return types.SimpleNamespace(**values)

    def plan_directory(self):
        plan = Path(os.environ.get("C1_PLAN_DIR", CANDIDATE.parents[1] / "C1.plan.r4"))
        if not (plan / "cohort.json").is_file():
            self.skipTest("set C1_PLAN_DIR to run the local manifest acceptance checks")
        return plan

    def test_normal_is_bit_identical_and_copies_inputs(self):
        original = np.linspace(-1, 1, 22, dtype=np.float32)
        state = audit.PolicyInput((22,))
        before, delivered = state.deliver(original)
        self.assertEqual(delivered.tobytes(), original.tobytes())
        delivered[0] = 9
        self.assertEqual(before.tobytes(), original.tobytes())

    def test_frozen_reset_and_only_two_indices(self):
        reference = {"constant": 0.54, "alpha": 0.7}
        state = audit.PolicyInput((22,), reference)
        observation = np.arange(22, dtype=np.float32) / 22
        observation[13:15] = 0.5
        _, initial = state.deliver(observation)
        np.testing.assert_array_equal(initial, observation)
        expected_ema = 0.5
        for step in range(40):
            state.advance(completed=True)
            expected_ema = 0.7 * expected_ema + 0.3 * 0.54
            observation[13:15] = [0.2, 0.8]
            original, delivered = state.deliver(observation)
            np.testing.assert_array_equal(np.delete(delivered, [13, 14]), np.delete(original, [13, 14]))
            self.assertEqual(delivered[13], np.float32(0.54))
            self.assertEqual(delivered[14], np.float32(expected_ema))
        self.assertEqual(state.completed, 40)

    def test_bad_reset_and_nonfinite_observations_rejected(self):
        state = audit.PolicyInput((22,), {"constant": 0.5, "alpha": 0.7})
        with self.assertRaises(ValueError):
            state.deliver(np.zeros(22))
        for value in (np.nan, np.inf, -np.inf):
            observation = np.zeros(22)
            observation[13:15] = 0.5
            observation[4] = value
            with self.assertRaises(ValueError):
                state.deliver(observation)

    def test_fault_does_not_advance_virtual_input(self):
        state = audit.PolicyInput((22,), {"constant": 0.6, "alpha": 0.7})
        for truncated, info in ((True, {}), (False, {"truncated_by_disconnect": True}),
                                (False, {"stroke": {"disconnect": True}})):
            with self.assertRaises(RuntimeError):
                audit.require_complete(False, truncated, info)
        state.advance(completed=False)
        self.assertEqual((state.completed, state.ema), (0, 0.5))
        self.assertFalse(audit.require_complete(False, False, {}))
        self.assertTrue(audit.require_complete(True, False, {}))

    def test_action_shape_and_bounds_before_float32_rounding(self):
        for values in ([0] * 5, [[0] * 6], [0] * 5 + [1 + 1e-10],
                       [0] * 5 + [np.nan], [True] * 6, ["0"] * 6):
            with self.assertRaises(ValueError):
                audit.finite_array(values, (6,), "action", bounded=True)
        valid = audit.finite_array([-1, 0, 1, 0.2, -0.2, 0], (6,), "action", bounded=True)
        self.assertEqual(valid.dtype, np.float32)

    def test_mode_rejections(self):
        valid = dict(policy_input_mode="frozen_nominal", frozen_references="refs.json",
                     frozen_reference_id="piece-s0", c1_cohort="cohort.json")
        audit.validate_mode(self.args(**valid), (22,), (6,))
        for change in (dict(baseline=True), dict(replay_actions="actions.json"), dict(checkpoint=None),
                       dict(compile=True), dict(async_score=True), dict(frozen_references=None)):
            with self.assertRaises(ValueError):
                audit.validate_mode(self.args(**dict(valid, **change)), (22,), (6,))
        for observation, action in (((23,), (6,)), ((22,), (2,))):
            with self.assertRaises(ValueError):
                audit.validate_mode(self.args(**valid), observation, action)
        with self.assertRaises(ValueError):
            audit.validate_mode(self.args(frozen_reference_id="unused"), (22,), (6,))
        with self.assertRaises(ValueError):
            audit.validate_mode(self.args(replay_preflight_hash="unused"), (22,), (6,))

    def test_replay_requires_score_content_identity_and_valid_actions(self):
        path = self.folder / "actions.json"
        document = {"piece_sha256": audit.file_record(self.piece)["sha256"], "actions": [[0] * 6] * 2}
        path.write_text(json.dumps(document))
        actions, metadata = audit.load_replay(path, self.piece, 6, audit.file_record(path)["sha256"])
        self.assertEqual(actions.shape, (2, 6))
        self.assertEqual(metadata["count"], 2)
        with self.assertRaises(ValueError):
            audit.load_replay(path, self.piece, 6, "wrong")
        for change in (dict(piece_sha256="wrong"), dict(actions=[]),
                       dict(actions=[[0] * 5]), dict(actions=[[0] * 5 + [1.1]])):
            path.write_text(json.dumps(dict(document, **change)))
            with self.assertRaises(ValueError):
                audit.load_replay(path, self.piece, 6)

    def test_no_automatic_scoring_approval(self):
        with self.assertRaises(ValueError):
            audit.load_scoring_approval(None)
        path = self.folder / "approval.json"
        document = dict(status="deferred", author="fixture", offline_pipeline="fixture",
                        window_alignment="fixture", live_mode="synchronous",
                        artifacts=[audit.file_record(self.piece)])
        path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            audit.load_scoring_approval(path)
        document["status"] = "approved"
        path.write_text(json.dumps(document))
        self.assertEqual(audit.load_scoring_approval(path)["configuration"]["status"], "approved")
        self.piece.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            audit.load_scoring_approval(path)

    def test_checkpoint_suffix_resolution(self):
        checkpoint = self.folder / "policy.zip"
        checkpoint.write_bytes(b"fixture")
        self.assertEqual(audit.checkpoint_file(checkpoint.with_suffix("")), checkpoint.resolve())

    def test_loaded_scorer_must_match_approved_checkpoint(self):
        scorer = types.SimpleNamespace(inner=types.SimpleNamespace(checkpoint_path=self.piece))
        approval = {"configuration": {"artifacts": [audit.file_record(self.piece)]}}
        audit.validate_scorer_approval(scorer, approval)
        approval["configuration"]["artifacts"][0]["sha256"] = "wrong"
        with self.assertRaises(ValueError):
            audit.validate_scorer_approval(scorer, approval)

    def test_actual_r4_cohort_and_all_references_read_only(self):
        plan = self.plan_directory()
        cohort = json.loads((plan / "cohort.json").read_text())
        count = 0
        for piece in cohort["pieces"]:
            for checkpoint in piece["checkpoints"]:
                binding = audit.bind_cohort(plan / "cohort.json", piece["score"]["path"], checkpoint["path"])
                reference = audit.load_reference(
                    plan / "frozen_references.json", f"{piece['piece']}-s{checkpoint['seed']}-frozen-nominal",
                    binding, 0.7)
                self.assertEqual(reference["checkpoint_sha256"], checkpoint["sha256"])
                count += 1
        self.assertEqual(count, 15)
        with self.assertRaises(ValueError):
            audit.bind_cohort(plan / "cohort.json", self.piece)

    def test_reference_rejects_wrong_binding_alpha_and_tampered_constant(self):
        plan = self.plan_directory()
        piece = json.loads((plan / "cohort.json").read_text())["pieces"][0]
        binding = audit.bind_cohort(plan / "cohort.json", piece["score"]["path"], piece["checkpoints"][0]["path"])
        document = json.loads((plan / "frozen_references.json").read_text())
        reference_id = document["references"][0]["reference_id"]
        for alpha in (np.nan, 1, -0.1, True):
            with self.assertRaises(ValueError):
                audit.load_reference(plan / "frozen_references.json", reference_id, binding, alpha)
        path = self.folder / "refs.json"
        document["references"][0]["constant"] += 0.01
        path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            audit.load_reference(path, reference_id, binding, 0.7)
        with self.assertRaises(ValueError):
            audit.load_reference(plan / "frozen_references.json", reference_id,
                                 dict(binding, piece="wrong"), 0.7)

    def test_failure_journal_survives_exception(self):
        path = self.folder / "ep01_strokes.json"
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with audit.EpisodeAudit(path, {"mode": "normal"}) as journal:
                journal.record({"event": "command", "action": np.zeros(6)})
                raise RuntimeError("fixture")
        document = json.loads(path.read_text())
        self.assertEqual(document["status"], "aborted")
        self.assertEqual(len(document["records"]), 3)
        self.assertEqual(document["records"][-1]["error"], "RuntimeError: fixture")
        self.assertEqual(len(path.with_suffix(".jsonl").read_text().splitlines()), 3)

    def test_journal_serializes_full_record_and_nonfinite_explicitly(self):
        path = self.folder / "ep01_strokes.json"
        with audit.EpisodeAudit(path, {}) as journal:
            journal.record({"info": {"stroke": {"quality": np.float32(0.5),
                                               "physical": np.zeros(6), "dbfs": -np.inf}}})
        document = json.loads(path.read_text())
        self.assertEqual(document["status"], "complete")
        self.assertEqual(document["records"][1]["info"]["stroke"]["dbfs"], {"nonfinite": "-inf"})
        self.assertEqual(len(document["serialization_s"]), 3)

    def test_runtime_snapshot_uses_effective_values_not_defaults(self):
        calibration_path = self.folder / "calibration.json"
        calibration_path.write_text('{"gain_offset_db": 4.2}')
        environment = types.SimpleNamespace(
            loudness=types.SimpleNamespace(path=calibration_path, gain_offset_db=5.76),
            ema_alpha=0.7, notes=[Record()], plan=[Record()])
        arguments = self.args(checkpoint=None)
        snapshot_path = self.folder / "snapshot.json"
        audit.write_snapshot(snapshot_path, self.folder, environment, types.SimpleNamespace(),
                             arguments, None, None, None, None)
        snapshot = json.loads(snapshot_path.read_text())
        self.assertEqual(snapshot["calibration_runtime"]["values"]["gain_offset_db"], 5.76)
        self.assertEqual(json.loads(snapshot["calibration_file"]["text"])["gain_offset_db"], 4.2)
        self.assertEqual(snapshot["runtime_environment"]["values"]["ema_alpha"], 0.7)
        self.assertEqual(snapshot["plan"], [{"duration": 1.0}])

    def test_take_dir_keeps_second_resolution_name_and_never_collides(self):
        import datetime, re
        fixed = datetime.datetime(2026, 9, 8, 21, 30, 0)
        first = audit.new_take_dir(self.folder, now=fixed)
        self.assertEqual(first.name, "perform_20260908_213000")
        second = audit.new_take_dir(self.folder, now=fixed)      # collides, waits for the next real second
        self.assertNotEqual(first, second)
        for path in (first, second):
            self.assertRegex(path.name, r"^perform_\d{8}_\d{6}$")
            self.assertTrue(path.is_dir())

    def test_json_value_handles_sets_and_dataclasses(self):
        self.assertEqual(audit.json_value({"a": frozenset({3, 1, 2})}), {"a": [1, 2, 3]})
        self.assertEqual(audit.json_value([Record(2.0)]), [{"duration": 2.0}])
        with self.assertRaises(TypeError):
            audit.json_value(object())

    def test_snapshot_skips_synthetic_relative_module_paths(self):
        synthetic = types.ModuleType("c1_synthetic_module")
        synthetic.__file__ = "_ops.py"                      # relative, does not exist (torch does this)
        sys.modules["c1_synthetic_module"] = synthetic
        self.addCleanup(sys.modules.pop, "c1_synthetic_module", None)
        environment = types.SimpleNamespace(loudness=None, ema_alpha=0.7, notes=[Record()], plan=[Record()])
        snapshot_path = self.folder / "snapshot.json"
        audit.write_snapshot(snapshot_path, Path.cwd(), environment, types.SimpleNamespace(),
                             self.args(checkpoint=None), None, None, None, None)
        self.assertNotIn("_ops.py", json.loads(snapshot_path.read_text())["source_files"])

    def test_source_parses_and_preflight_precedes_hardware_setup(self):
        source = (CANDIDATE / "rl/perform.py").read_text()
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        def calls(name):
            return [node for node in ast.walk(main) if isinstance(node, ast.Call)
                    and ((isinstance(node.func, ast.Name) and node.func.id == name)
                         or (isinstance(node.func, ast.Attribute) and node.func.attr == name))]
        self.assertLess(calls("validate_mode")[0].lineno, calls("compile_performance")[0].lineno)
        self.assertLess(calls("load_reference")[0].lineno, calls("make_perform_executor")[0].lineno)
        self.assertLess(calls("validate_scorer_approval")[0].lineno, calls("make_perform_executor")[0].lineno)
        self.assertLess(calls("write_snapshot")[0].lineno, calls("reset")[0].lineno)
        self.assertLess(calls("require_complete")[0].lineno, calls("advance")[0].lineno)
        for name in ("perform.py", "perform_audit.py", "piece_env.py", "piece_hardware.py"):
            path = CANDIDATE / "rl" / name
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
