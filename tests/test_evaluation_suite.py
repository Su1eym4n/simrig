from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from simrig.evaluation_suite import (
    EvaluationLimits,
    evaluate_checkpoint_directory,
    run_evaluation_suite,
)
from simrig.gates import adversarial_reward_probes
from simrig.ranking import load_suite_reports, rank_checkpoints
from simrig.task_contract import freeze_task_contract


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "tests" / "fixtures" / "analytic_reach"


def _frozen_contract(tmp: str, source: Path = EXAMPLE / "planar_reach_task.json") -> Path:
    contract = json.loads(source.read_text(encoding="utf-8"))
    path = Path(tmp) / "task.frozen.json"
    path.write_text(
        json.dumps(freeze_task_contract(contract, source=str(source))),
        encoding="utf-8",
    )
    return path


class EvaluationSuiteTests(unittest.TestCase):
    def test_run_directory_hash_identifies_policy_not_unrelated_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = _frozen_contract(tmp)
            (root / "policy.params").write_text(json.dumps({"controller": "valid"}))
            before = run_evaluation_suite(root, contract_path=contract, suite_name="promotion")
            (root / "progress.json").write_text('{"steps": 100}')
            after = run_evaluation_suite(root, contract_path=contract, suite_name="promotion")
        self.assertTrue(before["passed"])
        self.assertTrue(after["passed"])
        self.assertEqual(before["checkpoint"]["sha256"], after["checkpoint"]["sha256"])
        self.assertTrue(before["checkpoint"]["path"].endswith("policy.params"))

    def test_planar_reach_acceptance_and_reward_trap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contract = _frozen_contract(tmp)
            valid = run_evaluation_suite(
                EXAMPLE / "controllers" / "valid.json",
                contract_path=contract,
                suite_name="promotion",
            )
            trap = run_evaluation_suite(
                EXAMPLE / "controllers" / "reward_trap.json",
                contract_path=contract,
                suite_name="promotion",
            )

        self.assertTrue(valid["passed"])
        self.assertEqual(len(valid["records"]), 6)
        self.assertTrue(all(item["task_success"] for item in valid["records"]))
        self.assertFalse(trap["passed"])
        self.assertTrue(all(not item["task_success"] for item in trap["records"]))
        self.assertTrue(
            all(
                item["terminal_reason"]["category"] == "forbidden_contact"
                for item in trap["records"]
            )
        )
        self.assertEqual(len(valid["evaluator"]["sha256"]), 64)
        self.assertEqual(len(valid["report_sha256"]), 64)

        probe = adversarial_reward_probes(valid["records"] + trap["records"])
        ranking = rank_checkpoints([valid, trap])
        self.assertFalse(probe["passed"])
        self.assertEqual(len(probe["high_reward_failures"]), 6)
        self.assertTrue(ranking["checkpoints"][0]["promotion_passed"])
        self.assertIn("valid.json", ranking["checkpoints"][0]["checkpoint"])
        self.assertFalse(ranking["reward_used_for_ranking"])

    def test_bounded_matrix_is_not_promotion_eligible_due_to_missing_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contract = _frozen_contract(tmp)
            report = run_evaluation_suite(
                EXAMPLE / "controllers" / "valid.json",
                contract_path=contract,
                suite_name="promotion",
                limits=EvaluationLimits(max_scenarios=1, max_seeds_per_scenario=1),
            )

        self.assertTrue(report["bounded"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["gate"]["coverage"]["observed"], 1)
        self.assertEqual(len(report["gate"]["coverage"]["missing"]), 5)

    def test_bounded_checkpoint_path_records_evaluator_in_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = _frozen_contract(tmp)
            (root / "policy.params").write_text(
                json.dumps({"controller": "valid"}), encoding="utf-8"
            )
            (root / "run_manifest.json").write_text(
                json.dumps({"status": "running", "independent_evaluations": []}),
                encoding="utf-8",
            )

            summary = evaluate_checkpoint_directory(
                root,
                contract_path=contract,
                suite_name="promotion",
                max_checkpoints=1,
            )
            manifest = json.loads(
                (root / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertTrue(summary["bounded"])
        self.assertEqual(len(summary["reports"]), 1)
        self.assertEqual(
            manifest["independent_evaluations"][0]["evaluator"]["sha256"],
            summary["evaluator"]["sha256"],
        )

    def test_ranking_loader_rejects_tampered_suite_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contract = _frozen_contract(tmp)
            report = run_evaluation_suite(
                EXAMPLE / "controllers" / "valid.json",
                contract_path=contract,
                suite_name="promotion",
            )
            report["passed"] = False
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_suite_reports([path])


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo extra required")
class MujocoReachExampleTests(unittest.TestCase):
    example = ROOT / "examples" / "mujoco_reach"

    def test_real_rollouts_and_reward_free_ranking(self) -> None:
        import mujoco

        with tempfile.TemporaryDirectory() as tmp:
            contract = _frozen_contract(tmp, self.example / "task.json")
            with mock.patch.object(mujoco, "mj_step", wraps=mujoco.mj_step) as step:
                positive = run_evaluation_suite(
                    self.example / "controllers" / "ik.py",
                    contract_path=contract,
                    suite_name="promotion",
                )
            baseline = run_evaluation_suite(
                self.example / "controllers" / "zero.py",
                contract_path=contract,
                suite_name="promotion",
            )

        self.assertTrue(positive["passed"])
        self.assertFalse(baseline["passed"])
        self.assertEqual(len(positive["records"]), 6)
        self.assertEqual(
            step.call_count,
            sum(record["metrics"]["physics_steps"] for record in positive["records"]),
        )
        for record in positive["records"]:
            metrics = record["metrics"]
            self.assertGreater(metrics["physics_steps"], 0)
            self.assertAlmostEqual(metrics["simulation_time_sec"], metrics["physics_steps"] * 0.002)
            self.assertLess(metrics["minimum_target_error"], 0.05)
            self.assertIsNone(record.get("total_reward"))
        ranking = rank_checkpoints([baseline, positive])
        self.assertEqual(ranking["checkpoints"][0]["checkpoint"], positive["checkpoint"]["path"])
        self.assertFalse(ranking["reward_used_for_ranking"])

    def test_invalid_controller_action_fails_as_evaluator_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contract = _frozen_contract(tmp, self.example / "task.json")
            controller = Path(tmp) / "invalid.py"
            controller.write_text("def action(model, data, target):\n    return [float('nan'), 0]\n")
            report = run_evaluation_suite(
                controller, contract_path=contract, suite_name="promotion",
                limits=EvaluationLimits(max_scenarios=1, max_seeds_per_scenario=1),
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["records"][0]["terminal_reason"]["category"], "evaluator_error")


if __name__ == "__main__":
    unittest.main()
