from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from simrig.evaluation_suite import (
    EvaluationLimits,
    evaluate_checkpoint_directory,
    run_evaluation_suite,
)
from simrig.gates import adversarial_reward_probes
from simrig.ranking import load_suite_reports, rank_checkpoints
from simrig.task_contract import freeze_task_contract


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "phase1"


def _frozen_contract(tmp: str) -> Path:
    source = EXAMPLE / "planar_reach_task.json"
    contract = json.loads(source.read_text(encoding="utf-8"))
    path = Path(tmp) / "task.frozen.json"
    path.write_text(
        json.dumps(freeze_task_contract(contract, source=str(source))),
        encoding="utf-8",
    )
    return path


class EvaluationSuiteTests(unittest.TestCase):
    def test_planar_reach_acceptance_and_reward_trap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contract = _frozen_contract(tmp)
            valid = run_evaluation_suite(
                EXAMPLE / "checkpoints" / "valid.json",
                contract_path=contract,
                suite_name="promotion",
            )
            trap = run_evaluation_suite(
                EXAMPLE / "checkpoints" / "reward_trap.json",
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
                EXAMPLE / "checkpoints" / "valid.json",
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
                EXAMPLE / "checkpoints" / "valid.json",
                contract_path=contract,
                suite_name="promotion",
            )
            report["passed"] = False
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_suite_reports([path])


if __name__ == "__main__":
    unittest.main()
