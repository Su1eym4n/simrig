from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from simrig.gates import audit_reward_alignment, evaluate_gate, load_evaluation_records
from tests.test_task_contract import valid_contract


class GateTests(unittest.TestCase):
    def test_gate_requires_seed_coverage_and_passes_per_scenario(self) -> None:
        contract = valid_contract()
        suite = contract["evaluation"]["suites"]["nominal"]
        suite["scenarios"] = [
            {"name": "nominal", "seeds": [0, 1]},
            {"name": "boundary", "seeds": [0, 1]},
        ]
        records = [
            {"scenario": scenario, "seed": seed, "task_success": True}
            for scenario in ("nominal", "boundary")
            for seed in (0, 1)
        ]

        result = evaluate_gate(contract, suite_name="nominal", records=records)

        self.assertTrue(result["passed"])
        self.assertEqual(result["coverage"]["observed"], 4)
        self.assertEqual(len(result["requirements"]), 2)

    def test_gate_fails_missing_seed_even_when_available_records_pass(self) -> None:
        contract = valid_contract()
        contract["evaluation"]["suites"]["nominal"]["scenarios"][0]["seeds"] = [0, 1]

        result = evaluate_gate(
            contract,
            suite_name="nominal",
            records=[{"scenario": "nominal", "seed": 0, "task_success": True}],
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["coverage"]["missing"], [{"scenario": "nominal", "seed": 1}])

    def test_gate_fails_when_group_metric_is_below_threshold(self) -> None:
        contract = valid_contract()
        records = [
            {"scenario": "nominal", "seed": 0, "task_success": False},
        ]

        result = evaluate_gate(contract, suite_name="nominal", records=records)

        self.assertFalse(result["passed"])
        self.assertEqual(result["requirements"][0]["actual"], 0.0)

    def test_nested_metrics_are_available_to_generic_requirements(self) -> None:
        contract = valid_contract()
        contract["evaluation"]["suites"]["nominal"]["requirements"] = [
            {"metric": "tracking_error", "aggregate": "max", "operator": "<=", "value": 0.1}
        ]

        result = evaluate_gate(
            contract,
            suite_name="nominal",
            records=[{"seed": 0, "metrics": {"tracking_error": 0.08}}],
        )

        self.assertTrue(result["passed"])

    def test_reward_audit_flags_failure_reward_loophole(self) -> None:
        result = audit_reward_alignment(
            [
                {"total_reward": 10.0, "task_success": False},
                {"total_reward": 9.0, "task_success": False},
                {"total_reward": 2.0, "task_success": True},
                {"total_reward": 3.0, "task_success": True},
            ]
        )

        self.assertFalse(result["passed"])
        self.assertTrue(any("Failed episodes" in flag for flag in result["flags"]))

    def test_load_records_accepts_episode_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps({"episodes": [{"seed": 0}, {"seed": 1}]}))

            records = load_evaluation_records([path])

        self.assertEqual([item["seed"] for item in records], [0, 1])
        self.assertTrue(all("source_report" in item for item in records))


if __name__ == "__main__":
    unittest.main()
