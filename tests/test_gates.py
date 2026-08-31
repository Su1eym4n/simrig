from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from simrig.gates import audit_reward_alignment, evaluate_gate, load_evaluation_records
from tests.test_task_contract import valid_contract


class GateTests(unittest.TestCase):
    def test_different_policy_or_config_cannot_fill_one_acceptance_matrix(self):
        contract = valid_contract()
        contract["evaluation"]["suites"]["nominal"]["scenarios"][0]["seeds"] = [0, 1]
        for key in ("checkpoint_sha256", "checkpoint_config_sha256", "evaluator_sha256"):
            with self.subTest(key=key):
                records = [
                    {"scenario": "nominal", "seed": seed, "task_success": True, key: str(seed)}
                    for seed in (0, 1)
                ]
                result = evaluate_gate(contract, suite_name="nominal", records=records)
                self.assertFalse(result["passed"])
                self.assertTrue(result["identity_errors"])

    def test_changed_report_cannot_be_accepted_using_its_old_digest(self):
        payload = {"records": [{"seed": 0, "task_success": False}]}
        payload["report_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(payload))
            self.assertFalse(load_evaluation_records([path])[0]["task_success"])
            payload["records"][0]["task_success"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_evaluation_records([path])

    def test_missing_outcome_is_not_dropped_from_success_rate(self):
        contract = valid_contract()
        contract["evaluation"]["suites"]["nominal"]["scenarios"][0]["seeds"] = [0, 1]
        result = evaluate_gate(contract, suite_name="nominal", records=[
            {"scenario": "nominal", "seed": 0, "task_success": True},
            {"scenario": "nominal", "seed": 1, "task_success": None},
        ])
        self.assertFalse(result["passed"])
        self.assertEqual(result["requirements"][0]["missing_samples"], 1)

    def test_duplicate_or_unrequested_trials_cannot_inflate_success_rate(self):
        for records in (
            [{"scenario": "nominal", "seed": 0, "task_success": True}] * 2,
            [{"scenario": "nominal", "seed": seed, "task_success": True} for seed in (0, 99)],
        ):
            result = evaluate_gate(valid_contract(), suite_name="nominal", records=records)
            self.assertFalse(result["passed"])

    def test_invalid_state_cannot_pass_even_a_metric_only_gate(self):
        contract = valid_contract()
        contract["evaluation"]["suites"]["nominal"]["requirements"] = [
            {"metric": "distance", "aggregate": "max", "operator": "<", "value": 0.05}
        ]
        result = evaluate_gate(contract, suite_name="nominal", records=[{
            "scenario": "nominal", "seed": 0, "metrics": {"distance": 0},
            "terminal_reason": {"category": "invalid_state"},
        }])
        self.assertFalse(result["passed"])

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
