from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from simrig.task_contract import (
    AbortMonitor,
    contract_sha256,
    compare_task_contracts,
    diff_task_contracts,
    enforce_compute_budget,
    freeze_task_contract,
    load_task_contract,
    migrate_task_contract,
    save_frozen_task_contract,
    task_contract_template,
    validate_task_contract,
)


def valid_contract() -> dict:
    contract = task_contract_template("envs/reach.py", name="reach")
    contract["behavior"]["objective"] = "Move the end effector to a sampled target."
    contract["interfaces"]["actions"] = "Normalized joint target deltas at 20 Hz."
    contract["interfaces"]["observations"] = "Joint state and target relative position."
    contract["reset"]["training"] = "Nominal pose with reachable targets."
    contract["reset"]["native"] = "Same distribution; this skill has no predecessor."
    contract["episode"]["horizon_steps"] = 200
    contract["outcomes"]["success"] = "Target error below 2 cm for 10 steps."
    contract["outcomes"]["failure"] = "Forbidden contact, non-finite state, or timeout."
    return contract


class TaskContractTests(unittest.TestCase):
    def test_template_requires_explicit_decisions(self) -> None:
        result = validate_task_contract(task_contract_template("RobotTask"))

        self.assertFalse(result.passed)
        self.assertTrue(any("TODO" in item for item in result.errors))
        self.assertIn("episode.horizon_steps", "\n".join(result.errors))

    def test_valid_contract_has_deterministic_hash(self) -> None:
        contract = valid_contract()
        result = validate_task_contract(contract)

        self.assertTrue(result.passed, result.errors)
        self.assertEqual(result.sha256, contract_sha256(dict(reversed(list(contract.items())))))

    def test_freeze_and_load_verifies_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "task.json"
            source.write_text(json.dumps(valid_contract()), encoding="utf-8")
            frozen_path = save_frozen_task_contract(source)

            contract, envelope = load_task_contract(frozen_path, require_frozen=True)

        self.assertEqual(contract["name"], "reach")
        self.assertIsNotNone(envelope)
        self.assertEqual(envelope["sha256"], contract_sha256(contract))

    def test_tampered_frozen_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.json"
            payload = freeze_task_contract(valid_contract())
            payload["contract"]["name"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_task_contract(path)

    def test_semantic_diff_ignores_key_order(self) -> None:
        left = valid_contract()
        right = json.loads(json.dumps(left))
        right["episode"]["horizon_steps"] = 300

        changes = diff_task_contracts(left, right)

        self.assertEqual(changes, [{"path": "episode.horizon_steps", "left": 200, "right": 300}])

    def test_compute_budget_uses_resolved_rollout_estimate(self) -> None:
        contract = valid_contract()
        contract["compute"]["max_timesteps"] = 500_000

        with self.assertRaisesRegex(ValueError, "estimated=655,360"):
            enforce_compute_budget(
                contract,
                requested_timesteps=500_000,
                estimated_timesteps=655_360,
            )

    def test_abort_monitor_requires_consecutive_triggered_evaluations(self) -> None:
        monitor = AbortMonitor(
            [
                {
                    "metric": "eval/task_success",
                    "operator": "<=",
                    "value": 0,
                    "after_steps": 100,
                    "patience": 2,
                }
            ]
        )

        self.assertIsNone(
            monitor.observe(num_steps=50, metrics={"eval/task_success": 0.0})
        )
        self.assertIsNone(
            monitor.observe(num_steps=100, metrics={"eval/task_success": 0.0})
        )
        reason = monitor.observe(num_steps=200, metrics={"eval/task_success": 0.0})

        self.assertIn("abort rule triggered", reason)

    def test_schema_one_migration_is_explicit_and_valid(self) -> None:
        old = valid_contract()
        old["schema_version"] = 1
        old["evaluation"].pop("evaluator")
        old["evaluation"].pop("predicates")
        old["outcomes"].pop("failure_taxonomy")

        migrated, steps = migrate_task_contract(old)
        result = validate_task_contract(migrated)

        self.assertEqual(migrated["schema_version"], 2)
        self.assertEqual(steps[0]["from"], 1)
        self.assertTrue(result.passed, result.errors)

    def test_compatibility_policy_distinguishes_compute_and_interface_changes(self) -> None:
        original = valid_contract()
        compute_change = json.loads(json.dumps(original))
        compute_change["compute"]["max_timesteps"] *= 2
        interface_change = json.loads(json.dumps(original))
        interface_change["interfaces"]["actions"] = "Different action semantics."

        self.assertTrue(
            compare_task_contracts(
                original, compute_change, policy="training_resume"
            )["compatible"]
        )
        self.assertFalse(
            compare_task_contracts(
                original, compute_change, policy="exact"
            )["compatible"]
        )
        self.assertFalse(
            compare_task_contracts(
                original, interface_change, policy="training_resume"
            )["compatible"]
        )

    def test_contract_rejects_modified_taxonomy_and_duplicate_seeds(self) -> None:
        contract = valid_contract()
        contract["outcomes"]["failure_taxonomy"].remove("timeout")
        contract["evaluation"]["suites"]["nominal"]["scenarios"][0]["seeds"] = [0, 0]

        result = validate_task_contract(contract)

        self.assertFalse(result.passed)
        self.assertTrue(any("stable category" in item for item in result.errors))
        self.assertTrue(any("duplicates" in item for item in result.errors))


if __name__ == "__main__":
    unittest.main()
