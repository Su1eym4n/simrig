from __future__ import annotations

import unittest

from simrig.predicates import apply_predicates, evaluate_predicate, validate_predicate


class PredicateTests(unittest.TestCase):
    def test_sustained_signal_requires_consecutive_steps(self) -> None:
        record = {
            "events": [
                {"kind": "signal", "name": "inside", "step": step, "active": active}
                for step, active in enumerate([True, True, False, True, True, True])
            ]
        }
        result = evaluate_predicate(
            record,
            {"id": "hold", "type": "sustained", "event": "inside", "hold_steps": 3},
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.actual, 3)

    def test_forbidden_contact_overrides_ordinary_task_failure(self) -> None:
        record = {
            "events": [
                {"kind": "signal", "name": "inside", "step": 0, "active": False},
                {
                    "kind": "contact",
                    "name": "contact",
                    "step": 0,
                    "body_a": "tool",
                    "body_b": "wall",
                    "force": 2.0,
                },
            ]
        }
        result = apply_predicates(
            record,
            [
                {
                    "id": "hold",
                    "type": "sustained",
                    "event": "inside",
                    "hold_steps": 2,
                    "failure_category": "task_failure",
                },
                {
                    "id": "contact",
                    "type": "forbidden_contact",
                    "body_a": "tool",
                    "body_b": "wall",
                    "max_count": 0,
                    "failure_category": "forbidden_contact",
                    "failure_code": "tool_hit_wall",
                },
            ],
        )

        self.assertFalse(result["task_success"])
        self.assertEqual(result["terminal_reason"]["category"], "forbidden_contact")
        self.assertEqual(result["terminal_reason"]["code"], "tool_hit_wall")

    def test_forbidden_predicate_overrides_plugin_generic_failure(self) -> None:
        result = apply_predicates(
            {
                "task_success": False,
                "terminal_reason": {
                    "category": "task_failure",
                    "code": "plugin_failed",
                    "message": "Generic plugin failure.",
                },
                "events": [
                    {
                        "kind": "contact",
                        "name": "contact",
                        "step": 0,
                        "body_a": "tool",
                        "body_b": "wall",
                    }
                ],
            },
            [
                {
                    "id": "contact",
                    "type": "forbidden_contact",
                    "body_a": "tool",
                    "body_b": "wall",
                    "failure_category": "forbidden_contact",
                }
            ],
        )

        self.assertEqual(result["terminal_reason"]["category"], "forbidden_contact")

    def test_predicate_validation_rejects_unknown_type(self) -> None:
        errors = validate_predicate({"id": "x", "type": "robot_specific_magic"})

        self.assertTrue(errors)
        self.assertIn("type", errors[0])

    def test_predicate_validation_rejects_invalid_counts_and_required_flag(self) -> None:
        count_errors = validate_predicate(
            {
                "id": "events",
                "type": "event_count",
                "event": "touch",
                "min_count": 2,
                "max_count": 1,
            }
        )
        contact_errors = validate_predicate(
            {
                "id": "contact",
                "type": "forbidden_contact",
                "body_a": "a",
                "body_b": "b",
                "min_force": -1,
                "required": "yes",
            }
        )

        self.assertTrue(any("must not exceed" in item for item in count_errors))
        self.assertTrue(any("min_force" in item for item in contact_errors))
        self.assertTrue(any("required" in item for item in contact_errors))


if __name__ == "__main__":
    unittest.main()
