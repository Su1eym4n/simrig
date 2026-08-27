from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from simrig.evaluator import EvaluationRequest, load_evaluator, run_evaluator


class EvaluatorPluginTests(unittest.TestCase):
    def test_evaluator_hash_includes_source_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluator.py"
            path.write_text(
                textwrap.dedent(
                    """
                    EVALUATOR_SPEC = {"name": "test", "version": "1"}
                    def evaluate(request):
                        return {"metrics": {"x": 1}, "events": []}
                    """
                ),
                encoding="utf-8",
            )

            first = load_evaluator(path, config={"threshold": 1})
            second = load_evaluator(path, config={"threshold": 2})

        self.assertNotEqual(first.manifest["sha256"], second.manifest["sha256"])
        self.assertEqual(first.manifest["protocol_version"], 1)

    def test_evaluator_hash_is_portable_across_absolute_directories(self) -> None:
        source = textwrap.dedent(
            """
            EVALUATOR_SPEC = {"name": "portable", "version": "1"}
            def evaluate(request):
                return {"metrics": {}, "events": []}
            """
        )
        with (
            tempfile.TemporaryDirectory() as first_tmp,
            tempfile.TemporaryDirectory() as second_tmp,
        ):
            first_path = Path(first_tmp) / "evaluator.py"
            second_path = Path(second_tmp) / "evaluator.py"
            first_path.write_text(source, encoding="utf-8")
            second_path.write_text(source, encoding="utf-8")

            first = load_evaluator(first_path)
            second = load_evaluator(second_path)

        self.assertEqual(first.manifest["sha256"], second.manifest["sha256"])
        self.assertNotEqual(first.manifest["path"], second.manifest["path"])

    def test_evaluator_exception_becomes_machine_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text(
                "EVALUATOR_SPEC = {'name': 'broken', 'version': '1'}\n"
                "def evaluate(request):\n    raise RuntimeError('boom')\n",
                encoding="utf-8",
            )
            evaluator = load_evaluator(path)
            result = run_evaluator(
                evaluator,
                EvaluationRequest(
                    checkpoint="policy.params",
                    environment="Task",
                    backend="test",
                    suite="nominal",
                    scenario="nominal",
                    parameters={},
                    seed=0,
                    max_steps=10,
                    task_contract_sha256="abc",
                ),
                predicates=[],
            )

        self.assertFalse(result["task_success"])
        self.assertEqual(result["terminal_reason"]["category"], "evaluator_error")
        self.assertEqual(result["terminal_reason"]["code"], "evaluator_exception")

    def test_evaluator_module_supports_standard_dataclass_decorators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataclass_evaluator.py"
            path.write_text(
                textwrap.dedent(
                    """
                    from dataclasses import dataclass

                    EVALUATOR_SPEC = {"name": "dataclass-test", "version": "1"}

                    @dataclass
                    class Result:
                        value: int

                    def evaluate(request):
                        return {"metrics": {"value": Result(1).value}, "events": []}
                    """
                ),
                encoding="utf-8",
            )

            evaluator = load_evaluator(path)

        self.assertEqual(evaluator.spec["name"], "dataclass-test")


if __name__ == "__main__":
    unittest.main()
