from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "simrig" / "SKILL.md"
TASK_DESIGN = ROOT / "skills" / "simrig" / "references" / "task-design.md"
EVALUATION = (
    ROOT / "skills" / "simrig" / "references" / "evaluation-and-operations.md"
)


class SimRigSkillGuidanceTests(unittest.TestCase):
    def test_physical_success_gate_precedes_environment_authoring(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        gate = text.index("### 2. Define physical success before implementation")
        authoring = text.index("### 4. Build a custom scene or task")

        self.assertLess(gate, authoring)
        self.assertIn("Do not freeze the contract", text)
        self.assertIn("known-valid positive", text)
        self.assertIn("deliberately exploitative", text)
        self.assertRegex(text, re.compile(r"cannot\s+guarantee"))

    def test_references_cover_physical_definition_and_independent_controls(self) -> None:
        task_design = TASK_DESIGN.read_text(encoding="utf-8")
        evaluation = EVALUATION.read_text(encoding="utf-8")

        for requirement in (
            "Measured entities",
            "Units and frames",
            "Terminal precedence",
            "Feasibility audit",
            "False positives and controls",
            "held-out",
        ):
            self.assertIn(requirement, task_design)
        for requirement in (
            "independent of reward",
            "zero-action",
            "known-valid control",
            "deliberately exploitative",
            "simrig reward-probe",
        ):
            self.assertIn(requirement, evaluation)


if __name__ == "__main__":
    unittest.main()
