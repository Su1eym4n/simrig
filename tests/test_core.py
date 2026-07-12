from __future__ import annotations

import json
import unittest

from simrig.core import ModelInspectionReport, TrainabilityStatus, report_markdown, to_dict


class CoreTests(unittest.TestCase):
    def test_report_serializes_enum(self) -> None:
        report = ModelInspectionReport(
            name="tiny",
            path="/tmp/tiny.xml",
            backend="mujoco",
            status=TrainabilityStatus.SIMULATABLE,
            compiled=True,
            stepped=True,
        )

        data = to_dict(report)

        self.assertEqual(data["status"], "simulatable")
        self.assertEqual(json.loads(json.dumps(data))["name"], "tiny")

    def test_to_dict_serializes_numpy_scalars(self) -> None:
        try:
            np = __import__("numpy")
        except ImportError:
            self.skipTest("numpy is not installed")

        data = to_dict(
            {
                "eval/episode_reward": np.float32(-3.692),
                "eval/avg_episode_length": np.float64(30.0),
                "steps": np.int32(4160),
            }
        )

        self.assertAlmostEqual(data["eval/episode_reward"], -3.692, places=3)
        self.assertEqual(data["eval/avg_episode_length"], 30.0)
        self.assertEqual(data["steps"], 4160)
        self.assertEqual(json.loads(json.dumps(data))["steps"], 4160)
        report = ModelInspectionReport(
            name="tiny",
            path="/tmp/tiny.xml",
            backend="mujoco",
            status=TrainabilityStatus.SIMULATABLE,
            compiled=True,
            stepped=True,
        )

        rendered = report_markdown("Model Inspection: tiny", report)

        self.assertIn("# Model Inspection: tiny", rendered)
        self.assertIn("**Compiled:** True", rendered)


if __name__ == "__main__":
    unittest.main()

