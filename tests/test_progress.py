from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simrig.progress import describe_run, format_progress, record_progress
from simrig.success import UNKNOWN_REASON, evaluate_task_success, normalize_success_spec


class ProgressTests(unittest.TestCase):
    def test_record_progress_writes_jsonl_and_progress_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            payload = record_progress(
                run_dir,
                num_steps=1000,
                timesteps=4000,
                metrics={"eval/episode_reward": 1.5, "eval/avg_episode_length": 80.0},
                elapsed_sec=10.0,
            )

            self.assertEqual(payload["num_steps"], 1000)
            self.assertAlmostEqual(payload["fraction"], 0.25)
            self.assertAlmostEqual(payload["eta_sec"], 30.0)
            progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["num_steps"], 1000)
            lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("steps=1000/4000", format_progress(payload))

    def test_describe_run_includes_progress_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "train.pid").write_text("999999\n", encoding="utf-8")
            (run_dir / "train.log").write_text("hello\nworld\n", encoding="utf-8")
            record_progress(
                run_dir,
                num_steps=50,
                timesteps=100,
                metrics={"eval/episode_reward": 2.0},
                elapsed_sec=5.0,
            )

            text = describe_run(run_dir, lines=1)
            self.assertIn("status=stopped", text)
            self.assertIn("artifacts=incomplete", text)
            self.assertIn("progress steps=50/100", text)
            self.assertIn("world", text)
            self.assertNotIn("hello", text)

    def test_describe_run_uses_terminal_manifest_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "aborted",
                        "compute": {
                            "actual_progress_steps": 1000,
                            "elapsed_sec": 20.0,
                            "gpu_hours": 0.01,
                        },
                    }
                ),
                encoding="utf-8",
            )

            text = describe_run(run_dir)

        self.assertIn("status=aborted", text)
        self.assertIn("manifest=aborted", text)
        self.assertIn("actual_progress_steps=1000", text)


class SuccessTests(unittest.TestCase):
    def test_unknown_without_spec_or_values(self) -> None:
        passed, reason = evaluate_task_success([], None)
        self.assertIsNone(passed)
        self.assertEqual(reason, UNKNOWN_REASON)

    def test_any_mode_uses_threshold(self) -> None:
        spec = normalize_success_spec({"metric": "success", "threshold": 0.5, "mode": "any"})
        passed, _ = evaluate_task_success([0.0, 1.0, 0.0], spec)
        self.assertTrue(passed)
        failed, _ = evaluate_task_success([0.1, 0.2], spec)
        self.assertFalse(failed)

    def test_hold_mode_requires_consecutive_steps(self) -> None:
        spec = normalize_success_spec(
            {"metric": "success", "threshold": 1.0, "mode": "hold", "hold_steps": 3}
        )
        passed, _ = evaluate_task_success([0.0, 1.0, 1.0, 1.0], spec)
        self.assertTrue(passed)
        failed, _ = evaluate_task_success([1.0, 1.0, 0.0, 1.0], spec)
        self.assertFalse(failed)

    def test_declared_spec_without_samples_is_failure(self) -> None:
        spec = normalize_success_spec({"metric": "caught"})
        passed, reason = evaluate_task_success([], spec)
        self.assertFalse(passed)
        self.assertIn("caught", reason)


if __name__ == "__main__":
    unittest.main()
