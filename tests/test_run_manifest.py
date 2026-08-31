from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from simrig.core import RunConfig
from simrig.progress import record_progress
from simrig.run_manifest import finish_run_manifest, start_run_manifest


class RunManifestTests(unittest.TestCase):
    def test_manifest_records_contract_lineage_and_measured_compute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = RunConfig(
                env_name="RobotTask",
                backend="mujoco-playground",
                preset="smoke",
                output_dir=tmp,
                config={
                    "timesteps": 4096,
                    "estimated_total_timesteps": 4096,
                    "num_envs": 16,
                    "batch_size": 16,
                    "resumed_from": "runs/parent/checkpoints",
                    "runtime": {"python": "test"},
                    "provenance": {
                        "jax": {"devices": [{"platform": "gpu"}, {"platform": "gpu"}]}
                    },
                    "evaluator": {"name": "independent", "sha256": "eval123"},
                },
                command=["simrig", "train", "RobotTask"],
            )
            start_run_manifest(
                run,
                task_contract={"sha256": "abc", "path": "/tmp/task.frozen.json"},
            )
            record_progress(
                tmp,
                num_steps=2048,
                timesteps=4096,
                metrics={"eval/episode_reward": 1.0},
                elapsed_sec=900,
            )
            finish_run_manifest(
                tmp,
                status="completed",
                elapsed_sec=1800,
                gpu_hourly_cost=2.0,
            )
            payload = json.loads(
                (Path(tmp) / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["task_contract"]["sha256"], "abc")
        self.assertEqual(payload["lineage"]["resumed_from"], "runs/parent/checkpoints")
        self.assertEqual(payload["evaluator"]["sha256"], "eval123")
        self.assertEqual(payload["compute"]["actual_progress_steps"], 2048)
        self.assertEqual(payload["compute"]["gpu_hours"], 1.0)
        self.assertEqual(payload["compute"]["estimated_cost"], 2.0)
