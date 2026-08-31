"""Real learned-policy regression checks against a separately trained artifact.

Set SIMRIG_TEST_POLICY to policy.params (CI's tiny run is enough for plumbing).
The complete reference acceptance matrix is run by examples/learned_reach/verify.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = os.environ.get("SIMRIG_TEST_POLICY")


@unittest.skipUnless(POLICY, "Set SIMRIG_TEST_POLICY to exercise a real trained checkpoint")
class LearnedPolicyIntegrationTests(unittest.TestCase):
    def test_parameters_orbax_preview_and_eval_execute_the_same_policy(self):
        from simrig.playground_backend import eval_policy
        from simrig.preview import PolicyPreviewSession
        from simrig.rollout import PolicyRuntime

        policy = Path(POLICY).resolve()
        env_name = str(ROOT / "examples" / "demo_reach.py")
        runtime = PolicyRuntime(policy, env_name=env_name)
        state, rng = runtime.reset(5)
        state, rng, action = runtime.advance(state, rng)
        report = eval_policy(policy, env_name=env_name, seed=5, steps=1)
        self.assertAlmostEqual(report["total_reward"], float(state.reward))
        preview = PolicyPreviewSession(policy, env_name=env_name, seed=5, paused=True)
        try:
            with preview._lock:
                preview._step_once_unlocked()
            np.testing.assert_allclose(preview.state.data.qpos, state.data.qpos)
            with preview._lock:
                while not preview.done and preview.step_count < 200:
                    preview._step_once_unlocked()
            status = preview.status()
            self.assertTrue(status.done)
            self.assertEqual(status.episode_outcome, "success" if status.task_success else "failure")
            self.assertEqual(status.task_success, any(value >= 0.5 for value in preview.success_values))
            preview.reset()
            self.assertIsNone(preview.status().task_success)
            self.assertEqual(preview.status().episode_outcome, "pending")
            self.assertEqual(preview.success_values, [])
        finally:
            preview.close()
            runtime.close()
        directories = sorted((policy.parent / "checkpoints").glob("*"))
        checkpoint = next(path for path in reversed(directories) if path.is_dir())
        restored = PolicyRuntime(checkpoint, env_name=env_name)
        try:
            other, key = restored.reset(5)
            other, key, other_action = restored.advance(other, key)
            np.testing.assert_allclose(action, other_action, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(state.data.qpos, other.data.qpos, rtol=1e-5, atol=1e-6)
        finally:
            restored.close()

    def test_learned_checkpoint_runs_independent_matrix_with_measured_distances(self):
        from simrig.evaluation_suite import EvaluationLimits, run_evaluation_suite
        from simrig.task_contract import save_frozen_task_contract

        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / "contract.json"
            save_frozen_task_contract(ROOT / "examples/learned_reach/task.json", output=frozen)
            report = run_evaluation_suite(
                POLICY, contract_path=frozen, suite_name="promotion",
                limits=EvaluationLimits(max_scenarios=1, max_seeds_per_scenario=1),
            )
        self.assertFalse(report["passed"], "A capped run must not pass full promotion")
        record = report["records"][0]
        self.assertEqual(record["artifact_type"], "learned_policy")
        self.assertGreater(record["steps_completed"], 0)
        self.assertTrue(np.isfinite(record["metrics"]["minimum_target_error"]))
        self.assertEqual(record["task_success"], record["metrics"]["minimum_target_error"] < 0.05)
        self.assertTrue(record["runtime_compatibility"]["compatible"])
        json.dumps(report, allow_nan=False)
