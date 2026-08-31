from __future__ import annotations

from pathlib import Path
import unittest


class DemoReachTests(unittest.TestCase):
    def test_reward_and_observations_measure_the_final_integrated_pose(self):
        try:
            import jax
            import jax.numpy as jp
            import mujoco
            import numpy as np
        except ImportError as exc:
            self.skipTest(str(exc))
        from simrig.custom_env import load_custom_env

        env = load_custom_env(Path(__file__).resolve().parents[1] / "examples/demo_reach.py")
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        step = jax.jit(env.step)
        native = mujoco.MjData(env.mj_model)
        for _ in range(8):
            state = step(state, jp.array([0.4, -0.4]))
            native.qpos[:] = np.asarray(state.data.qpos)
            native.qvel[:] = np.asarray(state.data.qvel)
            mujoco.mj_forward(env.mj_model, native)
            actual_ee = native.site_xpos[env.mj_model.site("ee_site").id]
            actual_distance = np.linalg.norm(actual_ee - np.asarray(state.info["target_pos"]))
            self.assertAlmostEqual(float(state.metrics["distance"]), actual_distance, places=6)
            np.testing.assert_allclose(state.obs["state"][-3:], actual_ee - state.info["target_pos"], atol=1e-6)
            self.assertEqual(bool(state.metrics["success"]), actual_distance < 0.05)

    def test_sampled_targets_are_reachable_planar_and_visually_synchronized(self) -> None:
        try:
            import jax
            import jax.numpy as jp

            from simrig.custom_env import load_custom_env
        except ImportError as exc:
            self.skipTest(str(exc))

        env_path = Path(__file__).resolve().parents[1] / "examples" / "demo_reach.py"
        env = load_custom_env(env_path)

        for seed in range(5):
            state = env.reset(jax.random.PRNGKey(seed))
            target = state.info["target_pos"]
            offset = target - env._shoulder_origin
            radius = float(jp.linalg.norm(offset))
            marker = state.data.mocap_pos[env._target_mocap_id]

            self.assertAlmostEqual(float(offset[1]), 0.0)
            self.assertGreaterEqual(radius, float(env.config["target_radius_min"]))
            self.assertLessEqual(radius, float(env.config["target_radius_max"]))
            self.assertTrue(bool(jp.allclose(target, marker)))


if __name__ == "__main__":
    unittest.main()
