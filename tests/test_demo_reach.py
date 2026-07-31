from __future__ import annotations

from pathlib import Path
import unittest


class DemoReachTests(unittest.TestCase):
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
