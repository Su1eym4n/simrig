from __future__ import annotations

import unittest

from simrig.playground_backend import _apply_command, _copy_mocap_state, inspect_env, list_envs


class PlaygroundBackendTests(unittest.TestCase):
    def test_copy_mocap_state_copies_optional_arrays(self) -> None:
        class Data:
            mocap_pos = [[1.0, 2.0, 3.0]]
            mocap_quat = [[1.0, 0.0, 0.0, 0.0]]

        class Target:
            mocap_pos = [[0.0, 0.0, 0.0]]
            mocap_quat = [[0.0, 0.0, 0.0, 0.0]]

        target = Target()
        _copy_mocap_state(Data(), target)

        self.assertEqual(target.mocap_pos, [[1.0, 2.0, 3.0]])
        self.assertEqual(target.mocap_quat, [[1.0, 0.0, 0.0, 0.0]])

    def test_apply_command_rebuilds_command_observation(self) -> None:
        class State:
            info = {"command": [0.0, 0.0, 0.0]}
            data = object()

            def replace(self, **updates):
                for name, value in updates.items():
                    setattr(self, name, value)
                return self

        class Env:
            def _get_obs(self, data, info):
                del data
                return {"state": list(info["command"])}

        state, applied = _apply_command(Env(), State(), [0.8, 0.0, 0.0])

        self.assertTrue(applied)
        self.assertEqual(state.info["command"], [0.8, 0.0, 0.0])
        self.assertEqual(state.obs["state"], [0.8, 0.0, 0.0])

    def test_unknown_playground_env_reports_failure_when_backend_available(self) -> None:
        try:
            report = inspect_env("DefinitelyMissingEnv")
        except RuntimeError as exc:
            self.skipTest(str(exc))

        self.assertFalse(report.loaded)
        self.assertTrue(report.errors)

    def test_list_envs_when_backend_available(self) -> None:
        try:
            envs = list_envs()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        self.assertTrue(envs)
        self.assertIn("name", envs[0])


if __name__ == "__main__":
    unittest.main()
