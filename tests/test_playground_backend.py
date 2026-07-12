from __future__ import annotations

import unittest

from simrig.playground_backend import inspect_env, list_envs


class PlaygroundBackendTests(unittest.TestCase):
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

