from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simrig.scaffold import REQUIRED_SECTION_MARKERS, new_env
from simrig.validate_env import _vision_runtime_checks, validate_env


class ScaffoldTests(unittest.TestCase):
    def test_new_env_writes_section_markers_and_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = new_env("reach bot", "models/bot.xml", root=tmp)
            text = path.read_text(encoding="utf-8")

            self.assertEqual(path, Path(tmp) / "reach_bot.py")
            self.assertIn("NOT TRAINABLE YET", text)
            for marker in REQUIRED_SECTION_MARKERS:
                self.assertIn(marker, text)
            self.assertIn("def reset", text)
            self.assertIn("def step", text)
            self.assertIn("def make_env", text)
            self.assertIn("observation_size", text)
            self.assertIn("action_size", text)

    def test_new_env_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            new_env("bot", "models/bot.xml", root=tmp)
            with self.assertRaises(FileExistsError):
                new_env("bot", "models/bot.xml", root=tmp)


class ValidateEnvTests(unittest.TestCase):
    def test_validate_fresh_scaffold_passes_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = new_env("reach", "models/bot.xml", root=tmp)
            result = validate_env(path)

            self.assertTrue(result.passed)
            self.assertFalse(result.trainable)
            self.assertEqual(result.missing, [])
            self.assertTrue(any("NotImplementedError" in w for w in result.warnings))

    def test_validate_missing_file_fails(self) -> None:
        result = validate_env("/tmp/simrig_missing_env_does_not_exist.py")

        self.assertFalse(result.passed)
        self.assertTrue(any("file not found" in item for item in result.missing))

    def test_validate_incomplete_file_reports_missing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text(
                "MODEL_PATH = 'x'\nENV_NAME = 'y'\n\ndef default_config():\n    return {}\n\n"
                "class CustomEnv:\n    def __init__(self):\n        pass\n",
                encoding="utf-8",
            )
            result = validate_env(path)

            self.assertFalse(result.passed)
            self.assertTrue(any("section marker missing" in item for item in result.missing))
            self.assertTrue(any("CustomEnv method missing: reset" in item for item in result.missing))

    def test_reference_vision_env_passes_static_vision_validation(self) -> None:
        path = Path(__file__).resolve().parents[1] / "examples" / "vision_cartpole.py"

        result = validate_env(path, vision=True)

        self.assertTrue(result.passed, result.missing)
        self.assertEqual(result.network_type, "vision_cnn")
        self.assertEqual(result.vision["declared"]["pixel_keys"], ["pixels/view_0"])

    def test_vision_runtime_checks_shapes_ranges_keys_and_camera(self) -> None:
        import numpy as np

        class Model:
            def camera(self, name):
                if name != "fixed":
                    raise KeyError(name)
                return object()

        class Env:
            mj_model = Model()

        obs = {
            "pixels/view_0": np.zeros((8, 8, 3), dtype=np.float32),
            "state": np.zeros((1,), dtype=np.float32),
            "privileged_state": np.zeros((4,), dtype=np.float32),
        }
        next_obs = dict(obs)
        next_obs["pixels/view_0"] = np.ones((8, 8, 3), dtype=np.float32) * 0.25
        metadata = {
            "network_spec": {
                "factory": {
                    "policy_obs_key": "state",
                    "value_obs_key": "privileged_state",
                }
            },
            "vision_spec": {
                "pixel_keys": ["pixels/view_0"],
                "camera_names": ["fixed"],
                "resolution": [8, 8],
                "frame_stack": 3,
                "channels_per_frame": 1,
                "value_range": [-0.5, 0.5],
            },
        }

        missing, warnings, details = _vision_runtime_checks(
            Env(),
            obs,
            next_obs,
            {key: value.shape for key, value in obs.items()},
            require_vision=True,
            metadata=metadata,
        )

        self.assertEqual(missing, [])
        self.assertEqual(warnings, [])
        self.assertTrue(details["frames"]["pixels/view_0"]["changed_after_step"])

    def test_vision_runtime_accepts_singleton_warp_renderer_axis(self) -> None:
        import numpy as np

        class Env:
            mj_model = None

        obs = {
            "pixels/view_0": np.zeros((1, 8, 8, 3), dtype=np.float32),
            "state": np.zeros((1,), dtype=np.float32),
            "privileged_state": np.zeros((4,), dtype=np.float32),
        }
        next_obs = dict(obs)
        next_obs["pixels/view_0"] = np.ones((1, 8, 8, 3), dtype=np.float32)
        metadata = {
            "network_spec": {"factory": {}},
            "vision_spec": {
                "pixel_keys": ["pixels/view_0"],
                "resolution": [8, 8],
                "frame_stack": 3,
                "channels_per_frame": 1,
                "value_range": [0.0, 1.0],
            },
        }

        missing, warnings, details = _vision_runtime_checks(
            Env(),
            obs,
            next_obs,
            {"pixels/view_0": (8, 8, 3)},
            require_vision=True,
            metadata=metadata,
        )

        self.assertEqual(missing, [])
        self.assertEqual(warnings, [])
        self.assertEqual(
            details["frames"]["pixels/view_0"]["logical_shape"],
            [8, 8, 3],
        )


if __name__ == "__main__":
    unittest.main()
