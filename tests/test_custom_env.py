from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simrig.custom_env import (
    is_env_module_path,
    load_custom_env,
    load_custom_env_metadata,
    resolve_env_label,
)
from simrig.scaffold import new_env
from simrig.validate_env import validate_env


MINIMAL_ENV = '''\
"""Minimal custom env for loader tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_PATH = Path("models/bot.xml")
ENV_NAME = "minimal"


def default_config() -> dict[str, Any]:
    return {"action_scale": 1.0, "impl": "jax"}


def make_env(config_overrides: dict[str, Any] | None = None) -> "CustomEnv":
    config = default_config()
    if config_overrides:
        config.update(config_overrides)
    return CustomEnv(config=config)


@dataclass
class _State:
    obs: dict[str, Any]
    reward: float
    done: bool
    data: Any = None


class CustomEnv:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # SECTION: model loading
        self.config = config or default_config()
        self._action_size = 2

    def reset(self, rng: Any) -> _State:
        # SECTION: reset
        return _State(
            obs={"state": [0.0, 0.0], "privileged_state": [0.0, 0.0, 0.0]},
            reward=0.0,
            done=False,
        )

    def step(self, state: _State, action: Any) -> _State:
        # SECTION: action mapping
        # SECTION: observations
        # SECTION: rewards
        # SECTION: termination
        return _State(
            obs=state.obs,
            reward=1.0,
            done=False,
            data=state.data,
        )

    @property
    def observation_size(self) -> dict[str, int]:
        return {"state": 2, "privileged_state": 3}

    @property
    def action_size(self) -> int:
        return self._action_size
'''


class CustomEnvLoaderTests(unittest.TestCase):
    def test_is_env_module_path(self) -> None:
        self.assertTrue(is_env_module_path("envs/reach.py"))
        self.assertTrue(is_env_module_path(Path("foo/bar.py")))
        self.assertFalse(is_env_module_path("Go1JoystickFlatTerrain"))

    def test_resolve_env_label_uses_stem(self) -> None:
        self.assertEqual(resolve_env_label("envs/my_reach.py"), "my_reach")
        self.assertEqual(resolve_env_label("Go1JoystickFlatTerrain"), "Go1JoystickFlatTerrain")

    def test_load_custom_env_via_make_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.py"
            path.write_text(MINIMAL_ENV, encoding="utf-8")
            env = load_custom_env(path, config_overrides={"impl": "jax"})

            self.assertEqual(env.action_size, 2)
            self.assertEqual(env.config["impl"], "jax")
            self.assertIn("state", env.observation_size)

    def test_load_custom_env_metadata_supports_vision_hooks(self) -> None:
        source = MINIMAL_ENV + '''\

def network_spec():
    return {"type": "vision_cnn", "factory": {"policy_obs_key": "state"}}

VISION_SPEC = {"pixel_keys": ["pixels/view_0"], "requires_impl": "warp"}

def training_config():
    return {"num_timesteps": 1000, "vision": True}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vision.py"
            path.write_text(source, encoding="utf-8")

            metadata = load_custom_env_metadata(path)

        self.assertEqual(metadata["network_spec"]["type"], "vision_cnn")
        self.assertEqual(
            metadata["network_spec"]["factory"]["policy_obs_key"], "state"
        )
        self.assertEqual(metadata["vision_spec"]["requires_impl"], "warp")
        self.assertEqual(metadata["training_config"]["num_timesteps"], 1000)


class ValidateRuntimeTests(unittest.TestCase):
    def test_runtime_validate_minimal_env_without_jax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.py"
            path.write_text(MINIMAL_ENV, encoding="utf-8")
            result = validate_env(path, runtime=True)

            self.assertTrue(result.passed)
            # Without JAX, construct succeeds but trainable stays false.
            try:
                import jax  # noqa: F401

                self.assertTrue(result.trainable)
            except ImportError:
                self.assertFalse(result.trainable)
                self.assertTrue(any("JAX not installed" in w for w in result.warnings))

    def test_runtime_validate_fails_on_not_implemented_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = new_env("broken", "models/bot.xml", root=tmp)
            result = validate_env(path, runtime=True)

            self.assertFalse(result.passed)
            self.assertFalse(result.trainable)
            self.assertTrue(any("runtime load failed" in item for item in result.missing))


if __name__ == "__main__":
    unittest.main()
