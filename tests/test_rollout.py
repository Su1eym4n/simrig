from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from simrig.presets import checkpoint_config_path
from simrig.rollout import InvalidRolloutState, PolicyRuntime, validate_action, validate_state


def state():
    return SimpleNamespace(
        obs={"state": np.zeros(2)}, reward=0.0, done=0.0,
        data=SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1)),
    )


class RolloutValidityTests(unittest.TestCase):
    def test_non_finite_state_cannot_pass_execution_validation(self):
        for field in ("obs", "reward", "done", "qpos", "qvel"):
            with self.subTest(field=field):
                value = state()
                if field == "obs":
                    value.obs["state"][0] = np.nan
                elif field in ("qpos", "qvel"):
                    getattr(value.data, field)[0] = np.inf
                else:
                    setattr(value, field, np.nan)
                with self.assertRaises(InvalidRolloutState):
                    validate_state(value, {"state": 2})

    def test_shape_keys_and_terminal_flag_must_match(self):
        value = state()
        with self.assertRaisesRegex(InvalidRolloutState, "shape"):
            validate_state(value, {"state": 3})
        with self.assertRaisesRegex(InvalidRolloutState, "keys"):
            validate_state(value, {"pixels": 2})
        value.done = 0.3
        with self.assertRaisesRegex(InvalidRolloutState, "done"):
            validate_state(value)

    def test_actions_reject_wrong_shape_nonfinite_and_out_of_range(self):
        for action in ([0], [0, np.nan], [0, 2], [[0, 0]]):
            with self.subTest(action=action), self.assertRaises(InvalidRolloutState):
                validate_action(action, 2)
        validate_action(np.array([1.0, -1.0]), 2)

    def test_orbax_metadata_is_loaded_from_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            checkpoint = run / "checkpoints" / "000000000160"
            checkpoint.mkdir(parents=True)
            self.assertEqual(checkpoint_config_path(checkpoint), run.resolve() / "config.json")

    def test_unsupported_action_repeat_cannot_silently_change_control_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "policy.params"
            checkpoint.touch()
            (root / "config.json").write_text(json.dumps({"config": {"action_repeat": 2}}))
            with self.assertRaisesRegex(ValueError, "control rate"):
                PolicyRuntime(checkpoint, env_name="Task")

    def test_loader_preserves_unnormalized_policy_and_rejects_wrong_environment(self):
        from simrig import playground_backend as backend

        class Networks:
            @staticmethod
            def make_inference_fn(network):
                return lambda params, deterministic: lambda obs, rng: (obs, {})

        env = SimpleNamespace(observation_size={"state": 2}, action_size=1, reset=lambda r: state(), step=lambda s, a: s)
        normalized = []
        stats = SimpleNamespace(normalize=lambda obs, params: normalized.append(obs))
        captured = []

        def factory(*args, **kwargs):
            captured.append(kwargs["preprocess_observations_fn"])
            return object()

        deps = (SimpleNamespace(jit=lambda f: f), None, SimpleNamespace(load_params=lambda p: ()), stats, Networks)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "policy.params"
            checkpoint.touch()
            (root / "config.json").write_text(json.dumps({"config": {"env_ref": "Task", "normalize_observations": False}}))
            with (
                patch.object(backend, "_import_training_deps", return_value=deps),
                patch.object(backend, "load_env", return_value=env),
                patch("simrig.networks.make_network_factory", return_value=factory),
            ):
                PolicyRuntime(checkpoint, env_name="Task")
                obs = {"state": np.ones(2)}
                self.assertIs(captured[0](obs, None), obs)
                self.assertEqual(normalized, [])
                with self.assertRaisesRegex(ValueError, "Environment differs"):
                    PolicyRuntime(checkpoint, env_name="OtherTask")
                with self.assertWarns(RuntimeWarning):
                    PolicyRuntime(
                        checkpoint,
                        env_name="OtherTask",
                        allow_runtime_mismatch=True,
                    )
