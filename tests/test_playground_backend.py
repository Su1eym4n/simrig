from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from simrig.playground_backend import (
    _apply_command,
    _copy_mocap_state,
    inspect_env,
    list_envs,
    resolve_training_config,
    train_ppo,
)


class PlaygroundBackendTests(unittest.TestCase):
    def test_resolve_training_config_scales_upstream_without_losing_tuned_values(self) -> None:
        registry = MagicMock()
        registry.get_default_config.return_value = {"impl": "warp"}
        upstream = {
            "timesteps": 200_000_000,
            "num_envs": 8192,
            "batch_size": 256,
            "reward_scaling": 7.0,
            "network_factory": {"policy_hidden_layer_sizes": (512, 256, 128)},
        }
        with (
            patch("simrig.playground_backend._import_registry", return_value=registry),
            patch("simrig.playground_backend._warp_available", return_value=True),
            patch(
                "simrig.playground_backend._upstream_ppo_config",
                return_value=(upstream, "upstream:test"),
            ),
        ):
            config = resolve_training_config(
                "RobotTask",
                preset_name="smoke",
                impl="auto",
                seed=9,
            )

        self.assertEqual(config["impl"], "warp")
        self.assertEqual(config["impl_resolution"], "auto-upstream-default")
        self.assertEqual(config["seed"], 9)
        self.assertEqual(config["timesteps"], 4096)
        self.assertEqual(config["num_envs"], 16)
        self.assertEqual(config["batch_size"], 16)
        self.assertEqual(config["reward_scaling"], 7.0)
        self.assertEqual(config["ppo_config_source"], "upstream:test")
        self.assertEqual(
            config["network_factory"]["policy_hidden_layer_sizes"],
            (512, 256, 128),
        )

    def test_custom_env_auto_uses_jax_and_generic_smoke_network(self) -> None:
        config = resolve_training_config(
            "envs/reach.py",
            preset_name="smoke",
            impl="auto",
            seed=3,
        )

        self.assertEqual(config["impl"], "jax")
        self.assertEqual(config["seed"], 3)
        self.assertEqual(config["ppo_config_source"], "simrig-generic-custom-env")
        self.assertEqual(config["network_factory"]["policy_hidden_layer_sizes"], (64, 64))

    def test_auto_falls_back_to_jax_when_upstream_warp_has_no_gpu(self) -> None:
        registry = MagicMock()
        registry.get_default_config.return_value = {"impl": "warp"}
        with (
            patch("simrig.playground_backend._import_registry", return_value=registry),
            patch("simrig.playground_backend._warp_available", return_value=False),
            patch(
                "simrig.playground_backend._upstream_ppo_config",
                return_value=({"timesteps": 10}, "upstream:test"),
            ),
        ):
            config = resolve_training_config(
                "RobotTask",
                preset_name="cloud",
                impl="auto",
            )

        self.assertEqual(config["impl"], "jax")
        self.assertEqual(config["impl_resolution"], "auto-fallback-no-gpu")

    def test_train_passes_randomizer_seed_and_resolved_ppo_config(self) -> None:
        env = MagicMock()
        env.xml_path = None
        randomizer = MagicMock()
        model_io = MagicMock()
        ppo_networks = MagicMock()
        ppo = MagicMock()
        ppo.train.return_value = (None, "params", {"eval/episode_reward": 1.0})
        wrapper = MagicMock()
        config = {
            "timesteps": 32,
            "num_envs": 4,
            "batch_size": 4,
            "num_evals": 1,
            "num_eval_envs": 1,
            "episode_length": 8,
            "unroll_length": 2,
            "num_minibatches": 1,
            "num_updates_per_batch": 1,
            "discounting": 0.97,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "normalize_observations": True,
            "action_repeat": 1,
            "reward_scaling": 1.0,
            "num_resets_per_eval": 0,
            "seed": 7,
            "impl": "warp",
            "impl_requested": "auto",
            "network_factory": {"policy_hidden_layer_sizes": (32, 32)},
            "ppo_config_source": "upstream:test",
        }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "simrig.playground_backend._import_training_deps",
                return_value=(
                    MagicMock(),
                    MagicMock(),
                    model_io,
                    MagicMock(),
                    ppo_networks,
                    ppo,
                    wrapper,
                ),
            ),
            patch("simrig.playground_backend.resolve_training_config", return_value=config),
            patch("simrig.playground_backend.load_env", return_value=env) as load_env,
            patch("simrig.playground_backend._resolve_domain_randomizer", return_value=randomizer),
            patch("simrig.playground_backend.runtime_manifest", return_value={"python": "test"}),
            patch("simrig.playground_backend.training_provenance", return_value={"test": True}),
        ):
            run = train_ppo(
                "RobotTask",
                output=Path(tmp) / "run",
                impl="auto",
                seed=7,
                overrides={"timesteps": 32},
            )

        self.assertEqual(load_env.call_count, 2)
        self.assertEqual(load_env.call_args.kwargs["config_overrides"], {"impl": "warp"})
        self.assertIs(ppo.train.call_args.kwargs["randomization_fn"], randomizer)
        self.assertEqual(ppo.train.call_args.kwargs["seed"], 7)
        self.assertEqual(ppo.train.call_args.kwargs["reward_scaling"], 1.0)
        self.assertIn("--impl", run.command)
        self.assertIn("warp", run.command)
        self.assertIn("--timesteps", run.command)
        self.assertIn("32", run.command)
        self.assertTrue(run.config["domain_randomization"])
        self.assertEqual(run.config["provenance"], {"test": True})

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
