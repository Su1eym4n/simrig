from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simrig.presets import apply_preset_scale, resolve_network_factory, resolve_small_network


class PresetTests(unittest.TestCase):
    def test_smoke_scale_preserves_tuned_values_and_network(self) -> None:
        upstream = {
            "timesteps": 200_000_000,
            "num_envs": 8192,
            "batch_size": 256,
            "reward_scaling": 3.0,
            "network_factory": {"policy_hidden_layer_sizes": [512, 256, 128]},
        }

        resolved = apply_preset_scale("smoke", upstream)

        self.assertEqual(resolved["timesteps"], 4096)
        self.assertEqual(resolved["num_envs"], 16)
        self.assertEqual(resolved["batch_size"], 16)
        self.assertEqual(resolved["reward_scaling"], 3.0)
        self.assertEqual(resolved["network_factory"], upstream["network_factory"])

    def test_resolve_network_factory_reads_exact_recorded_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint = run_dir / "policy.params"
            checkpoint.write_text("params\n", encoding="utf-8")
            network = {
                "policy_hidden_layer_sizes": [32, 32, 32, 32],
                "value_hidden_layer_sizes": [256, 256, 256],
                "policy_obs_key": "state",
                "value_obs_key": "state",
            }
            (run_dir / "config.json").write_text(
                json.dumps({"config": {"network_factory": network}}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_network_factory(checkpoint), network)

    def test_resolve_small_network_reads_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint = run_dir / "policy.params"
            checkpoint.write_text("params\n", encoding="utf-8")
            (run_dir / "config.json").write_text(
                json.dumps({"config": {"small_network": True}}),
                encoding="utf-8",
            )

            self.assertTrue(resolve_small_network(checkpoint))
            self.assertFalse(resolve_small_network(checkpoint, small_network=False))

    def test_resolve_small_network_defaults_to_large(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "policy.params"
            checkpoint.write_text("params\n", encoding="utf-8")

            self.assertFalse(resolve_small_network(checkpoint))


if __name__ == "__main__":
    unittest.main()
