from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from simrig.networks import make_network_factory, normalize_network_spec


class NetworkTests(unittest.TestCase):
    def test_normalize_network_spec_preserves_legacy_mlp_default(self) -> None:
        spec = normalize_network_spec(
            None,
            default_factory={"policy_hidden_layer_sizes": (64, 64)},
        )

        self.assertEqual(spec["type"], "mlp")
        self.assertEqual(spec["factory"]["policy_hidden_layer_sizes"], (64, 64))

    def test_normalize_network_spec_accepts_compact_vision_form(self) -> None:
        spec = normalize_network_spec(
            {
                "type": "vision_cnn",
                "policy_obs_key": "state",
                "factory": {"cnn_output_channels": (16, 32)},
            }
        )

        self.assertEqual(spec["type"], "vision_cnn")
        self.assertEqual(spec["factory"]["policy_obs_key"], "state")
        self.assertEqual(spec["factory"]["cnn_output_channels"], (16, 32))

    def test_make_network_factory_keeps_mlp_behavior(self) -> None:
        ppo_networks = MagicMock()
        factory = make_network_factory(
            "mlp",
            {"policy_hidden_layer_sizes": (32, 32)},
            ppo_networks=ppo_networks,
        )

        factory("obs", 2, preprocess_observations_fn="normalize")

        ppo_networks.make_ppo_networks.assert_called_once_with(
            "obs",
            2,
            preprocess_observations_fn="normalize",
            policy_hidden_layer_sizes=(32, 32),
        )

    def test_make_network_factory_selects_brax_vision_cnn_when_available(self) -> None:
        ppo_networks = MagicMock()
        try:
            factory = make_network_factory(
                "vision_cnn",
                {"cnn_output_channels": (16, 32)},
                ppo_networks=ppo_networks,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        self.assertEqual(
            factory.func.__module__, "brax.training.agents.ppo.networks_vision"
        )
        self.assertEqual(factory.keywords["cnn_output_channels"], (16, 32))

    def test_unknown_network_type_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported network type"):
            normalize_network_spec({"type": "transformer"})


if __name__ == "__main__":
    unittest.main()
