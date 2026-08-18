"""Small rendered-pixel PPO reference for SimRig.

This environment uses MuJoCo's MJX renderer, so runtime validation and
training require a CUDA GPU and ``impl=warp``. Static metadata validation works
on CPU-only machines.

Try:
  simrig validate-env examples/vision_cartpole.py --vision
  simrig validate-env examples/vision_cartpole.py --runtime --vision
  simrig smoke examples/vision_cartpole.py --steps 5
  simrig train examples/vision_cartpole.py --preset smoke --output runs/vision-cartpole
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import cartpole
from mujoco_playground.config import dm_control_suite_params


ENV_NAME = "vision_cartpole"
MODEL_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cartpole.xml"


def default_config() -> config_dict.ConfigDict:
    """Return a one-world config that is cheap to construct for validation."""
    config = cartpole.default_config()
    config.impl = "warp"
    config.vision = True
    config.vision_config.nworld = 1
    return config


def network_spec() -> dict[str, Any]:
    """Select Brax's vision CNN and its actor/critic state inputs."""
    config = dm_control_suite_params.brax_vision_ppo_config("CartpoleBalance")
    factory = config.network_factory.to_dict()
    factory.update(
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    return {"type": "vision_cnn", "factory": factory}


def vision_spec() -> dict[str, Any]:
    """Describe the renderer contract checked by ``validate-env --vision``."""
    return {
        "pixel_keys": ["pixels/view_0"],
        "camera_names": ["fixed"],
        "modalities": ["grayscale"],
        "resolution": [64, 64],
        "frame_stack": 3,
        "channels_per_frame": 1,
        "value_range": [-0.5, 0.5],
        "requires_impl": "warp",
        "nworld_config_key": "vision_config.nworld",
    }


def training_config() -> dict[str, Any]:
    """Reuse Playground's tuned Cartpole vision PPO hyperparameters."""
    config = dm_control_suite_params.brax_vision_ppo_config("CartpoleBalance")
    result = config.to_dict()
    result["vision"] = True
    result["augment_pixels"] = True
    # The upstream 250-step horizon is enough for a training smoke test but
    # too short to demonstrate sustained visual balancing in a preview.
    result["episode_length"] = 1_000
    return result


def make_env(config_overrides: dict[str, Any] | None = None) -> "CustomEnv":
    return CustomEnv(config=default_config(), config_overrides=config_overrides)


class CustomEnv(cartpole.Balance):
    """Cartpole balance with pixels for the actor and state for the critic."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        # SECTION: model loading
        super().__init__(
            swing_up=False,
            sparse=False,
            config=config or default_config(),
            config_overrides=config_overrides,
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # SECTION: reset
        state = super().reset(rng)
        last_action = jp.zeros((self.action_size,))
        info = dict(state.info)
        info["last_action"] = last_action
        obs = self._vision_observation(state.data, state.obs, last_action)
        return state.replace(obs=obs, info=info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # SECTION: action mapping
        # Parent maps the normalized action directly to the cart actuator.
        next_state = super().step(state, action)
        # SECTION: rewards
        # Parent computes its dense vision reward before returning next_state.
        # SECTION: termination
        # Parent terminates on invalid state, rail escape, or a fallen pole.
        info = dict(next_state.info)
        info["last_action"] = action
        obs = self._vision_observation(next_state.data, next_state.obs, action)
        return next_state.replace(obs=obs, info=info)

    def _vision_observation(
        self,
        data: Any,
        pixels: dict[str, jax.Array],
        last_action: jax.Array,
    ) -> dict[str, jax.Array]:
        # SECTION: observations
        privileged = jp.concatenate([last_action, self._get_obs(data, {})])
        pixel_obs = pixels["pixels/view_0"]
        # An unbatched Warp renderer exposes its one internal world as a
        # leading singleton axis.  Remove only that axis so eval/inference sees
        # logical HWC pixels; vectorized training keeps its real batch axis.
        if pixel_obs.ndim == 4 and pixel_obs.shape[0] == 1:
            pixel_obs = pixel_obs[0]
        return {
            "pixels/view_0": pixel_obs,
            "state": last_action,
            "privileged_state": privileged,
        }

    @property
    def observation_size(self) -> dict[str, tuple[int, ...]]:
        base_size = 1 + 2 * (int(self.mjx_model.nq) - 1) + int(self.mjx_model.nv)
        return {
            "pixels/view_0": (64, 64, 3),
            "state": (self.action_size,),
            "privileged_state": (self.action_size + base_size,),
        }

    @property
    def action_size(self) -> int:
        return int(self.mjx_model.nu)
