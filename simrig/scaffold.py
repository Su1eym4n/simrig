"""Editable starter files for custom environments."""

from __future__ import annotations

from pathlib import Path

from simrig.io import slugify

# Markers used by validate-env static checklist (keep in sync with template).
REQUIRED_SECTION_MARKERS = (
    "SECTION: model loading",
    "SECTION: reset",
    "SECTION: action mapping",
    "SECTION: observations",
    "SECTION: rewards",
    "SECTION: termination",
)

REQUIRED_CLASS_METHODS = (
    "__init__",
    "reset",
    "step",
)


def new_env(name: str, model: str | Path, *, template: str = "mjx", root: Path | str = "envs") -> Path:
    """Create an editable starter env module."""
    if template != "mjx":
        raise ValueError("SimRig v0 supports only the 'mjx' env template.")
    module_name = slugify(name).replace("-", "_").replace(".", "_")
    path = Path(root) / f"{module_name}.py"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing env template: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_mjx_template(name=name, model=str(model)), encoding="utf-8")
    return path


def _mjx_template(*, name: str, model: str) -> str:
    return f'''"""Editable SimRig MJX environment starter for {name}.

NOT TRAINABLE YET.
Fill the SECTION blocks below, remove this banner when reset/step work, then:

    simrig validate-env PATH --runtime
    simrig smoke PATH --steps 10
    simrig train PATH --preset smoke

Prefer subclassing mujoco_playground._src.mjx_env.MjxEnv when Playground is
installed. Return obs as dict keys `state` and `privileged_state` for SimRig PPO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


MODEL_PATH = Path({model!r}).expanduser()
ENV_NAME = {name!r}


def default_config() -> dict[str, Any]:
    return {{
        "episode_length": 1000,
        "action_scale": 1.0,
        "ctrl_dt": 0.02,
        "sim_dt": 0.002,
        "impl": "jax",
    }}


def make_env(config_overrides: dict[str, Any] | None = None) -> "CustomEnv":
    """Factory used by `simrig smoke/train/eval` for this module."""
    config = default_config()
    if config_overrides:
        config.update(config_overrides)
    return CustomEnv(config=config)


class CustomEnv:
    """Replace this starter with a real mjx_env.MjxEnv implementation.

    Required for SimRig smoke/train/eval:
    - reset(rng) -> state with .obs, .reward, .done, .data
    - step(state, action) -> state
    - observation_size (dict with `state` and `privileged_state` preferred)
    - action_size
    - mj_model / mjx_model (for demos and previews)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or default_config()
        # SECTION: model loading
        # Load MODEL_PATH into mj_model / mjx_model and cache actuator limits.
        raise NotImplementedError(
            "Define model loading, reset, step, rewards, observations, and termination."
        )

    def reset(self, rng: Any) -> Any:
        # SECTION: reset
        # Sample initial qpos/qvel (and any task state). Return a Brax/MJX state.
        raise NotImplementedError("Implement reset randomization.")

    def step(self, state: Any, action: Any) -> Any:
        # SECTION: action mapping
        # Scale/clip `action` into actuator controls.

        # SECTION: observations
        # Build policy obs (`state`) and privileged obs (`privileged_state`).

        # SECTION: rewards
        # Compute dense/sparse reward terms; do not invent them from the model name.

        # SECTION: termination
        # Set done / truncations from falls, limits, success, or time.

        raise NotImplementedError("Implement step, including action mapping, obs, reward, termination.")

    @property
    def observation_size(self) -> Any:
        raise NotImplementedError("Return dict observation sizes for state/privileged_state.")

    @property
    def action_size(self) -> int:
        raise NotImplementedError("Return the actuator/action dimension.")
'''
