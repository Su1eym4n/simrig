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
This file is intentionally incomplete. A raw MuJoCo model is not a training
task until you define reset logic, observations, rewards, termination, and
action mapping. SimRig v0.1 does not train custom env modules end-to-end —
use `simrig validate-env` for a static checklist only.
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
        "impl": "jax",
    }}


class CustomEnv:
    """Replace this starter with a real mjx_env.MjxEnv implementation.

    Required methods/properties for future SimRig/Brax training:
    - reset(rng)
    - step(state, action)
    - observation_size
    - action_size
    - mj_model
    - mjx_model

    Recommended observation keys:
    - state: policy observation
    - privileged_state: value-function observation
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
        # Sample initial qpos/qvel (and any task state). Return a Brax-style state.
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
        raise NotImplementedError("Return policy/privileged observation sizes.")

    @property
    def action_size(self) -> int:
        raise NotImplementedError("Return the actuator/action dimension.")
'''
