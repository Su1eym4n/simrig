"""Shared, checked policy execution for evaluation and visualization.

Imports of the training stack are lazy, so inspection and report commands still
work without Playground. Task semantics and physical outcome measurements belong
in task-owned evaluators, not this runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any, Callable
import warnings

from simrig.presets import (
    checkpoint_config,
    normalize_checkpoint_path,
    resolve_network_factory,
    resolve_network_type,
)
from simrig.runtime import verify_checkpoint_runtime


class InvalidRolloutState(ValueError):
    """Non-finite, malformed, or out-of-range simulator/policy output."""


def validate_state(state: Any, observation_size: Any = None) -> None:
    """Check numerical state without interpreting task reward or success."""
    import numpy as np

    def check(value: Any, name: str, size: Any = None) -> None:
        if isinstance(value, Mapping):
            if isinstance(size, Mapping) and set(value) != set(size):
                raise InvalidRolloutState(f"{name} keys differ from observation_size")
            for key, item in value.items():
                check(item, f"{name}.{key}", size.get(key) if isinstance(size, Mapping) else None)
            return
        if value is None:
            raise InvalidRolloutState(f"{name} is missing")
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise InvalidRolloutState(f"{name} must contain finite numeric values")
        if size is not None:
            shape = (int(size),) if isinstance(size, int) else tuple(size)
            if array.shape != shape:
                raise InvalidRolloutState(f"{name} shape {array.shape} differs from {shape}")

    check(getattr(state, "obs", None), "observation", observation_size)
    for name in ("reward", "done"):
        value = getattr(state, name, None)
        check(value, name)
        if np.asarray(value).shape != ():
            raise InvalidRolloutState(f"{name} must be scalar")
    if float(state.done) not in (0.0, 1.0):
        raise InvalidRolloutState("done must be 0 or 1")
    data = getattr(state, "data", None)
    if data is not None:
        for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
            if hasattr(data, name):
                check(getattr(data, name), f"data.{name}")


def validate_action(action: Any, action_size: int) -> None:
    import numpy as np

    array = np.asarray(action)
    if array.shape != (action_size,) or array.dtype.kind not in "iuf":
        raise InvalidRolloutState(f"action must be a numeric vector of shape ({action_size},)")
    if not np.isfinite(array).all() or np.any(np.abs(array) > 1.0):
        raise InvalidRolloutState("action must be finite and normalized to [-1, 1]")


class PolicyRuntime:
    """One environment and compiled policy; reuse across sequential seeded trials.

    ``checkpoint=None`` is reserved for explicitly supplied baseline controllers.
    Such controllers receive (state, rng); learned policies receive only the
    environment's declared observations. Instances are not thread-safe.
    """

    def __init__(
        self,
        checkpoint: Path | str | None,
        *,
        env_name: str,
        backend: str = "mujoco-playground",
        small_network: bool | None = None,
        allow_runtime_mismatch: bool = False,
        controller_factory: Callable[[Any], Callable] | None = None,
    ) -> None:
        from simrig.playground_backend import (
            _checkpoint_env_overrides, _ensure_checkpoint_vision_runtime,
            _import_training_deps, _validate_backend, load_env,
        )
        from simrig.networks import make_network_factory

        _validate_backend(backend)
        self.env_name = env_name
        self._allow_runtime_mismatch = bool(allow_runtime_mismatch)
        self.checkpoint: Path | None = None
        self.runtime_compatibility: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        if checkpoint is not None:
            path = normalize_checkpoint_path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            self.checkpoint = path
            self.config = checkpoint_config(path) or {}
            if self.config.get("action_repeat", 1) != 1:
                raise ValueError(
                    "PolicyRuntime currently requires action_repeat=1; replaying this "
                    "checkpoint at every environment tick would change its control rate"
                )
            self.runtime_compatibility = verify_checkpoint_runtime(
                path, allow_mismatch=allow_runtime_mismatch,
            )
            _ensure_checkpoint_vision_runtime(path)
        elif controller_factory is None:
            raise ValueError("A checkpoint or explicit controller_factory is required")

        self.jax, self.jp, brax_model, stats, networks, *_ = _import_training_deps()
        self.env = load_env(
            env_name,
            config_overrides=_checkpoint_env_overrides(self.checkpoint)
            if self.checkpoint is not None else None,
        )
        self.reset_fn = self.jax.jit(self.env.reset)
        self.step_fn = self.jax.jit(self.env.step)
        self.controller = None
        if self.checkpoint is None:
            self.controller = controller_factory(self.env)
            if not callable(self.controller):
                raise TypeError("controller_factory must return a callable(state, rng)")
        else:
            self._verify_environment()
            factory = make_network_factory(
                resolve_network_type(self.checkpoint),
                resolve_network_factory(self.checkpoint, small_network=small_network),
                ppo_networks=networks,
            )
            # Vision and other unnormalized policies must retain their training
            # preprocessor; applying running statistics unconditionally changes it.
            normalize = stats.normalize if self.config.get("normalize_observations", True) else (lambda x, _: x)
            network = factory(
                self.env.observation_size, self.env.action_size,
                preprocess_observations_fn=normalize,
            )
            if self.checkpoint.is_dir():
                from brax.training.agents.ppo import checkpoint as ppo_checkpoint

                params = ppo_checkpoint.load(str(self.checkpoint))
            else:
                params = brax_model.load_params(str(self.checkpoint))
            self.policy = self.jax.jit(networks.make_inference_fn(network)(params, deterministic=True))

    def _artifact_mismatch(self, message: str) -> None:
        if self._allow_runtime_mismatch:
            warnings.warn(
                f"{message}; --allow-runtime-mismatch makes this rollout qualitative only.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        raise ValueError(message)

    def _verify_environment(self) -> None:
        recorded = self.config.get("env_ref")
        source = (self.config.get("provenance") or {}).get("source") or {}
        if recorded:
            if str(recorded).endswith(".py"):
                active = Path(self.env_name).expanduser()
                expected_hash = source.get("env_module_sha256")
                if expected_hash:
                    if not active.is_file() or hashlib.sha256(active.read_bytes()).hexdigest() != expected_hash:
                        self._artifact_mismatch(
                            "Environment source differs from the checkpoint's recorded module"
                        )
                elif active.resolve() != Path(str(recorded)).expanduser().resolve():
                    self._artifact_mismatch(
                        "Environment differs from the checkpoint's recorded environment"
                    )
            elif self.env_name != recorded:
                self._artifact_mismatch(
                    "Environment differs from the checkpoint's recorded environment"
                )
        expected_model = source.get("model_xml_sha256")
        if expected_model:
            model_path = Path(str(getattr(self.env, "xml_path", ""))).expanduser()
            if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_model:
                self._artifact_mismatch(
                    "Model XML differs from the checkpoint's recorded model"
                )

    def reset(self, seed: int) -> tuple[Any, Any]:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        rng = self.jax.random.PRNGKey(seed)
        return self.reset_key(rng), rng

    def reset_key(self, rng: Any) -> Any:
        state = self.reset_fn(rng)
        validate_state(state, self.env.observation_size)
        return state

    def apply_command(self, state: Any, command: Any | None) -> Any:
        if command is not None:
            from simrig.playground_backend import _apply_command

            state, applied = _apply_command(self.env, state, self.jp.asarray(command))
            if not applied:
                raise ValueError(f"Environment {self.env_name} does not expose command-like state")
            validate_state(state, self.env.observation_size)
        return state

    def advance(self, state: Any, rng: Any, *, command: Any | None = None) -> tuple[Any, Any, Any]:
        state = self.apply_command(state, command)
        validate_state(state, self.env.observation_size)
        if bool(state.done):
            raise InvalidRolloutState("Cannot advance a terminated episode; reset first")
        rng, action_rng = self.jax.random.split(rng)
        if self.controller is not None:
            action = self.controller(state, action_rng)
        else:
            action, _ = self.policy(state.obs, action_rng)
        validate_action(action, int(self.env.action_size))
        next_state = self.step_fn(state, self.jp.asarray(action))
        validate_state(next_state, self.env.observation_size)
        return next_state, rng, action

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
