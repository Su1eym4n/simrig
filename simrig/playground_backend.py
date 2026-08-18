"""MuJoCo Playground environment backend."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from simrig.core import (
    BackendInfo,
    EnvInspectionReport,
    RunConfig,
    SmokeResult,
    TrainabilityStatus,
)
from simrig.custom_env import (
    is_env_module_path,
    load_custom_env,
    load_custom_env_metadata,
    resolve_env_label,
)
from simrig.io import default_run_dir, save_json
from simrig.networks import (
    MLP_NETWORK,
    is_vision_network,
    make_network_factory,
    normalize_network_spec,
)
from simrig.paths import find_menagerie
from simrig.presets import (
    apply_preset_scale,
    checkpoint_config,
    legacy_network_factory,
    preset,
    resolve_network_factory,
    resolve_network_type,
)
from simrig.runtime import runtime_manifest, training_provenance, verify_checkpoint_runtime


_PPO_CONFIG_KEYS = (
    "num_eval_envs",
    "num_evals",
    "episode_length",
    "normalize_observations",
    "action_repeat",
    "unroll_length",
    "num_minibatches",
    "num_updates_per_batch",
    "discounting",
    "learning_rate",
    "entropy_cost",
    "num_envs",
    "batch_size",
    "reward_scaling",
    "num_resets_per_eval",
    "max_grad_norm",
    "clipping_epsilon",
    "vision",
    "augment_pixels",
)


def _import_registry():
    try:
        from mujoco_playground._src import registry  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Playground is not installed. Install SimRig with the playground "
            "extra, or install the `playground` package used by MuJoCo Playground."
        ) from exc
    return registry


def _import_training_deps():
    try:
        import jax  # type: ignore
        import jax.numpy as jp  # type: ignore
        _patch_jax_brax_compat(jax)
        from brax.io import model as brax_model  # type: ignore
        from brax.training.acme import running_statistics  # type: ignore
        from brax.training.agents.ppo import networks as ppo_networks  # type: ignore
        from brax.training.agents.ppo import train as ppo  # type: ignore
        from mujoco_playground import wrapper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are not installed. Install JAX, Brax, and "
            "MuJoCo Playground before running train/eval commands."
        ) from exc
    return jax, jp, brax_model, running_statistics, ppo_networks, ppo, wrapper


def resolve_training_config(
    env_name: str,
    *,
    preset_name: str,
    impl: str = "auto",
    seed: int = 0,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a complete, serializable PPO configuration for one run."""
    if impl not in {"auto", "jax", "warp"}:
        raise ValueError("Implementation must be one of: auto, jax, warp")
    if seed < 0:
        raise ValueError("Training seed must be non-negative.")

    if is_env_module_path(env_name):
        try:
            metadata = load_custom_env_metadata(env_name)
        except FileNotFoundError:
            metadata = {
                "default_config": {},
                "network_spec": {"type": MLP_NETWORK, "factory": {}},
                "vision_spec": {},
                "training_config": {},
            }
        default_impl = str(metadata["default_config"].get("impl", "jax"))
        resolved_impl = default_impl if impl == "auto" else impl
        vision_spec = _plain_dict(metadata["vision_spec"])
        required_impl = vision_spec.get("requires_impl")
        if required_impl and resolved_impl != required_impl:
            raise RuntimeError(
                f"This vision environment requires impl={required_impl}; "
                f"resolved impl={resolved_impl}."
            )
        if resolved_impl == "warp" and not _warp_available():
            raise RuntimeError(
                "MuJoCo Warp training requires a JAX-visible GPU. Standard "
                "MuJoCo camera checks may still run locally, but vectorized "
                "vision PPO needs CUDA/Warp."
            )
        impl_resolution = "auto-custom-env-default" if impl == "auto" else "explicit"
        custom_training = _plain_dict(metadata["training_config"])
        if "num_timesteps" in custom_training:
            custom_training["timesteps"] = custom_training.pop("num_timesteps")
        if custom_training:
            default_network = custom_training.pop("network_factory", {})
            config = apply_preset_scale(preset_name, custom_training)
            source = "custom-env:training_config"
        else:
            config = preset(preset_name)
            default_network = legacy_network_factory(bool(config.pop("small_network")))
            source = "simrig-generic-custom-env"
        network_spec = normalize_network_spec(
            _plain_dict(metadata["network_spec"]),
            default_factory=default_network,
        )
        network = network_spec["factory"]
        network_type = network_spec["type"]
    else:
        registry = _import_registry()
        resolved_impl, impl_resolution = _resolve_impl(registry, env_name, impl)
        upstream, source = _upstream_ppo_config(env_name, resolved_impl)
        network = upstream.pop("network_factory", {})
        config = apply_preset_scale(preset_name, upstream)
        network_type = MLP_NETWORK
        vision_spec = {}

    config.update(overrides or {})
    vision_network = is_vision_network(network_type)
    config.setdefault("normalize_observations", not vision_network)
    config.setdefault("action_repeat", 1)
    config.setdefault("reward_scaling", 1.0)
    config.setdefault("num_resets_per_eval", 0)
    # Brax's vision wrapper must agree with the selected network implementation.
    config["vision"] = vision_network
    config.setdefault("augment_pixels", vision_network)
    config.update(
        {
            "seed": seed,
            "impl": resolved_impl,
            "impl_requested": impl,
            "impl_resolution": impl_resolution,
            "network_factory": network,
            "network_type": network_type,
            "vision_spec": vision_spec,
            "ppo_config_source": source,
        }
    )
    return config


def _resolve_impl(registry: Any, env_name: str, requested: str) -> tuple[str, str]:
    if requested != "auto":
        if requested == "warp" and not _warp_available():
            raise RuntimeError("MuJoCo Warp training requires a JAX-visible GPU.")
        return requested, "explicit"
    try:
        config = registry.get_default_config(env_name)
        resolved = config.get("impl", "jax")
    except Exception:
        resolved = "jax"
    if str(resolved) == "warp" and not _warp_available():
        return "jax", "auto-fallback-no-gpu"
    return str(resolved), "auto-upstream-default"


def _warp_available() -> bool:
    try:
        import jax  # type: ignore
    except ImportError:
        return False
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def _upstream_ppo_config(env_name: str, impl: str) -> tuple[dict[str, Any], str]:
    try:
        from mujoco_playground._src import dm_control_suite, locomotion, manipulation  # type: ignore
        from mujoco_playground.config import (  # type: ignore
            dm_control_suite_params,
            locomotion_params,
            manipulation_params,
        )
    except ImportError as exc:
        raise RuntimeError("MuJoCo Playground PPO configuration modules are unavailable.") from exc

    if env_name in locomotion.ALL_ENVS:
        module = locomotion_params
        family = "locomotion"
    elif env_name in manipulation.ALL_ENVS:
        module = manipulation_params
        family = "manipulation"
    elif env_name in dm_control_suite.ALL_ENVS:
        module = dm_control_suite_params
        family = "dm-control-suite"
    else:
        raise ValueError(f"Unknown MuJoCo Playground environment: {env_name}")

    raw = _plain_dict(module.brax_ppo_config(env_name, impl=impl))
    if "num_timesteps" in raw:
        raw["timesteps"] = raw.pop("num_timesteps")
    return raw, f"mujoco-playground:{family}:brax_ppo_config"


def _plain_dict(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain_dict(to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_dict(item) for item in value)
    if isinstance(value, list):
        return [_plain_dict(item) for item in value]
    return value


def _patch_jax_brax_compat(jax: Any) -> None:
    """Restore APIs Brax still calls that newer JAX hid behind AttributeError.

    JAX 0.10+ makes ``jax.device_put_replicated`` raise, while Brax 0.14 PPO
    still calls it. Re-bind the private implementation when missing.
    """
    if "device_put_replicated" not in jax.__dict__:
        try:
            from jax._src.api import device_put_replicated as _device_put_replicated  # type: ignore
        except ImportError:
            pass
        else:
            jax.device_put_replicated = _device_put_replicated  # type: ignore[attr-defined]

    if "device_put_sharded" not in jax.__dict__:
        try:
            from jax._src.api import device_put_sharded as _device_put_sharded  # type: ignore
        except ImportError:
            pass
        else:
            jax.device_put_sharded = _device_put_sharded  # type: ignore[attr-defined]


def backend_info() -> BackendInfo:
    try:
        registry = _import_registry()
    except RuntimeError as exc:
        return BackendInfo(name="mujoco-playground", available=False, detail=str(exc))
    return BackendInfo(
        name="mujoco-playground",
        available=True,
        detail=f"{len(registry.ALL_ENVS)} registered envs",
    )


def list_envs(backend: str = "mujoco-playground") -> list[dict[str, Any]]:
    """List MuJoCo Playground environments."""
    _validate_backend(backend)
    registry = _import_registry()
    return [{"name": name, "backend": backend} for name in sorted(registry.ALL_ENVS)]


def load_env(env_name: str, config_overrides: dict[str, Any] | None = None):
    """Load a Playground registry env or a custom ``*.py`` env module."""
    if is_env_module_path(env_name):
        return load_custom_env(env_name, config_overrides=config_overrides)

    _configure_menagerie()
    registry = _import_registry()
    config_overrides = _default_overrides(registry, env_name, config_overrides)
    return registry.load(env_name, config_overrides=config_overrides)


def inspect_env(env_name: str, *, backend: str = "mujoco-playground") -> EnvInspectionReport:
    """Load a Playground or custom env and report trainability metadata."""
    _validate_backend(backend)
    label = resolve_env_label(env_name)

    if is_env_module_path(env_name):
        return _inspect_custom_env(env_name, label=label, backend=backend)

    registry = _import_registry()
    if env_name not in registry.ALL_ENVS:
        return EnvInspectionReport(
            name=label,
            backend=backend,
            status=TrainabilityStatus.FAILED,
            available=False,
            loaded=False,
            errors=[f"Unknown environment: {env_name}"],
            notes=[
                f"Available environments: {', '.join(sorted(registry.ALL_ENVS))}",
                "Or pass a custom env module path ending in .py",
            ],
        )

    try:
        env = load_env(env_name)
    except Exception as exc:
        return EnvInspectionReport(
            name=label,
            backend=backend,
            status=TrainabilityStatus.FAILED,
            available=True,
            loaded=False,
            errors=[str(exc)],
        )

    mj_model = getattr(env, "mj_model", None)
    xml_path = getattr(env, "xml_path", None)
    try:
        randomizer = registry.get_domain_randomizer(env_name)
    except Exception:
        randomizer = None
    return EnvInspectionReport(
        name=label,
        backend=backend,
        status=TrainabilityStatus.TRAINABLE_EXISTING_ENV,
        available=True,
        loaded=True,
        observation_size=getattr(env, "observation_size", None),
        action_size=int(getattr(env, "action_size", 0) or 0),
        xml_path=str(xml_path) if xml_path else None,
        model_bodies=int(mj_model.nbody) if mj_model is not None else None,
        model_actuators=int(mj_model.nu) if mj_model is not None else None,
        has_domain_randomizer=randomizer is not None,
        notes=[
            "Existing MuJoCo Playground envs are trainable because they define "
            "reset, step, reward, observations, and termination."
        ],
    )


def _inspect_custom_env(env_name: str, *, label: str, backend: str) -> EnvInspectionReport:
    try:
        metadata = load_custom_env_metadata(env_name)
    except Exception:
        metadata = {
            "network_spec": {"type": MLP_NETWORK},
            "vision_spec": {},
        }
    network_type = str(metadata["network_spec"]["type"])
    vision_spec = _plain_dict(metadata["vision_spec"])
    if (
        is_vision_network(network_type)
        and vision_spec.get("requires_impl") == "warp"
        and not _warp_available()
    ):
        return EnvInspectionReport(
            name=label,
            backend=backend,
            status=TrainabilityStatus.INSPECTABLE,
            available=True,
            loaded=False,
            network_type=network_type,
            vision=vision_spec,
            warnings=[
                "Rendered vision runtime requires a JAX-visible CUDA GPU and "
                "MuJoCo Warp. Metadata is available, but the env was not constructed."
            ],
            notes=[f"Custom env module: {env_name}"],
        )
    try:
        env = load_env(env_name)
    except Exception as exc:
        return EnvInspectionReport(
            name=label,
            backend=backend,
            status=TrainabilityStatus.NEEDS_CUSTOM_ENV,
            available=True,
            loaded=False,
            errors=[str(exc)],
            notes=[
                "Custom env module failed to load. Finish implementation, then "
                "run `simrig validate-env PATH --runtime` and `simrig smoke PATH`."
            ],
        )

    mj_model = getattr(env, "mj_model", None)
    xml_path = getattr(env, "xml_path", None)
    warnings = _obs_key_warnings(getattr(env, "observation_size", None))
    return EnvInspectionReport(
        name=label,
        backend=backend,
        status=TrainabilityStatus.TRAINABLE_EXISTING_ENV,
        available=True,
        loaded=True,
        observation_size=getattr(env, "observation_size", None),
        action_size=int(getattr(env, "action_size", 0) or 0),
        xml_path=str(xml_path) if xml_path else None,
        model_bodies=int(mj_model.nbody) if mj_model is not None else None,
        model_actuators=int(mj_model.nu) if mj_model is not None else None,
        has_domain_randomizer=False,
        network_type=network_type,
        vision=vision_spec,
        warnings=warnings,
        notes=[
            f"Custom env module: {env_name}",
            "Run `simrig smoke` before `simrig train`.",
            "SimRig PPO defaults expect observation keys `state` and `privileged_state`.",
        ],
    )


def _obs_key_warnings(observation_size: Any) -> list[str]:
    warnings: list[str] = []
    if observation_size is None:
        warnings.append("observation_size is missing.")
        return warnings
    if isinstance(observation_size, dict):
        if "state" not in observation_size:
            warnings.append("observation_size is missing key `state` (policy obs).")
        if "privileged_state" not in observation_size:
            warnings.append(
                "observation_size is missing key `privileged_state` (value obs). "
                "SimRig PPO defaults expect it."
            )
    else:
        warnings.append(
            "observation_size is flat; SimRig PPO defaults expect dict keys "
            "`state` and `privileged_state`."
        )
    return warnings


def smoke_env(
    env_name: str,
    *,
    steps: int = 10,
    backend: str = "mujoco-playground",
    seed: int = 0,
) -> SmokeResult:
    """Run a short reset/zero-action step test."""
    _validate_backend(backend)
    label = resolve_env_label(env_name)
    jax, jp, *_ = _import_training_deps()
    try:
        if is_env_module_path(env_name):
            metadata = load_custom_env_metadata(env_name)
            if (
                is_vision_network(str(metadata["network_spec"]["type"]))
                and metadata["vision_spec"].get("requires_impl") == "warp"
                and not _warp_available()
            ):
                raise RuntimeError(
                    "Rendered vision smoke requires a JAX-visible CUDA GPU and "
                    "MuJoCo Warp."
                )
        env = load_env(env_name)
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        state = reset(jax.random.PRNGKey(seed))
        completed = 0
        for completed in range(1, steps + 1):
            state = step(state, jp.zeros(env.action_size))
        return SmokeResult(
            env_name=label,
            backend=backend,
            steps_requested=steps,
            steps_completed=completed,
            passed=True,
            action_size=int(env.action_size),
            observation_size=getattr(env, "observation_size", None),
            final_reward=float(state.reward),
            final_done=bool(state.done),
        )
    except Exception as exc:
        return SmokeResult(
            env_name=label,
            backend=backend,
            steps_requested=steps,
            steps_completed=0,
            passed=False,
            errors=[str(exc)],
        )


def train_ppo(
    env_name: str,
    *,
    preset_name: str = "smoke",
    output: Path | str | None = None,
    backend: str = "mujoco-playground",
    overrides: dict[str, Any] | None = None,
    impl: str = "auto",
    seed: int = 0,
    domain_randomization: bool = True,
) -> RunConfig:
    """Train a Playground or custom env module with Brax PPO."""
    _validate_backend(backend)
    jax, jp, brax_model, _, ppo_networks, ppo, wrapper = _import_training_deps()
    del jp
    label = resolve_env_label(env_name)
    config = resolve_training_config(
        env_name,
        preset_name=preset_name,
        impl=impl,
        seed=seed,
        overrides=overrides,
    )
    output_dir = Path(output) if output is not None else default_run_dir(label, preset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Orbax requires absolute checkpoint paths.
    output_dir = output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = load_env(
        env_name,
        config_overrides=_training_env_overrides(config, evaluation=False),
    )
    eval_env = load_env(
        env_name,
        config_overrides=_training_env_overrides(config, evaluation=True),
    )
    network_factory = make_network_factory(
        config.get("network_type", MLP_NETWORK),
        config["network_factory"],
        ppo_networks=ppo_networks,
    )
    randomizer = _resolve_domain_randomizer(env_name, env) if domain_randomization else None
    randomizer_name = _callable_name(randomizer) if randomizer is not None else None
    config.update(
        {
            "domain_randomization_requested": domain_randomization,
            "domain_randomization": randomizer is not None,
            "domain_randomizer": randomizer_name,
            "env_ref": str(env_name),
            "runtime": runtime_manifest(),
            "provenance": training_provenance(env_name, env, jax=jax),
        }
    )

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        reward = float(metrics.get("eval/episode_reward", 0.0))
        length = float(metrics.get("eval/avg_episode_length", 0.0))
        print(f"steps={num_steps:,} eval_reward={reward:.3f} eval_length={length:.1f}")

    run_config = RunConfig(
        env_name=label,
        backend=backend,
        preset=preset_name,
        output_dir=str(output_dir),
        config=config,
        command=[
            sys.executable,
            "-m",
            "simrig.cli",
            "train",
            str(env_name),
            "--preset",
            preset_name,
            "--impl",
            config["impl"],
            "--seed",
            str(seed),
            "--output",
            str(output_dir),
        ],
    )
    if not domain_randomization:
        run_config.command.append("--no-domain-randomization")
    for flag, key in (
        ("--timesteps", "timesteps"),
        ("--num-envs", "num_envs"),
        ("--batch-size", "batch_size"),
    ):
        if overrides is not None and key in overrides:
            run_config.command.extend([flag, str(overrides[key])])
    save_json(output_dir / "config.json", run_config)
    train_kwargs = {
        key: config[key]
        for key in _PPO_CONFIG_KEYS
        if key in config
    }
    _, params, metrics = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_timesteps=config["timesteps"],
        seed=seed,
        randomization_fn=randomizer,
        network_factory=network_factory,
        progress_fn=progress,
        save_checkpoint_path=str(checkpoint_dir),
        **train_kwargs,
    )
    brax_model.save_params(str(output_dir / "policy.params"), params)
    save_json(output_dir / "final_metrics.json", metrics)
    return run_config


def _resolve_domain_randomizer(env_name: str, env: Any) -> Any | None:
    if is_env_module_path(env_name):
        candidate = getattr(env, "domain_randomizer", None)
        return candidate if callable(candidate) else None
    try:
        return _import_registry().get_domain_randomizer(env_name)
    except Exception:
        return None


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    return f"{module}.{name}" if module else str(name)


def eval_policy(
    checkpoint: Path | str,
    *,
    env_name: str,
    steps: int = 500,
    backend: str = "mujoco-playground",
    small_network: bool | None = None,
    seed: int = 0,
    command: tuple[float, ...] | None = None,
    allow_runtime_mismatch: bool = False,
) -> dict[str, Any]:
    """Headless deterministic policy rollout."""
    _validate_backend(backend)
    runtime = verify_checkpoint_runtime(
        checkpoint,
        allow_mismatch=allow_runtime_mismatch,
    )
    _ensure_checkpoint_vision_runtime(checkpoint)
    jax, jp, brax_model, running_statistics, ppo_networks, *_ = _import_training_deps()
    env = load_env(env_name, config_overrides=_checkpoint_env_overrides(checkpoint))
    network_config = resolve_network_factory(checkpoint, small_network=small_network)
    network_factory = make_network_factory(
        resolve_network_type(checkpoint),
        network_config,
        ppo_networks=ppo_networks,
    )
    networks = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    params = brax_model.load_params(str(checkpoint))
    policy = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    command_applied = False
    if command is not None:
        state, command_applied = _apply_command(env, state, jp.asarray(command))
        if not command_applied:
            raise ValueError(
                f"Environment {resolve_env_label(env_name)} does not expose command-like state."
            )
    total_reward = 0.0
    completed = 0
    for completed in range(1, steps + 1):
        if command is not None:
            state, command_applied = _apply_command(env, state, jp.asarray(command))
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        state = step(state, action)
        total_reward += float(state.reward)
        if bool(state.done):
            break
    completed_requested_steps = completed == steps and not bool(state.done)
    return {
        "env_name": resolve_env_label(env_name),
        "backend": backend,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "command": list(command) if command is not None else None,
        "command_applied": command_applied,
        "steps_requested": steps,
        "steps_completed": completed,
        "total_reward": total_reward,
        "average_reward": total_reward / max(completed, 1),
        "terminated": bool(state.done),
        "completed_requested_steps": completed_requested_steps,
        "task_success": None,
        "task_success_reason": (
            "No task-specific success evaluator is configured; rollout completion and "
            "reward do not prove command tracking or task success."
        ),
        "runtime_compatibility": runtime,
    }


def demo_policy(
    checkpoint: Path | str,
    *,
    env_name: str,
    steps: int = 5000,
    backend: str = "mujoco-playground",
    small_network: bool | None = None,
    seed: int = 0,
    command: tuple[float, ...] | None = None,
    speed: float = 1.0,
    camera_distance: float | None = None,
    allow_runtime_mismatch: bool = False,
) -> dict[str, Any]:
    """Run a trained policy in a desktop MuJoCo viewer."""
    _validate_backend(backend)
    verify_checkpoint_runtime(checkpoint, allow_mismatch=allow_runtime_mismatch)
    _ensure_checkpoint_vision_runtime(checkpoint)
    jax, jp, brax_model, running_statistics, ppo_networks, *_ = _import_training_deps()
    try:
        import mujoco  # type: ignore
        from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Interactive demo requires MuJoCo and Gymnasium's MuJoCo viewer."
        ) from exc

    env = load_env(env_name, config_overrides=_checkpoint_env_overrides(checkpoint))
    network_config = resolve_network_factory(checkpoint, small_network=small_network)
    network_factory = make_network_factory(
        resolve_network_type(checkpoint),
        network_config,
        ppo_networks=ppo_networks,
    )
    networks = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    params = brax_model.load_params(str(checkpoint))
    policy = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    command_applied = False
    if command is not None:
        state, command_applied = _apply_command(env, state, jp.asarray(command))

    mj_data = mujoco.MjData(env.mj_model)
    viewer = WindowViewer(env.mj_model, mj_data, width=None, height=None, max_geom=1000)
    if camera_distance is not None:
        viewer.cam.distance = camera_distance

    total_reward = 0.0
    completed = 0
    try:
        for completed in range(1, steps + 1):
            if command is not None:
                state, command_applied = _apply_command(env, state, jp.asarray(command))
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(state.obs, action_rng)
            state = step(state, action)
            total_reward += float(state.reward)

            mj_data.qpos = state.data.qpos
            mj_data.qvel = state.data.qvel
            _copy_mocap_state(state.data, mj_data)
            mujoco.mj_forward(env.mj_model, mj_data)
            viewer.data = mj_data
            viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "SimRig demo\nEnv\nReward\nSteps",
                f"\n{resolve_env_label(env_name)}\n{float(state.reward):.4f}\n{completed}",
            )
            viewer.render()
            time.sleep(max(0.0, env.dt / max(speed, 1e-6) - 0.001))
            if viewer.window is None or bool(state.done):
                break
    finally:
        viewer.close()

    return {
        "env_name": resolve_env_label(env_name),
        "backend": backend,
        "checkpoint": str(checkpoint),
        "steps_requested": steps,
        "steps_completed": completed,
        "total_reward": total_reward,
        "average_reward": total_reward / max(completed, 1),
        "terminated": bool(state.done),
        "command": list(command) if command is not None else None,
        "command_applied": command_applied,
    }


def _validate_backend(backend: str) -> None:
    if backend != "mujoco-playground":
        raise ValueError(
            f"Unsupported backend for v0: {backend}. "
            "SimRig v0 supports only mujoco-playground training envs."
        )


def _checkpoint_env_overrides(checkpoint: Path | str) -> dict[str, Any] | None:
    config = checkpoint_config(checkpoint)
    if config is None:
        return None
    overrides: dict[str, Any] = {}
    impl = config.get("impl")
    if impl in {"jax", "warp"}:
        overrides["impl"] = impl
    vision_spec = config.get("vision_spec")
    if isinstance(vision_spec, Mapping):
        nworld_key = vision_spec.get("nworld_config_key")
        if nworld_key:
            overrides[str(nworld_key)] = 1
    return overrides or None


def _ensure_checkpoint_vision_runtime(checkpoint: Path | str) -> None:
    if resolve_network_type(checkpoint) == "vision_cnn" and not _warp_available():
        raise RuntimeError(
            "This vision checkpoint requires a JAX-visible CUDA GPU and MuJoCo Warp "
            "for environment rendering and policy rollout."
        )


def _training_env_overrides(
    config: Mapping[str, Any],
    *,
    evaluation: bool,
) -> dict[str, Any]:
    """Build environment overrides, including renderer batch cardinality."""
    overrides = {"impl": config["impl"]}
    vision_spec = config.get("vision_spec")
    if isinstance(vision_spec, Mapping):
        nworld_key = vision_spec.get("nworld_config_key")
        if nworld_key:
            count_key = "num_eval_envs" if evaluation else "num_envs"
            overrides[str(nworld_key)] = int(config[count_key])
    return overrides


def _copy_mocap_state(source_data: Any, target_data: Any) -> None:
    """Copy optional MJX mocap state into native MuJoCo render data."""
    for name in ("mocap_pos", "mocap_quat"):
        source = getattr(source_data, name, None)
        target = getattr(target_data, name, None)
        if source is not None and target is not None:
            target[:] = source


def _apply_command(env: Any, state: Any, command: Any) -> tuple[Any, bool]:
    if hasattr(env, "set_command"):
        return env.set_command(state, command), True
    info = dict(getattr(state, "info", {}))
    if "command" not in info:
        return state, False
    info["command"] = command
    for key in ("steps_until_next_cmd", "steps_until_next_command"):
        if key in info:
            try:
                import jax.numpy as jp  # type: ignore

                info[key] = jp.iinfo(jp.int32).max
            except Exception:
                pass
    obs = _rebuild_observation(env, state, info)
    if obs is None:
        return state.replace(info=info), True
    return state.replace(info=info, obs=obs), True


def _rebuild_observation(env: Any, state: Any, info: dict[str, Any]) -> Any | None:
    target = getattr(env, "unwrapped", env)
    get_obs = getattr(target, "_get_obs", None)
    if get_obs is None:
        get_obs = getattr(env, "_get_obs", None)
    if get_obs is None:
        return None
    try:
        return get_obs(state.data, info)
    except TypeError:
        # Some locomotion envs require contact flags as a third argument.
        try:
            import jax.numpy as jp  # type: ignore

            foot_sensor_ids = getattr(target, "_feet_floor_found_sensor", None)
            mj_model = getattr(target, "_mj_model", getattr(target, "mj_model", None))
            if foot_sensor_ids is None or mj_model is None:
                return None
            contact = jp.array(
                [
                    state.data.sensordata[mj_model.sensor_adr[sensor_id]] > 0
                    for sensor_id in foot_sensor_ids
                ]
            )
            return get_obs(state.data, info, contact)
        except Exception:
            return None


def _default_overrides(
    registry: Any,
    env_name: str,
    config_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    overrides = dict(config_overrides or {})
    if "impl" in overrides:
        return overrides
    try:
        config = registry.get_default_config(env_name)
    except Exception:
        return overrides or None
    if "impl" in config:
        overrides["impl"] = "jax"
    return overrides or None


def _configure_menagerie() -> None:
    """Point MuJoCo Playground at an existing local Menagerie when possible."""
    try:
        menagerie = find_menagerie()
    except FileNotFoundError:
        return
    os.environ.setdefault("MUJOCO_MENAGERIE_PATH", str(menagerie))
    try:
        from etils import epath  # type: ignore
        from mujoco_playground._src import mjx_env  # type: ignore

        mjx_env.MENAGERIE_PATH = epath.Path(menagerie)
    except Exception:
        # Loading may still work if the backend uses the environment variable.
        return
