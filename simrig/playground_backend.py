"""MuJoCo Playground environment backend."""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path
from typing import Any

from simrig.core import BackendInfo, EnvInspectionReport, RunConfig, SmokeResult, TrainabilityStatus
from simrig.custom_env import is_env_module_path, load_custom_env, resolve_env_label
from simrig.io import default_run_dir, save_json
from simrig.paths import find_menagerie
from simrig.presets import hidden_sizes, preset, resolve_small_network


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
) -> RunConfig:
    """Train a Playground or custom env module with Brax PPO."""
    _validate_backend(backend)
    jax, jp, brax_model, _, ppo_networks, ppo, wrapper = _import_training_deps()
    del jax, jp
    label = resolve_env_label(env_name)
    config = preset(preset_name)
    config.update(overrides or {})
    output_dir = Path(output) if output is not None else default_run_dir(label, preset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Orbax requires absolute checkpoint paths.
    output_dir = output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = load_env(env_name, config_overrides={"impl": config.get("impl", "jax")})
    sizes = hidden_sizes(bool(config["small_network"]))
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=sizes,
        value_hidden_layer_sizes=sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
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
        config={**config, "env_ref": str(env_name)},
        command=[
            sys.executable,
            "-m",
            "simrig.cli",
            "train",
            str(env_name),
            "--preset",
            preset_name,
            "--output",
            str(output_dir),
        ],
    )
    save_json(output_dir / "config.json", run_config)
    _, params, metrics = ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_timesteps=config["timesteps"],
        num_evals=config["num_evals"],
        num_eval_envs=config["num_eval_envs"],
        episode_length=config["episode_length"],
        normalize_observations=True,
        action_repeat=1,
        unroll_length=config["unroll_length"],
        num_minibatches=config["num_minibatches"],
        num_updates_per_batch=config["num_updates_per_batch"],
        discounting=config["discounting"],
        learning_rate=config["learning_rate"],
        entropy_cost=config["entropy_cost"],
        num_envs=config["num_envs"],
        batch_size=config["batch_size"],
        network_factory=network_factory,
        progress_fn=progress,
        save_checkpoint_path=str(checkpoint_dir),
    )
    brax_model.save_params(str(output_dir / "policy.params"), params)
    save_json(output_dir / "final_metrics.json", metrics)
    return run_config


def eval_policy(
    checkpoint: Path | str,
    *,
    env_name: str,
    steps: int = 500,
    backend: str = "mujoco-playground",
    small_network: bool | None = None,
    seed: int = 0,
    command: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Headless deterministic policy rollout."""
    _validate_backend(backend)
    jax, jp, brax_model, running_statistics, ppo_networks, *_ = _import_training_deps()
    env = load_env(env_name)
    sizes = hidden_sizes(resolve_small_network(checkpoint, small_network=small_network))
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=sizes,
        value_hidden_layer_sizes=sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
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
) -> dict[str, Any]:
    """Run a trained policy in a desktop MuJoCo viewer."""
    _validate_backend(backend)
    jax, jp, brax_model, running_statistics, ppo_networks, *_ = _import_training_deps()
    try:
        import mujoco  # type: ignore
        from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Interactive demo requires MuJoCo and Gymnasium's MuJoCo viewer."
        ) from exc

    env = load_env(env_name)
    sizes = hidden_sizes(resolve_small_network(checkpoint, small_network=small_network))
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=sizes,
        value_hidden_layer_sizes=sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
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
