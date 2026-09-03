"""Training presets shared by CLI and library calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simrig.networks import MLP_NETWORK


PRESET_NAMES = ("smoke", "local", "large")
PRESET_ALIASES = {"cloud": "large"}

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "timesteps": 4096,
        "num_envs": 16,
        "num_eval_envs": 1,
        "num_evals": 2,
        "episode_length": 100,
        "batch_size": 16,
        "unroll_length": 10,
        "num_minibatches": 2,
        "num_updates_per_batch": 1,
        "discounting": 0.97,
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "small_network": True,
    },
    "local": {
        "timesteps": 1_000_000,
        "num_envs": 128,
        "num_eval_envs": 4,
        "num_evals": 10,
        "episode_length": 1000,
        "batch_size": 128,
        "unroll_length": 20,
        "num_minibatches": 32,
        "num_updates_per_batch": 4,
        "discounting": 0.97,
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "small_network": False,
    },
    "large": {
        "timesteps": 200_000_000,
        "num_envs": 8192,
        "num_eval_envs": 128,
        "num_evals": 20,
        "episode_length": 1000,
        "batch_size": 256,
        "unroll_length": 20,
        "num_minibatches": 32,
        "num_updates_per_batch": 4,
        "discounting": 0.97,
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "small_network": False,
    },
}


_SCALE_KEYS = (
    "timesteps",
    "num_envs",
    "num_eval_envs",
    "num_evals",
    "episode_length",
    "batch_size",
    "unroll_length",
    "num_minibatches",
    "num_updates_per_batch",
)


def canonical_preset(name: str) -> str:
    """Return the canonical preset name, accepting deprecated aliases.

    ``cloud`` remains a hidden alias for the large PPO scale so old run
    configs and scripts still load. It does not mean SSH or a cloud VM.
    Use ``simrig remote`` to train on another Linux GPU over SSH.
    """
    resolved = PRESET_ALIASES.get(name, name)
    if resolved not in PRESETS:
        choices = ", ".join(PRESET_NAMES)
        raise ValueError(f"Unknown preset: {name}. Choose one of: {choices}")
    return resolved


def preset(name: str) -> dict[str, Any]:
    """Return a mutable copy of a named preset."""
    return dict(PRESETS[canonical_preset(name)])


def apply_preset_scale(name: str, upstream: dict[str, Any]) -> dict[str, Any]:
    """Bound an upstream PPO config for SimRig's smoke/local run sizes.

    The large preset (formerly ``cloud``) preserves the upstream task-specific
    configuration. Smoke and local keep tuned optimizer, reward, and network
    settings while limiting the expensive rollout/update dimensions.
    """
    resolved = dict(upstream)
    if canonical_preset(name) == "large":
        return resolved

    limits = preset(name)
    for key in _SCALE_KEYS:
        if key not in limits:
            continue
        limit = limits[key]
        current = resolved.get(key)
        resolved[key] = min(current, limit) if current is not None else limit
    return resolved


def hidden_sizes(small_network: bool) -> tuple[int, ...]:
    return (64, 64) if small_network else (512, 256, 128)


def legacy_network_factory(small_network: bool) -> dict[str, Any]:
    """Return the network layout used by SimRig checkpoints before v0.4."""
    sizes = hidden_sizes(small_network)
    return {
        "policy_hidden_layer_sizes": sizes,
        "value_hidden_layer_sizes": sizes,
        "policy_obs_key": "state",
        "value_obs_key": "privileged_state",
    }


def resolve_network_factory(
    checkpoint: Path | str,
    *,
    small_network: bool | None = None,
) -> dict[str, Any]:
    """Resolve exact PPO network kwargs recorded beside a checkpoint.

    ``--small-network`` remains an explicit compatibility override for legacy
    checkpoints. New runs persist their complete upstream/custom network
    factory settings in ``config.json``.
    """
    if small_network is not None:
        return legacy_network_factory(small_network)

    config = checkpoint_config(checkpoint)
    if config is not None and "network_factory" in config:
        network = config["network_factory"]
        if isinstance(network, dict):
            return dict(network)

    legacy_small = bool(config.get("small_network")) if config is not None else False
    return legacy_network_factory(legacy_small)


def resolve_network_type(checkpoint: Path | str) -> str:
    """Resolve the recorded network implementation, defaulting to legacy MLP."""
    config = checkpoint_config(checkpoint)
    if config is None:
        return MLP_NETWORK
    value = config.get("network_type", MLP_NETWORK)
    return str(value)


def resolve_small_network(
    checkpoint: Path | str,
    *,
    small_network: bool | None = None,
) -> bool:
    """Resolve network size from an explicit flag or a sibling run config.json."""
    if small_network is not None:
        return small_network

    config = checkpoint_config(checkpoint)
    if config is not None and "small_network" in config:
        return bool(config["small_network"])
    return False


def checkpoint_config(checkpoint: Path | str) -> dict[str, Any] | None:
    """Load the resolved training config stored beside a checkpoint."""
    config_path = checkpoint_config_path(checkpoint)
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    config = data.get("config")
    return config if isinstance(config, dict) else None


def normalize_checkpoint_path(checkpoint: Path | str) -> Path:
    """Return the checkpoint file without resolving Hugging Face blob symlinks."""
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_dir() and (path / "policy.params").is_file():
        path = path / "policy.params"
    return path


def checkpoint_config_path(checkpoint: Path | str) -> Path:
    """Locate SimRig metadata for final parameters or a numeric Orbax checkpoint."""
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    # Hugging Face snapshots symlink policy.params to a content-addressed blob.
    # config.json sits next to the snapshot name, so do not resolve that file.
    if path.is_file():
        sibling = path.parent / "config.json"
        if sibling.is_file():
            return sibling
    path = path.resolve()
    if path.is_dir() and (path / "policy.params").is_file():
        return path / "config.json"
    if path.is_dir() and path.parent.name == "checkpoints":
        return path.parent.parent / "config.json"
    return path.parent / "config.json"
