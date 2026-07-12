"""Training presets shared by CLI and library calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    "cloud": {
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


def preset(name: str) -> dict[str, Any]:
    """Return a mutable copy of a named preset."""
    if name not in PRESETS:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset: {name}. Choose one of: {choices}")
    return dict(PRESETS[name])


def hidden_sizes(small_network: bool) -> tuple[int, ...]:
    return (64, 64) if small_network else (512, 256, 128)


def resolve_small_network(
    checkpoint: Path | str,
    *,
    small_network: bool | None = None,
) -> bool:
    """Resolve network size from an explicit flag or a sibling run config.json."""
    if small_network is not None:
        return small_network

    config_path = Path(checkpoint).resolve().parent / "config.json"
    if not config_path.exists():
        return False

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    config = data.get("config")
    if isinstance(config, dict) and "small_network" in config:
        return bool(config["small_network"])
    return False

