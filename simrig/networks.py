"""Network metadata and factory selection for SimRig policies."""

from __future__ import annotations

import functools
from typing import Any, Mapping


MLP_NETWORK = "mlp"
VISION_CNN_NETWORK = "vision_cnn"
SUPPORTED_NETWORK_TYPES = (MLP_NETWORK, VISION_CNN_NETWORK)


def normalize_network_spec(
    value: Any,
    *,
    default_factory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated, JSON-friendly network specification.

    Environment authors may return either ``{"type": ..., "factory": {...}}``
    or put factory keyword arguments beside ``type`` for compact declarations.
    """
    if value is None:
        return {
            "type": MLP_NETWORK,
            "factory": dict(default_factory or {}),
        }
    if isinstance(value, str):
        value = {"type": value}
    if not isinstance(value, Mapping):
        raise TypeError("network_spec must be a mapping, string, or None.")

    raw = dict(value)
    network_type = str(raw.pop("type", raw.get("kind", MLP_NETWORK)))
    raw.pop("kind", None)
    if network_type not in SUPPORTED_NETWORK_TYPES:
        choices = ", ".join(SUPPORTED_NETWORK_TYPES)
        raise ValueError(
            f"Unsupported network type: {network_type}. Choose one of: {choices}"
        )

    declared_factory = raw.pop("factory", {})
    if not isinstance(declared_factory, Mapping):
        raise TypeError("network_spec['factory'] must be a mapping.")
    factory = dict(default_factory or {})
    factory.update(dict(declared_factory))
    # Compact form: remaining keys are Brax network factory kwargs.
    factory.update(raw)
    return {"type": network_type, "factory": factory}


def make_network_factory(
    network_type: str,
    network_config: Mapping[str, Any],
    *,
    ppo_networks: Any,
) -> Any:
    """Build the configured Brax PPO network factory lazily."""
    if network_type == MLP_NETWORK:
        constructor = ppo_networks.make_ppo_networks
    elif network_type == VISION_CNN_NETWORK:
        try:
            from brax.training.agents.ppo import networks_vision  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Vision PPO requires a Brax build with "
                "brax.training.agents.ppo.networks_vision."
            ) from exc
        constructor = networks_vision.make_ppo_networks_vision
    else:
        choices = ", ".join(SUPPORTED_NETWORK_TYPES)
        raise ValueError(
            f"Unsupported network type: {network_type}. Choose one of: {choices}"
        )
    return functools.partial(constructor, **dict(network_config))


def is_vision_network(network_type: str) -> bool:
    return network_type == VISION_CNN_NETWORK
