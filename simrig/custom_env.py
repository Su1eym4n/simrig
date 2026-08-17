"""Load user-authored custom environment modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from simrig.networks import normalize_network_spec


def is_env_module_path(env_ref: str | Path) -> bool:
    """Return True when env_ref points at a custom *.py env module."""
    text = str(env_ref).strip()
    if not text.endswith(".py"):
        return False
    return True


def resolve_env_label(env_ref: str | Path) -> str:
    """Stable label for reports/run dirs (basename without .py for modules)."""
    if is_env_module_path(env_ref):
        return Path(str(env_ref)).expanduser().stem
    return str(env_ref)


def import_env_module(path: Path | str) -> ModuleType:
    """Import a custom env module from a filesystem path."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise FileNotFoundError(f"Custom env module not found: {env_path}")
    if env_path.suffix != ".py":
        raise ValueError(f"Custom env module must be a .py file: {env_path}")

    module_name = f"simrig_custom_env_{env_path.stem}_{abs(hash(str(env_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom env module: {env_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_custom_env_metadata(path: Path | str) -> dict[str, Any]:
    """Load declarative metadata without constructing the environment.

    Supported optional module hooks are ``network_spec()``, ``vision_spec()``,
    and ``training_config()``. Uppercase mapping constants with the same names
    are also accepted for simple modules.
    """
    module = import_env_module(path)
    default_config = _call_optional_mapping(module, "default_config")
    network_value = _call_or_value(module, "network_spec", "NETWORK_SPEC")
    vision_value = _call_or_value(module, "vision_spec", "VISION_SPEC")
    training_value = _call_or_value(module, "training_config", "TRAINING_CONFIG")
    return {
        "default_config": default_config,
        "network_spec": normalize_network_spec(network_value),
        "vision_spec": _mapping_or_empty(vision_value, "vision_spec"),
        "training_config": _mapping_or_empty(training_value, "training_config"),
    }


def load_custom_env(
    path: Path | str,
    *,
    class_name: str = "CustomEnv",
    config_overrides: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a custom env from a module path.

    Supports:
    - make_env(config_overrides=...) factory if present
    - CustomEnv(config=...) / CustomEnv(config=..., config_overrides=...)
    """
    module = import_env_module(path)
    overrides = dict(config_overrides or {})

    default_config = getattr(module, "default_config", None)
    config = default_config() if callable(default_config) else None
    _ensure_mjx_warp_compat(config)

    make_env = getattr(module, "make_env", None)
    if callable(make_env):
        try:
            return make_env(config_overrides=overrides or None)
        except TypeError:
            return make_env()

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Custom env module {path} must define make_env() or class {class_name}."
        )

    if isinstance(config, dict):
        merged = dict(config)
        merged.update(overrides)
        return _construct_env(cls, config=merged, config_overrides=None)

    if config is not None and overrides:
        return _construct_env(cls, config=config, config_overrides=overrides)
    if config is not None:
        return _construct_env(cls, config=config, config_overrides=None)
    if overrides:
        return _construct_env(cls, config=overrides, config_overrides=None)
    return _construct_env(cls, config=None, config_overrides=None)


def _ensure_mjx_warp_compat(config: Any) -> bool:
    """Bridge MuJoCo 3.10 to Warp 1.13's relocated GraphMode enum.

    MuJoCo 3.10 imports an internal pre-1.13 Warp module.  Warp 1.13 is the
    first release with the ``fill_mode`` API used by MuJoCo's tiled Cholesky
    kernels, so use its public compatibility namespace when the generated
    MuJoCo types fell back to placeholders.
    """

    impl = (
        config.get("impl")
        if isinstance(config, Mapping)
        else getattr(config, "impl", None)
    )
    if str(impl).lower() != "warp":
        return False

    try:
        import mujoco.mjx.warp as mjxw  # type: ignore
        from warp.jax_experimental import GraphMode  # type: ignore
    except ImportError:
        return False

    changed = False
    if not hasattr(mjxw.types.GraphMode, "WARP"):
        mjxw.types.GraphMode = GraphMode
        changed = True
    if getattr(mjxw.types, "Callback", None) is None and getattr(
        mjxw,
        "mjwp_types",
        None,
    ) is not None:
        mjxw.types.Callback = mjxw.mjwp_types.Callback
        changed = True
    return changed


def _construct_env(cls: type, *, config: Any, config_overrides: dict[str, Any] | None) -> Any:
    if config is None and config_overrides is None:
        return cls()
    if config_overrides is None:
        try:
            return cls(config=config)
        except TypeError:
            return cls(config)
    try:
        return cls(config=config, config_overrides=config_overrides)
    except TypeError:
        try:
            return cls(config, config_overrides)
        except TypeError as exc:
            raise TypeError(
                f"Could not construct {cls.__name__} with config/config_overrides. "
                "Prefer make_env(config_overrides=...) in the module."
            ) from exc


def _call_optional_mapping(module: ModuleType, name: str) -> dict[str, Any]:
    value = getattr(module, name, None)
    if value is None:
        return {}
    if callable(value):
        value = value()
    return _mapping_or_empty(value, name)


def _call_or_value(module: ModuleType, function_name: str, constant_name: str) -> Any:
    value = getattr(module, function_name, None)
    if callable(value):
        return value()
    if value is not None:
        return value
    return getattr(module, constant_name, None)


def _mapping_or_empty(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must return a mapping.")
    return dict(value)
