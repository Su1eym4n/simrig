"""Load user-authored custom environment modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


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

    config = None
    default_config = getattr(module, "default_config", None)
    if callable(default_config):
        config = default_config()

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
