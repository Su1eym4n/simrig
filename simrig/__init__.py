"""SimRig public API."""

from __future__ import annotations

from typing import Any

from simrig.core import (
    BackendInfo,
    EnvInspectionReport,
    ModelInspectionReport,
    RunConfig,
    SmokeResult,
    TrainabilityStatus,
)
from simrig.mujoco_backend import inspect_model, list_models
from simrig.playground_backend import (
    demo_policy,
    eval_policy,
    inspect_env,
    list_envs,
    smoke_env,
    train_ppo,
)
from simrig.scaffold import new_env
from simrig.validate_env import EnvValidationResult, validate_env

__all__ = [
    "BackendInfo",
    "EnvInspectionReport",
    "EnvValidationResult",
    "ModelInspectionReport",
    "RunConfig",
    "SmokeResult",
    "TrainabilityStatus",
    "demo_policy",
    "eval_policy",
    "inspect_env",
    "inspect_model",
    "list_envs",
    "list_models",
    "new_env",
    "smoke_env",
    "serve_model_view",
    "serve_policy_preview",
    "train_ppo",
    "validate_env",
]


def __getattr__(name: str) -> Any:
    """Lazy-load browser helpers so core imports work without numpy/mujoco."""
    if name == "serve_model_view":
        from simrig.model_view import serve_model_view

        return serve_model_view
    if name == "serve_policy_preview":
        from simrig.preview import serve_policy_preview

        return serve_policy_preview
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
