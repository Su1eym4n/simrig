"""SimRig public API."""

from __future__ import annotations

from typing import Any

from simrig._version import __version__
from simrig.core import (
    BackendInfo,
    EnvInspectionReport,
    ModelInspectionReport,
    RunConfig,
    SmokeResult,
    TrainabilityStatus,
)
from simrig.evaluation_suite import (
    EvaluationLimits,
    evaluate_checkpoint_directory,
    run_evaluation_suite,
)
from simrig.evaluator import (
    EvaluationRequest,
    EvaluatorPlugin,
    LoadedEvaluator,
    load_evaluator,
    run_evaluator,
)
from simrig.failures import FailureCategory
from simrig.mujoco_backend import inspect_model, list_models
from simrig.playground_backend import (
    demo_policy,
    eval_policy,
    inspect_env,
    list_envs,
    smoke_env,
    train_ppo,
)
from simrig.predicates import (
    PredicateResult,
    apply_predicates,
    evaluate_predicate,
    validate_predicate,
)
from simrig.ranking import rank_checkpoints
from simrig.scaffold import new_env
from simrig.task_contract import (
    COMPATIBILITY_POLICIES,
    ContractValidation,
    compare_task_contracts,
    contract_sha256,
    diff_task_contracts,
    freeze_task_contract,
    load_task_contract,
    migrate_task_contract,
    save_migrated_task_contract,
    task_contract_template,
    validate_task_contract,
)
from simrig.validate_env import EnvValidationResult, validate_env

__all__ = [
    "__version__",
    "BackendInfo",
    "COMPATIBILITY_POLICIES",
    "ContractValidation",
    "EvaluationLimits",
    "EvaluationRequest",
    "EvaluatorPlugin",
    "EnvInspectionReport",
    "EnvValidationResult",
    "FailureCategory",
    "LoadedEvaluator",
    "ModelInspectionReport",
    "RunConfig",
    "SmokeResult",
    "TrainabilityStatus",
    "PredicateResult",
    "apply_predicates",
    "compare_task_contracts",
    "demo_policy",
    "contract_sha256",
    "diff_task_contracts",
    "eval_policy",
    "evaluate_checkpoint_directory",
    "evaluate_predicate",
    "inspect_env",
    "inspect_model",
    "is_env_module_path",
    "list_envs",
    "list_models",
    "load_evaluator",
    "LiveWebViewer",
    "load_task_contract",
    "load_custom_env",
    "load_env",
    "new_env",
    "migrate_task_contract",
    "rank_checkpoints",
    "freeze_task_contract",
    "smoke_env",
    "serve_model_view",
    "serve_policy_preview",
    "run_evaluation_suite",
    "run_evaluator",
    "save_migrated_task_contract",
    "train_ppo",
    "task_contract_template",
    "validate_task_contract",
    "validate_predicate",
    "validate_env",
]


def __getattr__(name: str) -> Any:
    """Lazy-load browser helpers so core imports work without numpy/mujoco."""
    if name == "LiveWebViewer":
        from simrig.live_view import LiveWebViewer

        return LiveWebViewer
    if name == "serve_model_view":
        from simrig.model_view import serve_model_view

        return serve_model_view
    if name == "serve_policy_preview":
        from simrig.preview import serve_policy_preview

        return serve_policy_preview
    if name == "is_env_module_path":
        from simrig.custom_env import is_env_module_path

        return is_env_module_path
    if name == "load_custom_env":
        from simrig.custom_env import load_custom_env

        return load_custom_env
    if name == "load_env":
        from simrig.playground_backend import load_env

        return load_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
