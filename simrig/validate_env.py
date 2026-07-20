"""Validation for custom env starter modules (static + optional runtime)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simrig.scaffold import REQUIRED_CLASS_METHODS, REQUIRED_SECTION_MARKERS


@dataclass(frozen=True)
class EnvValidationResult:
    """Result of custom-env validation.

    ``trainable`` is True only when ``--runtime`` reset/step checks succeed.
    Static-only passes never claim trainability.
    """

    path: str
    passed: bool
    trainable: bool = False
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def validate_env(path: Path | str, *, runtime: bool = False) -> EnvValidationResult:
    """Validate a custom env module.

    Static mode checks scaffold structure. Runtime mode also imports the module,
    constructs the env, and runs a short reset/step smoke when JAX is available.
    """
    env_path = Path(path).expanduser()
    missing: list[str] = []
    warnings: list[str] = []
    notes = [
        "Static checklist checks structure only.",
        "Use --runtime before proposing simrig smoke/train on a custom module.",
    ]

    if not env_path.is_file():
        return EnvValidationResult(
            path=str(env_path),
            passed=False,
            missing=[f"file not found: {env_path}"],
            notes=notes,
        )

    source = env_path.read_text(encoding="utf-8")
    for marker in REQUIRED_SECTION_MARKERS:
        if marker not in source:
            missing.append(f"section marker missing: {marker}")

    try:
        tree = ast.parse(source, filename=str(env_path))
    except SyntaxError as exc:
        return EnvValidationResult(
            path=str(env_path),
            passed=False,
            missing=[f"syntax error: {exc}"],
            notes=notes,
        )

    if "MODEL_PATH" not in source:
        missing.append("symbol missing: MODEL_PATH")
    if "ENV_NAME" not in source:
        missing.append("symbol missing: ENV_NAME")
    if "def default_config" not in source and "def make_env" not in source:
        missing.append("function missing: default_config or make_env")

    custom_env = _find_class(tree, "CustomEnv")
    has_make_env = any(
        isinstance(node, ast.FunctionDef) and node.name == "make_env" for node in tree.body
    )
    if custom_env is None and not has_make_env:
        missing.append("class missing: CustomEnv (or define make_env)")
    elif custom_env is not None:
        method_names = {
            node.name
            for node in custom_env.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in REQUIRED_CLASS_METHODS:
            if name not in method_names:
                missing.append(f"CustomEnv method missing: {name}")
        for prop in ("observation_size", "action_size"):
            if prop not in method_names:
                missing.append(f"CustomEnv property missing: {prop}")

    if "NotImplementedError" in source:
        warnings.append("NotImplementedError still present; implementation is incomplete.")
    if "NOT TRAINABLE YET" in source:
        warnings.append("File still marked NOT TRAINABLE YET.")

    passed = not missing
    trainable = False

    if runtime and passed:
        runtime_missing, runtime_warnings, runtime_notes, trainable = _runtime_checks(env_path)
        missing.extend(runtime_missing)
        warnings.extend(runtime_warnings)
        notes.extend(runtime_notes)
        passed = not missing
    elif passed:
        notes.append(
            "Checklist structure looks complete. Fill SECTION bodies, then "
            "`simrig validate-env PATH --runtime` and `simrig smoke PATH`."
        )

    return EnvValidationResult(
        path=str(env_path),
        passed=passed,
        trainable=trainable,
        missing=missing,
        warnings=warnings,
        notes=notes,
    )


def _runtime_checks(env_path: Path) -> tuple[list[str], list[str], list[str], bool]:
    missing: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    trainable = False

    try:
        from simrig.custom_env import load_custom_env

        env = load_custom_env(env_path)
    except Exception as exc:
        missing.append(f"runtime load failed: {exc}")
        return missing, warnings, notes, False

    notes.append("Custom env module constructed successfully.")
    action_size = getattr(env, "action_size", None)
    observation_size = getattr(env, "observation_size", None)

    if action_size is None:
        missing.append("runtime: action_size missing")
    obs_warnings = _obs_size_warnings(observation_size)
    warnings.extend(obs_warnings)

    for attr in ("mj_model", "mjx_model"):
        if getattr(env, attr, None) is None:
            warnings.append(f"runtime: {attr} missing (needed for Playground/Brax demos).")

    try:
        import jax  # type: ignore
        import jax.numpy as jp  # type: ignore
    except ImportError:
        warnings.append("runtime: JAX not installed; skipped reset/step smoke.")
        notes.append("Install the playground extra to run reset/step validation.")
        return missing, warnings, notes, False

    try:
        state = env.reset(jax.random.PRNGKey(0))
        obs = getattr(state, "obs", None)
        if isinstance(obs, dict):
            if "state" not in obs:
                missing.append("runtime: state.obs missing key `state`")
            if "privileged_state" not in obs:
                warnings.append("runtime: state.obs missing key `privileged_state`")
        elif obs is None:
            missing.append("runtime: state.obs missing")
        else:
            warnings.append(
                "runtime: state.obs is flat; SimRig PPO defaults expect dict keys "
                "`state` and `privileged_state`."
            )

        if action_size is None:
            return missing, warnings, notes, False
        state = env.step(state, jp.zeros(int(action_size)))
        _ = float(state.reward), bool(state.done)
    except Exception as exc:
        missing.append(f"runtime reset/step failed: {exc}")
        return missing, warnings, notes, False

    if missing:
        return missing, warnings, notes, False

    trainable = True
    notes.append("Runtime reset/step succeeded. Safe to try `simrig smoke` then `simrig train`.")
    return missing, warnings, notes, trainable


def _obs_size_warnings(observation_size: Any) -> list[str]:
    warnings: list[str] = []
    if observation_size is None:
        warnings.append("runtime: observation_size missing")
        return warnings
    if isinstance(observation_size, dict):
        if "state" not in observation_size:
            warnings.append("runtime: observation_size missing key `state`")
        if "privileged_state" not in observation_size:
            warnings.append("runtime: observation_size missing key `privileged_state`")
    else:
        warnings.append(
            "runtime: observation_size is flat; SimRig PPO defaults expect "
            "`state` / `privileged_state`."
        )
    return warnings


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None
