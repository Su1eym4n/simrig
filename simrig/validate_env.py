"""Static checklist validation for custom env starter modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from simrig.scaffold import REQUIRED_CLASS_METHODS, REQUIRED_SECTION_MARKERS


@dataclass(frozen=True)
class EnvValidationResult:
    """Result of a static custom-env checklist (not a trainability claim)."""

    path: str
    passed: bool
    trainable: bool = False
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def validate_env(path: Path | str) -> EnvValidationResult:
    """Check that a custom env file has the expected scaffold structure.

    Passing means the static checklist is present. It does **not** mean the
    environment is trainable or that rewards/observations are correct.
    """
    env_path = Path(path).expanduser()
    missing: list[str] = []
    warnings: list[str] = []
    notes = [
        "Static checklist only. Passing does not mean the env is trainable.",
        "SimRig v0.1 does not train custom env modules end-to-end.",
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
    if "def default_config" not in source:
        missing.append("function missing: default_config")

    custom_env = _find_class(tree, "CustomEnv")
    if custom_env is None:
        missing.append("class missing: CustomEnv")
    else:
        method_names = {node.name for node in custom_env.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
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
    if passed:
        notes.append("Checklist structure looks complete. Fill in the SECTION bodies before proposing training.")
    return EnvValidationResult(
        path=str(env_path),
        passed=passed,
        missing=missing,
        warnings=warnings,
        notes=notes,
    )


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None
