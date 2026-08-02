"""Runtime manifests and checkpoint compatibility checks."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Any
import warnings

from simrig._version import __version__


_DISTRIBUTIONS = {
    "simrig": "simrig",
    "jax": "jax",
    "brax": "brax",
    "mujoco": "mujoco",
    "mujoco_mjx": "mujoco-mjx",
    "playground": "playground",
    "numpy": "numpy",
}


def runtime_manifest() -> dict[str, Any]:
    """Return the runtime versions needed to reproduce a policy rollout."""
    packages: dict[str, str | None] = {}
    for label, distribution in _DISTRIBUTIONS.items():
        if label == "simrig":
            packages[label] = __version__
            continue
        try:
            packages[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[label] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def checkpoint_runtime(
    checkpoint: Path | str,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the training runtime from a checkpoint's sibling config.json."""
    config_path = Path(checkpoint).expanduser().resolve().parent / "config.json"
    if not config_path.is_file():
        return None, None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, config_path
    config = payload.get("config")
    if not isinstance(config, dict):
        return None, config_path
    manifest = config.get("runtime")
    return (manifest if isinstance(manifest, dict) else None), config_path


def runtime_mismatches(
    expected: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    """Compare compatibility-critical Python and package versions."""
    mismatches: list[str] = []
    expected_python = str(expected.get("python") or "")
    current_python = str(current.get("python") or "")
    if _major_minor(expected_python) != _major_minor(current_python):
        mismatches.append(f"python: trained={expected_python} current={current_python}")

    expected_packages = expected.get("packages")
    current_packages = current.get("packages")
    if not isinstance(expected_packages, dict) or not isinstance(current_packages, dict):
        return mismatches
    for name in _DISTRIBUTIONS:
        trained = expected_packages.get(name)
        active = current_packages.get(name)
        if trained is not None and trained != active:
            mismatches.append(f"{name}: trained={trained} current={active}")
    return mismatches


def verify_checkpoint_runtime(
    checkpoint: Path | str,
    *,
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    """Verify the active runtime against recorded checkpoint metadata."""
    expected, config_path = checkpoint_runtime(checkpoint)
    current = runtime_manifest()
    if expected is None:
        return {
            "recorded": False,
            "compatible": None,
            "config_path": str(config_path) if config_path else None,
            "mismatches": [],
            "current": current,
        }

    mismatches = runtime_mismatches(expected, current)
    if mismatches and not allow_mismatch:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(
            "Checkpoint runtime differs from the training runtime:\n"
            f"{details}\n"
            "Recreate the recorded environment or pass --allow-runtime-mismatch "
            "for an explicitly qualitative compatibility check."
        )
    if mismatches:
        warnings.warn(
            "Checkpoint runtime mismatch explicitly allowed; this rollout is "
            "qualitative only: " + "; ".join(mismatches),
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "recorded": True,
        "compatible": not mismatches,
        "config_path": str(config_path),
        "mismatches": mismatches,
        "training": expected,
        "current": current,
    }


def _major_minor(version: str) -> tuple[int, int] | None:
    try:
        major, minor, *_ = version.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None
