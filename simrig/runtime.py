"""Runtime manifests and checkpoint compatibility checks."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
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


def training_provenance(
    env_ref: Path | str,
    env: Any,
    *,
    jax: Any,
) -> dict[str, Any]:
    """Capture reproducibility-relevant source, device, and process details."""
    source: dict[str, Any] = {"env_ref": str(env_ref)}
    env_path = Path(str(env_ref)).expanduser()
    if env_path.is_file():
        source["env_module_path"] = str(env_path.resolve())
        source["env_module_sha256"] = _sha256(env_path)

    xml_path = getattr(env, "xml_path", None)
    if xml_path is not None:
        model_path = Path(str(xml_path)).expanduser()
        source["model_xml_path"] = str(model_path.resolve())
        if model_path.is_file():
            source["model_xml_sha256"] = _sha256(model_path)

    devices = []
    for device in jax.devices():
        devices.append(
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
            }
        )

    return {
        "runtime": runtime_manifest(),
        "source": source,
        "git": _git_state(Path.cwd()),
        "jax": {
            "default_backend": jax.default_backend(),
            "devices": devices,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_PLATFORMS",
                "MUJOCO_GL",
            )
        },
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any] | None:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {"root": root, "commit": commit, "dirty": dirty}
