"""Lifecycle and compute accounting for reproducible SimRig runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from simrig.core import RunConfig
from simrig.io import save_json
from simrig.progress import read_progress


MANIFEST_SCHEMA_VERSION = 2


def run_manifest_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "run_manifest.json"


def start_run_manifest(
    run_config: RunConfig,
    *,
    task_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write an immutable-input manifest before the training call starts."""
    config = run_config.config
    provenance = config.get("provenance") if isinstance(config, Mapping) else None
    runtime = config.get("runtime") if isinstance(config, Mapping) else None
    payload = {
        "kind": "simrig.run-manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "env_name": run_config.env_name,
        "backend": run_config.backend,
        "preset": run_config.preset,
        "command": list(run_config.command),
        "task_contract": dict(task_contract) if task_contract is not None else None,
        "evaluator": config.get("evaluator"),
        "independent_evaluations": [],
        "lineage": {
            "resumed_from": config.get("resumed_from"),
        },
        "compute": {
            "requested_timesteps": config.get("timesteps"),
            "estimated_timesteps": config.get(
                "estimated_total_timesteps", config.get("timesteps")
            ),
            "num_envs": config.get("num_envs"),
            "batch_size": config.get("batch_size"),
            "elapsed_sec": None,
            "actual_progress_steps": 0,
            "gpu_count": _gpu_count(provenance),
            "gpu_hours": None,
            "estimated_cost": None,
        },
        "runtime": runtime,
        "provenance": provenance,
        "failure": None,
    }
    save_json(run_manifest_path(run_config.output_dir), payload)
    return payload


def record_independent_evaluation(
    run_dir: Path | str,
    *,
    evaluator: Mapping[str, Any],
    report_paths: list[str],
    bounded: bool,
) -> dict[str, Any] | None:
    """Attach evaluator identity and report paths to an existing run manifest."""
    path = run_manifest_path(run_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    evaluations = payload.setdefault("independent_evaluations", [])
    if not isinstance(evaluations, list):
        evaluations = []
        payload["independent_evaluations"] = evaluations
    evaluations.append(
        {
            "recorded_at": _now(),
            "evaluator": dict(evaluator),
            "report_paths": list(report_paths),
            "bounded": bool(bounded),
        }
    )
    save_json(path, payload)
    return payload


def finish_run_manifest(
    run_dir: Path | str,
    *,
    status: str,
    elapsed_sec: float,
    failure: BaseException | None = None,
    gpu_hourly_cost: float | None = None,
) -> dict[str, Any]:
    """Finalize status and measured compute while preserving original inputs."""
    path = run_manifest_path(run_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Run manifest is missing or invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Run manifest root is invalid: {path}")
    compute = payload.get("compute")
    if not isinstance(compute, dict):
        compute = {}
        payload["compute"] = compute
    progress = read_progress(run_dir) or {}
    gpu_count = int(compute.get("gpu_count") or 0)
    gpu_hours = float(elapsed_sec) * gpu_count / 3600.0
    compute.update(
        {
            "elapsed_sec": float(elapsed_sec),
            "actual_progress_steps": int(progress.get("num_steps") or 0),
            "gpu_hours": gpu_hours,
            "estimated_cost": (
                gpu_hours * float(gpu_hourly_cost)
                if gpu_hourly_cost is not None
                else None
            ),
        }
    )
    payload["status"] = status
    payload["finished_at"] = _now()
    payload["failure"] = (
        {"type": type(failure).__name__, "message": str(failure)}
        if failure is not None
        else None
    )
    save_json(path, payload)
    return payload


def _gpu_count(provenance: Any) -> int:
    if not isinstance(provenance, Mapping):
        return 0
    jax = provenance.get("jax")
    devices = jax.get("devices") if isinstance(jax, Mapping) else None
    if not isinstance(devices, list):
        return 0
    return sum(
        str(device.get("platform", "")).lower() in {"cuda", "gpu", "rocm"}
        for device in devices
        if isinstance(device, Mapping)
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
