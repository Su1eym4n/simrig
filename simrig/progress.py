"""Local-first training progress artifacts for a run directory."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping

from simrig.io import save_json


def metrics_log_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "metrics.jsonl"


def progress_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "progress.json"


def jsonable(value: Any) -> Any:
    """Convert JAX/numpy scalars and nested mappings into JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def record_progress(
    run_dir: Path | str,
    *,
    num_steps: int,
    timesteps: int,
    metrics: Mapping[str, Any],
    elapsed_sec: float,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Append one eval row and rewrite progress.json."""
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    fraction = (num_steps / timesteps) if timesteps else None
    remaining = max(timesteps - num_steps, 0)
    eta_sec = None
    if num_steps > 0 and elapsed_sec > 0 and timesteps:
        eta_sec = elapsed_sec * remaining / num_steps
    payload = {
        "num_steps": int(num_steps),
        "timesteps": int(timesteps),
        "fraction": fraction,
        "elapsed_sec": float(elapsed_sec),
        "eta_sec": eta_sec,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": jsonable(dict(metrics)),
    }
    log_path = metrics_log_path(output)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    save_json(progress_path(output), payload)
    if writer is not None:
        _write_tensorboard(writer, payload)
    return payload


def read_progress(run_dir: Path | str) -> dict[str, Any] | None:
    path = progress_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def format_progress(progress: Mapping[str, Any] | None) -> str:
    if not progress:
        return "progress=unknown"
    num_steps = progress.get("num_steps")
    timesteps = progress.get("timesteps")
    metrics = progress.get("metrics") if isinstance(progress.get("metrics"), Mapping) else {}
    reward = metrics.get("eval/episode_reward")
    length = metrics.get("eval/avg_episode_length")
    parts = [f"progress steps={num_steps}/{timesteps}"]
    fraction = progress.get("fraction")
    if isinstance(fraction, (int, float)):
        parts.append(f"fraction={float(fraction):.4f}")
    if isinstance(reward, (int, float)):
        parts.append(f"eval_reward={float(reward):.3f}")
    if isinstance(length, (int, float)):
        parts.append(f"eval_length={float(length):.1f}")
    eta = progress.get("eta_sec")
    if isinstance(eta, (int, float)):
        parts.append(f"eta_sec={float(eta):.0f}")
    return " ".join(str(part) for part in parts)


def describe_run(run_dir: Path | str, *, lines: int = 30) -> str:
    """Describe a local or fetched run the same way remote status does."""
    if lines <= 0:
        raise ValueError("Log line count must be positive.")
    output = Path(run_dir).expanduser()
    pid = _read_pid(output / "train.pid")
    artifacts = (
        "complete"
        if (output / "policy.params").is_file()
        and (output / "final_metrics.json").is_file()
        and any((output / "checkpoints").glob("**/*"))
        else "incomplete"
    )
    manifest = _read_json_object(output / "run_manifest.json")
    manifest_status = manifest.get("status") if manifest is not None else None
    if pid is not None and _pid_running(pid):
        state = "running"
    elif manifest_status in {"completed", "failed", "aborted"}:
        state = str(manifest_status)
    elif artifacts == "complete":
        state = "completed"
    else:
        state = "stopped"
    chunks = [f"status={state} pid={pid if pid is not None else 'unknown'} artifacts={artifacts}"]
    progress = read_progress(output)
    if progress is not None:
        chunks.append(format_progress(progress))
    if manifest is not None:
        compute = manifest.get("compute")
        if isinstance(compute, Mapping):
            summary = [f"manifest={manifest_status or 'unknown'}"]
            for key in ("actual_progress_steps", "elapsed_sec", "gpu_hours", "estimated_cost"):
                value = compute.get(key)
                if value is not None:
                    summary.append(f"{key}={value}")
            chunks.append(" ".join(summary))
    log_path = output / "train.log"
    if log_path.is_file() and lines:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(text.splitlines()[-lines:])
        if tail:
            chunks.append(tail)
    return "\n".join(chunks)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def maybe_tensorboard_writer(run_dir: Path | str) -> Any | None:
    """Return a TensorBoard writer when an optional backend is installed."""
    log_dir = str(Path(run_dir) / "tb")
    for importer in (
        lambda: _summary_writer("torch.utils.tensorboard", log_dir),
        lambda: _summary_writer("tensorboardX", log_dir),
    ):
        writer = importer()
        if writer is not None:
            return writer
    return None


def _summary_writer(module_name: str, log_dir: str) -> Any | None:
    try:
        module = __import__(module_name, fromlist=["SummaryWriter"])
    except ImportError:
        return None
    writer_cls = getattr(module, "SummaryWriter", None)
    if writer_cls is None:
        return None
    return writer_cls(log_dir=log_dir)


def _write_tensorboard(writer: Any, payload: Mapping[str, Any]) -> None:
    step = int(payload.get("num_steps") or 0)
    add_scalar = getattr(writer, "add_scalar", None)
    if not callable(add_scalar):
        return
    add_scalar("train/fraction", float(payload.get("fraction") or 0.0), step)
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                add_scalar(str(key), float(value), step)


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        return None
    return int(text)


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
