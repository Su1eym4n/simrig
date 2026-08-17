"""Backend-neutral SimRig data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class TrainabilityStatus(str, Enum):
    """Coarse status for what SimRig can safely do with an asset."""

    UNKNOWN = "unknown"
    INSPECTABLE = "inspectable"
    SIMULATABLE = "simulatable"
    TRAINABLE_EXISTING_ENV = "trainable_existing_env"
    NEEDS_CUSTOM_ENV = "needs_custom_env"
    FAILED = "failed"


@dataclass(frozen=True)
class BackendInfo:
    """Information about a simulation/training backend."""

    name: str
    available: bool
    version: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ModelInspectionReport:
    """Summary of a MuJoCo model inspection."""

    name: str
    path: str
    backend: str
    status: TrainabilityStatus
    compiled: bool
    stepped: bool
    bodies: int = 0
    joints: int = 0
    dofs: int = 0
    actuators: int = 0
    sensors: int = 0
    keyframes: int = 0
    has_freejoint: bool = False
    has_mjx_hint: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EnvInspectionReport:
    """Summary of a training environment inspection."""

    name: str
    backend: str
    status: TrainabilityStatus
    available: bool
    loaded: bool
    observation_size: Any = None
    action_size: int | None = None
    xml_path: str | None = None
    model_bodies: int | None = None
    model_actuators: int | None = None
    has_domain_randomizer: bool | None = None
    network_type: str | None = None
    vision: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunConfig:
    """Resolved training or evaluation run metadata."""

    env_name: str
    backend: str
    preset: str
    output_dir: str
    config: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SmokeResult:
    """Result of a short environment smoke test."""

    env_name: str
    backend: str
    steps_requested: int
    steps_completed: int
    passed: bool
    action_size: int | None = None
    observation_size: Any = None
    final_reward: float | None = None
    final_done: bool | None = None
    errors: list[str] = field(default_factory=list)


def to_dict(value: Any) -> Any:
    """Convert dataclasses, enums, paths, and containers into JSONable values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # JAX/NumPy scalars and arrays.
    item = getattr(value, "item", None)
    if callable(item):
        shape = getattr(value, "shape", None)
        if shape == ():
            return to_dict(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_dict(tolist())

    return value


def report_markdown(title: str, report: Any) -> str:
    """Render a compact Markdown report for humans and agents."""
    data = to_dict(report)
    lines = [f"# {title}", ""]
    for key, value in data.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value) if value else "none"
        else:
            rendered = str(value)
        lines.append(f"- **{label}:** {rendered}")
    lines.append("")
    return "\n".join(lines)
