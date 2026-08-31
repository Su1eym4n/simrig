"""Task-success helpers for headless evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from simrig.custom_env import is_env_module_path, load_custom_env_metadata


UNKNOWN_REASON = (
    "No task-specific success evaluator is configured; rollout completion and "
    "reward do not prove command tracking or task success."
)


def normalize_success_spec(value: Any) -> dict[str, Any]:
    """Return a plain success spec mapping, or empty if none was declared."""
    if value is None:
        return {}
    if callable(value):
        value = value()
    if not isinstance(value, Mapping):
        raise TypeError("success_spec must return a mapping.")
    spec = dict(value)
    spec.setdefault("metric", "success")
    spec.setdefault("threshold", 0.5)
    spec.setdefault("mode", "any")
    spec.setdefault("hold_steps", 1)
    return spec


def load_success_spec(env_name: str, env: Any) -> dict[str, Any]:
    """Resolve a success spec from the env instance or custom module."""
    instance = getattr(env, "success_spec", None)
    if callable(instance):
        return normalize_success_spec(instance())
    if isinstance(instance, Mapping):
        return normalize_success_spec(instance)
    if is_env_module_path(env_name):
        metadata = load_custom_env_metadata(env_name)
        return normalize_success_spec(metadata.get("success_spec") or None)
    return {}


def metric_value(metrics: Any, key: str) -> float | None:
    if metrics is None:
        return None
    if isinstance(metrics, Mapping) and key in metrics:
        return _as_float(metrics[key])
    getter = getattr(metrics, "get", None)
    if callable(getter):
        value = getter(key)
        if value is not None:
            return _as_float(value)
    value = getattr(metrics, key, None)
    if value is not None:
        return _as_float(value)
    return None


def collect_success_value(state: Any, spec: Mapping[str, Any] | None) -> float | None:
    """Read one success metric sample from an environment state."""
    key = str((spec or {}).get("metric") or "success")
    metrics = getattr(state, "metrics", None)
    value = metric_value(metrics, key)
    if value is not None:
        return value
    if key != "success":
        return metric_value(metrics, "success")
    return None


def evaluate_task_success(
    values: list[float],
    spec: Mapping[str, Any] | None,
) -> tuple[bool | None, str]:
    """Interpret collected success samples using an optional spec."""
    declared = bool(spec)
    if not values:
        if declared:
            metric = spec.get("metric", "success") if spec else "success"
            return False, f"Metric {metric!r} was not present in rollout metrics."
        return None, UNKNOWN_REASON

    threshold = float((spec or {}).get("threshold", 0.5))
    mode = str((spec or {}).get("mode", "any"))
    hold_steps = int((spec or {}).get("hold_steps", 1))
    if hold_steps < 1:
        hold_steps = 1

    if mode == "last":
        passed = values[-1] >= threshold
        reason = f"Last {spec_metric(spec)}={values[-1]:.3f} vs threshold {threshold:g}."
    elif mode == "max":
        peak = max(values)
        passed = peak >= threshold
        reason = f"Max {spec_metric(spec)}={peak:.3f} vs threshold {threshold:g}."
    elif mode == "hold":
        passed = _held(values, threshold, hold_steps)
        reason = (
            f"{spec_metric(spec)} held >={threshold:g} for {hold_steps} consecutive step(s)."
            if passed
            else (
                f"{spec_metric(spec)} never held >={threshold:g} for {hold_steps} "
                "consecutive step(s)."
            )
        )
    else:
        passed = any(value >= threshold for value in values)
        reason = (
            f"{spec_metric(spec)} reached {max(values):.3f} vs threshold {threshold:g}."
        )
    return passed, reason


def spec_metric(spec: Mapping[str, Any] | None) -> str:
    return str((spec or {}).get("metric") or "success")


def _held(values: list[float], threshold: float, hold_steps: int) -> bool:
    run = 0
    for value in values:
        if value >= threshold:
            run += 1
            if run >= hold_steps:
                return True
        else:
            run = 0
    return False


def _as_float(value: Any) -> float | None:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
