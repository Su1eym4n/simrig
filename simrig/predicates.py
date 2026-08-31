"""Generic predicates over evaluator metrics, events, signals, and contacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from simrig.failures import (
    FAILURE_PRIORITY,
    FailureCategory,
    normalize_terminal_reason,
    terminal_reason,
)


PREDICATE_TYPES = frozenset(
    {"event_count", "forbidden_contact", "metric", "sequence", "sustained"}
)
OPERATORS = frozenset({"<", "<=", "==", ">=", ">", "!="})


@dataclass(frozen=True)
class PredicateResult:
    predicate_id: str
    predicate_type: str
    passed: bool
    required: bool
    actual: Any
    expected: Any
    reason: str
    failure_category: str
    failure_code: str
    evidence_available: bool = True


def validate_predicate(predicate: Mapping[str, Any], *, path: str = "predicate") -> list[str]:
    """Return schema errors for one backend-neutral predicate declaration."""
    errors: list[str] = []
    predicate_type = predicate.get("type")
    if predicate_type not in PREDICATE_TYPES:
        errors.append(f"{path}.type must be one of {sorted(PREDICATE_TYPES)}")
        return errors
    predicate_id = predicate.get("id")
    if not isinstance(predicate_id, str) or not predicate_id.strip():
        errors.append(f"{path}.id must be non-empty text")
    if "required" in predicate and not isinstance(predicate["required"], bool):
        errors.append(f"{path}.required must be a boolean")
    category = predicate.get("failure_category", FailureCategory.TASK_FAILURE.value)
    try:
        terminal_reason(
            str(category),
            str(predicate.get("failure_code") or "predicate_failed"),
            "x",
        )
    except ValueError as exc:
        errors.append(f"{path}: {exc}")

    if predicate_type in {"event_count", "sequence", "sustained"}:
        if predicate_type == "sequence":
            events = predicate.get("events")
            if not isinstance(events, list) or not events or not all(
                isinstance(item, str) and item for item in events
            ):
                errors.append(f"{path}.events must be a non-empty text list")
        else:
            event = predicate.get("event")
            if not isinstance(event, str) or not event:
                errors.append(f"{path}.event must be non-empty text")
    if predicate_type == "sustained":
        hold_steps = predicate.get("hold_steps")
        if not isinstance(hold_steps, int) or isinstance(hold_steps, bool) or hold_steps < 1:
            errors.append(f"{path}.hold_steps must be a positive integer")
    if predicate_type == "forbidden_contact":
        for key in ("body_a", "body_b"):
            if not isinstance(predicate.get(key), str) or not predicate.get(key):
                errors.append(f"{path}.{key} must be non-empty text")
        max_count = predicate.get("max_count", 0)
        if not isinstance(max_count, int) or isinstance(max_count, bool) or max_count < 0:
            errors.append(f"{path}.max_count must be a non-negative integer")
        min_force = _number(predicate.get("min_force", 0.0))
        if min_force is None or min_force < 0:
            errors.append(f"{path}.min_force must be a non-negative number")
    if predicate_type == "metric":
        if not isinstance(predicate.get("metric"), str) or not predicate.get("metric"):
            errors.append(f"{path}.metric must be non-empty text")
        if predicate.get("operator", "<=") not in OPERATORS:
            errors.append(f"{path}.operator must be one of {sorted(OPERATORS)}")
        if _number(predicate.get("value")) is None:
            errors.append(f"{path}.value must be numeric")
    if predicate_type == "event_count":
        for key in ("min_count", "max_count"):
            if key in predicate:
                value = predicate[key]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"{path}.{key} must be a non-negative integer")
        minimum = predicate.get("min_count", 0)
        maximum = predicate.get("max_count")
        if (
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and minimum > maximum
        ):
            errors.append(f"{path}.min_count must not exceed max_count")
    return errors


def apply_predicates(
    record: Mapping[str, Any],
    predicates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive independent success and terminal reason from raw evaluator data."""
    normalized = dict(record)
    results = [evaluate_predicate(normalized, predicate) for predicate in predicates]
    plugin_reason = normalize_terminal_reason(
        normalized.get("terminal_reason"),
        task_success=normalized.get("task_success")
        if isinstance(normalized.get("task_success"), bool)
        else None,
    )
    plugin_failed = plugin_reason["category"] not in {
        FailureCategory.SUCCESS.value,
        FailureCategory.UNKNOWN.value,
    }
    failed_results = [item for item in results if item.required and not item.passed]
    failed = (
        min(
            failed_results,
            key=lambda item: FAILURE_PRIORITY.get(item.failure_category, 100),
        )
        if failed_results
        else None
    )
    predicate_priority = (
        FAILURE_PRIORITY.get(failed.failure_category, 100) if failed is not None else 100
    )
    plugin_priority = (
        FAILURE_PRIORITY.get(str(plugin_reason["category"]), 100) if plugin_failed else 100
    )
    if failed is not None and predicate_priority < plugin_priority:
        success = False
        reason = terminal_reason(
            failed.failure_category,
            failed.failure_code,
            failed.reason,
            details={
                "predicate_id": failed.predicate_id,
                "actual": failed.actual,
                "expected": failed.expected,
            },
        )
    elif plugin_failed:
        success = False
        reason = plugin_reason
    elif failed is not None:
        success = False
        reason = terminal_reason(
            failed.failure_category,
            failed.failure_code,
            failed.reason,
            details={
                "predicate_id": failed.predicate_id,
                "actual": failed.actual,
                "expected": failed.expected,
            },
        )
    elif any(item.required for item in results):
        success = True
        reason = terminal_reason(
            FailureCategory.SUCCESS,
            "all_required_predicates_passed",
            "All required independent predicates passed.",
        )
    else:
        success = None
        reason = terminal_reason(
            FailureCategory.INCOMPLETE, "insufficient_evidence",
            "No required independent success predicates were evaluated.",
        )
    if reason["code"] == "insufficient_evidence":
        success = None
    normalized["predicate_results"] = [item.__dict__ for item in results]
    normalized["task_success"] = success
    normalized["terminal_reason"] = reason
    return normalized


def evaluate_predicate(
    record: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> PredicateResult:
    errors = validate_predicate(predicate)
    if errors:
        raise ValueError("Invalid predicate: " + "; ".join(errors))
    predicate_type = str(predicate["type"])
    predicate_id = str(predicate["id"])
    required = bool(predicate.get("required", True))
    expected: Any
    actual: Any
    if predicate_type == "sustained":
        expected = int(predicate["hold_steps"])
        actual = _longest_active_run(record.get("events"), str(predicate["event"]))
        passed = actual >= expected
    elif predicate_type == "forbidden_contact":
        expected = int(predicate.get("max_count", 0))
        actual = _contact_count(
            record.get("events"),
            str(predicate["body_a"]),
            str(predicate["body_b"]),
            min_force=float(predicate.get("min_force", 0.0)),
        )
        passed = actual <= expected
    elif predicate_type == "event_count":
        actual = _event_count(record.get("events"), str(predicate["event"]))
        minimum = int(predicate.get("min_count", 0))
        maximum = predicate.get("max_count")
        expected = {"min_count": minimum, "max_count": maximum}
        passed = actual >= minimum and (maximum is None or actual <= int(maximum))
    elif predicate_type == "sequence":
        expected = list(predicate["events"])
        actual = _event_sequence(record.get("events"))
        passed = _is_subsequence(expected, actual)
    else:
        metrics = record.get("metrics")
        metric = str(predicate["metric"])
        raw = metrics.get(metric) if isinstance(metrics, Mapping) else None
        actual = _number(raw)
        expected = float(predicate["value"])
        passed = actual is not None and _compare(
            actual,
            str(predicate.get("operator", "<=")),
            expected,
        )
    missing = _missing_evidence(record, predicate, actual, passed)
    if missing:
        return PredicateResult(
            predicate_id, predicate_type, False, required, actual, expected,
            missing, FailureCategory.INCOMPLETE.value, "insufficient_evidence", False,
        )
    category = str(predicate.get("failure_category", FailureCategory.TASK_FAILURE.value))
    code = str(predicate.get("failure_code") or f"{predicate_id}_failed")
    reason = (
        f"Predicate {predicate_id!r} passed."
        if passed
        else f"Predicate {predicate_id!r} failed: actual={actual!r} expected={expected!r}."
    )
    return PredicateResult(
        predicate_id=predicate_id,
        predicate_type=predicate_type,
        passed=passed,
        required=required,
        actual=actual,
        expected=expected,
        reason=reason,
        failure_category=category,
        failure_code=code,
    )


def _missing_evidence(record: Mapping[str, Any], predicate: Mapping[str, Any], actual: Any, passed: bool) -> str | None:
    """Absence of reported events is not evidence that a sensor observed none."""
    kind = predicate["type"]
    if kind == "metric":
        return None if actual is not None else f"Missing finite metric {predicate['metric']!r}."
    events = record.get("events")
    if not isinstance(events, list):
        return "Missing event stream."
    evidence = record.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if kind == "forbidden_contact":
        # An observed violation is enough to fail even an incomplete stream.
        if not passed:
            return None
        target = {predicate["body_a"], predicate["body_b"]}
        covered = any(
            isinstance(pair, Mapping) and pair.get("complete") is True
            and {pair.get("body_a"), pair.get("body_b")} == target
            for pair in evidence.get("contacts", [])
        )
        if not covered:
            return "No complete contact evidence declared for the required body pair."
        if float(predicate.get("min_force", 0)) > 0 and any(
            event.get("kind") == "contact"
            and {event.get("body_a"), event.get("body_b")} == target
            and _number(event.get("force")) is None
            for event in _events(events)
        ):
            return "Contact force was not measured."
        return None
    names = predicate["events"] if kind == "sequence" else [predicate["event"]]
    declared = evidence.get("events", [])
    observed = {event.get("name") for event in _events(events) if event.get("kind") != "contact"}
    if any(name not in observed and name not in declared for name in names):
        return "Required event/signal channel was not measured."
    if kind == "event_count" and "max_count" in predicate and passed and predicate["event"] not in declared:
        return "An upper event-count bound requires a declared complete event channel."
    return None


def _events(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _event_count(value: Any, name: str) -> int:
    return sum(
        str(item.get("name")) == name and bool(item.get("active", True))
        for item in _events(value)
        if item.get("kind", "event") != "contact"
    )


def _event_sequence(value: Any) -> list[str]:
    ordered = sorted(_events(value), key=lambda item: int(item.get("step", 0)))
    return [
        str(item.get("name"))
        for item in ordered
        if item.get("kind", "event") != "contact" and bool(item.get("active", True))
    ]


def _longest_active_run(value: Any, name: str) -> int:
    samples = sorted(
        (
            (int(item.get("step", 0)), bool(item.get("active", True)))
            for item in _events(value)
            if str(item.get("name")) == name and item.get("kind", "signal") != "contact"
        ),
        key=lambda item: item[0],
    )
    longest = current = 0
    previous_step: int | None = None
    for step, active in samples:
        if active and (previous_step is None or step == previous_step + 1):
            current += 1
        elif active:
            current = 1
        else:
            current = 0
        longest = max(longest, current)
        previous_step = step
    return longest


def _contact_count(value: Any, body_a: str, body_b: str, *, min_force: float) -> int:
    target = {body_a, body_b}
    return sum(
        item.get("kind") == "contact"
        and bool(item.get("active", True))
        and {str(item.get("body_a")), str(item.get("body_b"))} == target
        and float(item.get("force", 0.0)) >= min_force
        for item in _events(value)
    )


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    iterator = iter(actual)
    return all(any(item == target for item in iterator) for target in expected)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _compare(actual: float, operator: str, expected: float) -> bool:
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    if operator == "==":
        return math.isclose(actual, expected)
    if operator == ">=":
        return actual >= expected
    if operator == ">":
        return actual > expected
    if operator == "!=":
        return not math.isclose(actual, expected)
    raise ValueError(f"Unsupported operator: {operator}")
