"""Independent, task-agnostic promotion gates over evaluation artifacts."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_evaluation_records(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Load records from JSON reports without importing task environments."""
    records: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Evaluation report must be JSON: {path}: {exc}") from exc
        items: Any = payload
        if isinstance(payload, Mapping):
            for key in ("episodes", "records", "results"):
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break
        if isinstance(items, Mapping):
            items = [items]
        if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
            raise ValueError(f"Evaluation report must contain an object or object list: {path}")
        for item in items:
            record = dict(item)
            record.setdefault("source_report", str(path))
            records.append(record)
    return records


def evaluate_gate(
    contract: Mapping[str, Any],
    *,
    suite_name: str,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one contract suite against independently produced records."""
    suites = ((contract.get("evaluation") or {}).get("suites") or {})
    suite = suites.get(suite_name) if isinstance(suites, Mapping) else None
    if not isinstance(suite, Mapping):
        raise ValueError(f"Task contract has no evaluation suite {suite_name!r}.")
    normalized = [_normalize_record(record) for record in records]
    scenarios = suite.get("scenarios") or []
    scenario_names = [
        str(item.get("name")) for item in scenarios if isinstance(item, Mapping)
    ]
    if len(scenario_names) == 1:
        for record in normalized:
            record.setdefault("scenario", scenario_names[0])

    coverage = _scenario_coverage(scenarios, normalized)
    group_keys = suite.get("group_by") or []
    groups = _group_records(normalized, group_keys)
    requirement_results: list[dict[str, Any]] = []
    for requirement in suite.get("requirements") or []:
        for group, group_records in groups.items():
            requirement_results.append(
                _evaluate_requirement(requirement, group, group_records)
            )
    passed = bool(normalized) and coverage["passed"] and all(
        item["passed"] for item in requirement_results
    )
    return {
        "suite": suite_name,
        "passed": passed,
        "records": len(normalized),
        "group_by": list(group_keys),
        "coverage": coverage,
        "requirements": requirement_results,
    }


def audit_reward_alignment(
    records: Iterable[Mapping[str, Any]],
    *,
    reward_metric: str = "total_reward",
    success_metric: str = "task_success",
) -> dict[str, Any]:
    """Flag obvious disagreement between reward and independent success."""
    pairs: list[tuple[float, float]] = []
    for raw in records:
        record = _normalize_record(raw)
        reward = _numeric(record.get(reward_metric))
        success = _numeric(record.get(success_metric))
        if reward is not None and success is not None:
            pairs.append((reward, success))
    flags: list[str] = []
    successes = [reward for reward, success in pairs if success >= 0.5]
    failures = [reward for reward, success in pairs if success < 0.5]
    if not pairs:
        flags.append("No records contained both reward and independent success metrics.")
    if successes and failures and _mean(failures) >= _mean(successes):
        flags.append("Failed episodes have mean reward greater than or equal to successful episodes.")
    correlation = _pearson([item[0] for item in pairs], [item[1] for item in pairs])
    if correlation is not None and correlation <= 0.1:
        flags.append("Reward has weak or negative correlation with independent success.")
    return {
        "passed": not flags,
        "records": len(pairs),
        "reward_metric": reward_metric,
        "success_metric": success_metric,
        "successful_mean_reward": _mean(successes) if successes else None,
        "failed_mean_reward": _mean(failures) if failures else None,
        "reward_success_correlation": correlation,
        "flags": flags,
    }


def adversarial_reward_probes(
    records: Iterable[Mapping[str, Any]],
    *,
    reward_metric: str = "total_reward",
    success_metric: str = "task_success",
) -> dict[str, Any]:
    """Find concrete high-reward failures without treating reward as success."""
    normalized: list[dict[str, Any]] = []
    for raw in records:
        record = _normalize_record(raw)
        reward = _numeric(record.get(reward_metric))
        success = _numeric(record.get(success_metric))
        if reward is not None and success is not None:
            normalized.append({**record, "_reward": reward, "_success": success >= 0.5})
    successes = [item for item in normalized if item["_success"]]
    failures = [item for item in normalized if not item["_success"]]
    flags: list[str] = []
    cases: list[dict[str, Any]] = []
    if not successes:
        flags.append("No independently successful records were available as a reward baseline.")
    if successes:
        success_floor = min(item["_reward"] for item in successes)
        for failure in failures:
            if failure["_reward"] >= success_floor:
                terminal = failure.get("terminal_reason")
                category = terminal.get("category") if isinstance(terminal, Mapping) else None
                cases.append(
                    {
                        "checkpoint": failure.get("checkpoint"),
                        "scenario": failure.get("scenario"),
                        "seed": failure.get("seed"),
                        "reward": failure["_reward"],
                        "successful_reward_floor": success_floor,
                        "failure_category": category,
                    }
                )
        if cases:
            flags.append(
                f"Found {len(cases)} unsuccessful rollout(s) whose reward met or exceeded "
                "the lowest independently successful reward."
            )
    alignment = audit_reward_alignment(
        normalized,
        reward_metric="_reward",
        success_metric="_success",
    )
    for flag in alignment["flags"]:
        if flag not in flags:
            flags.append(flag)
    return {
        "passed": not flags,
        "records": len(normalized),
        "successful_records": len(successes),
        "failed_records": len(failures),
        "reward_metric": reward_metric,
        "success_metric": success_metric,
        "high_reward_failures": cases,
        "alignment": alignment,
        "flags": flags,
    }


def _normalize_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        for key, item in metrics.items():
            record.setdefault(str(key), item)
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping):
        for key, item in outcome.items():
            record.setdefault(str(key), item)
    return record


def _scenario_coverage(
    scenarios: Any,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected: set[tuple[str, int]] = set()
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, Mapping):
                continue
            name = str(scenario.get("name"))
            for seed in scenario.get("seeds") or []:
                expected.add((name, int(seed)))
    observed = {
        (str(record.get("scenario")), int(record["seed"]))
        for record in records
        if record.get("scenario") is not None and isinstance(record.get("seed"), int)
    }
    missing = sorted(expected - observed)
    return {
        "passed": not missing,
        "expected": len(expected),
        "observed": len(expected & observed),
        "missing": [{"scenario": name, "seed": seed} for name, seed in missing],
    }


def _group_records(
    records: list[dict[str, Any]],
    keys: list[str],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    if not keys:
        return {(): records}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(key) for key in keys)].append(record)
    return dict(groups)


def _evaluate_requirement(
    requirement: Mapping[str, Any],
    group: tuple[Any, ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metric = str(requirement.get("metric"))
    where = requirement.get("where") or {}
    filtered = [
        record for record in records
        if all(record.get(key) == value for key, value in where.items())
    ]
    values = [item for item in (_numeric(record.get(metric)) for record in filtered) if item is not None]
    aggregate = str(requirement.get("aggregate", "mean"))
    actual = _aggregate(values, aggregate)
    operator = str(requirement.get("operator", ">="))
    expected = float(requirement.get("value"))
    passed = actual is not None and _compare(actual, operator, expected)
    return {
        "metric": metric,
        "group": list(group),
        "where": dict(where),
        "aggregate": aggregate,
        "operator": operator,
        "expected": expected,
        "actual": actual,
        "samples": len(values),
        "passed": passed,
        "reason": (
            f"{aggregate}({metric})={actual:g} {operator} {expected:g}"
            if actual is not None
            else f"metric {metric!r} had no numeric samples"
        ),
    }


def _aggregate(values: list[float], mode: str) -> float | None:
    if not values:
        return None
    if mode == "all":
        return float(all(value >= 0.5 for value in values))
    if mode == "any":
        return float(any(value >= 0.5 for value in values))
    if mode == "count":
        return float(len(values))
    if mode == "max":
        return max(values)
    if mode == "min":
        return min(values)
    if mode == "rate":
        return sum(value >= 0.5 for value in values) / len(values)
    if mode == "sum":
        return sum(values)
    return _mean(values)


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


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)
