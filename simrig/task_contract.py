"""Versioned, task-agnostic contracts for robot-learning projects."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simrig.io import save_json, slugify
from simrig.failures import FAILURE_CATEGORIES
from simrig.predicates import validate_predicate


SCHEMA_VERSION = 2
CONTRACT_KIND = "simrig.task-contract"
FROZEN_KIND = "simrig.frozen-task-contract"
_AGGREGATES = {"all", "any", "count", "max", "mean", "min", "rate", "sum"}
_OPERATORS = {"<", "<=", "==", ">=", ">", "!="}
COMPATIBILITY_POLICIES = frozenset(
    {"exact", "training_resume", "checkpoint_evaluation", "result_comparison"}
)
_COMPATIBILITY_FIELDS = {
    "training_resume": (
        "environment",
        "behavior",
        "interfaces",
        "scene",
        "reset",
        "episode",
        "outcomes",
    ),
    "checkpoint_evaluation": ("environment", "interfaces", "scene", "episode"),
    "result_comparison": (
        "environment",
        "behavior",
        "interfaces",
        "scene",
        "reset",
        "episode",
        "outcomes",
    ),
}


class TrainingBudgetExceeded(RuntimeError):
    """Raised at a safe progress boundary when a frozen budget is exhausted."""


class AbortMonitor:
    """Stateful evaluator for generic metric-based abort rules."""

    def __init__(self, rules: list[Mapping[str, Any]] | None = None) -> None:
        self.rules = [dict(rule) for rule in (rules or [])]
        self._counts = [0 for _ in self.rules]

    def observe(
        self,
        *,
        num_steps: int,
        metrics: Mapping[str, Any],
    ) -> str | None:
        for index, rule in enumerate(self.rules):
            after_steps = int(rule.get("after_steps", 0))
            if num_steps < after_steps:
                continue
            metric = str(rule.get("metric"))
            actual = _metric_number(metrics.get(metric))
            if actual is None:
                if bool(rule.get("require_metric", True)):
                    return f"abort rule metric {metric!r} was missing at step {num_steps:,}"
                continue
            operator = str(rule.get("operator", "<="))
            expected = float(rule.get("value"))
            triggered = _compare_number(actual, operator, expected)
            self._counts[index] = self._counts[index] + 1 if triggered else 0
            patience = max(int(rule.get("patience", 1)), 1)
            if self._counts[index] >= patience:
                return (
                    f"abort rule triggered: {metric}={actual:g} {operator} {expected:g} "
                    f"for {self._counts[index]} evaluation(s) at step {num_steps:,}"
                )
        return None


@dataclass(frozen=True)
class ContractValidation:
    """Validation result returned without importing or constructing an env."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sha256: str | None = None


def task_contract_template(env_ref: str, *, name: str | None = None) -> dict[str, Any]:
    """Return an editable contract template with explicit unresolved fields."""
    label = name or slugify(Path(env_ref).stem if env_ref.endswith(".py") else env_ref)
    return {
        "kind": CONTRACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "name": label,
        "environment": {
            "ref": env_ref,
            "backend": "mujoco-playground",
        },
        "behavior": {
            "objective": "TODO: state the observable behavior the robot must perform",
        },
        "interfaces": {
            "actions": "TODO: define action ownership, units, scaling, and control rate",
            "observations": "TODO: separate deployable and privileged observations",
        },
        "scene": {
            "assumptions": [],
            "allowed_contacts": [],
            "forbidden_contacts": [],
        },
        "reset": {
            "training": "TODO: define the training reset distribution",
            "native": "TODO: define predecessor/native transition states",
        },
        "episode": {
            "horizon_steps": 0,
        },
        "outcomes": {
            "success": "TODO: define measurable physical success",
            "failure": "TODO: define failures and terminal reasons",
            "failure_taxonomy": sorted(FAILURE_CATEGORIES),
        },
        "evaluation": {
            "evaluator": {"plugin": None, "config": {}},
            "predicates": [],
            "suites": {
                "nominal": {
                    "scenarios": [{"name": "nominal", "seeds": [0]}],
                    "group_by": ["scenario"],
                    "requirements": [
                        {
                            "metric": "task_success",
                            "aggregate": "rate",
                            "operator": ">=",
                            "value": 0.8,
                        }
                    ],
                }
            }
        },
        "compute": {
            "max_timesteps": 4096,
            "max_wall_time_sec": None,
            "max_gpu_hours": None,
            "gpu_hourly_cost": None,
            "abort_rules": [],
        },
    }


def validate_task_contract(contract: Mapping[str, Any]) -> ContractValidation:
    """Validate the portable contract schema and promotion declarations."""
    errors: list[str] = []
    warnings: list[str] = []
    data = dict(contract)
    if data.get("kind") != CONTRACT_KIND:
        errors.append(f"kind must be {CONTRACT_KIND!r}")
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    _required_text(data, "name", errors)

    environment = _required_mapping(data, "environment", errors)
    _required_text(environment, "ref", errors, prefix="environment.")
    _required_text(environment, "backend", errors, prefix="environment.")

    behavior = _required_mapping(data, "behavior", errors)
    _required_text(behavior, "objective", errors, prefix="behavior.")
    interfaces = _required_mapping(data, "interfaces", errors)
    _required_text(interfaces, "actions", errors, prefix="interfaces.")
    _required_text(interfaces, "observations", errors, prefix="interfaces.")
    reset = _required_mapping(data, "reset", errors)
    _required_text(reset, "training", errors, prefix="reset.")
    _required_text(reset, "native", errors, prefix="reset.")
    outcomes = _required_mapping(data, "outcomes", errors)
    _required_text(outcomes, "success", errors, prefix="outcomes.")
    _required_text(outcomes, "failure", errors, prefix="outcomes.")
    taxonomy = outcomes.get("failure_taxonomy")
    if not isinstance(taxonomy, list):
        errors.append("outcomes.failure_taxonomy must be a list")
    elif len(taxonomy) != len(set(taxonomy)) or set(taxonomy) != FAILURE_CATEGORIES:
        errors.append(
            "outcomes.failure_taxonomy must contain each stable category exactly once: "
            f"{sorted(FAILURE_CATEGORIES)}"
        )

    episode = _required_mapping(data, "episode", errors)
    horizon = episode.get("horizon_steps")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        errors.append("episode.horizon_steps must be a positive integer")

    scene = _required_mapping(data, "scene", errors)
    for key in ("assumptions", "allowed_contacts", "forbidden_contacts"):
        value = scene.get(key)
        if not isinstance(value, list):
            errors.append(f"scene.{key} must be a list")

    evaluation = _required_mapping(data, "evaluation", errors)
    evaluator = evaluation.get("evaluator", {})
    if not isinstance(evaluator, Mapping):
        errors.append("evaluation.evaluator must be an object")
    else:
        plugin = evaluator.get("plugin")
        if plugin is not None and (not isinstance(plugin, str) or not plugin.strip()):
            errors.append("evaluation.evaluator.plugin must be non-empty text or null")
        if not isinstance(evaluator.get("config", {}), Mapping):
            errors.append("evaluation.evaluator.config must be an object")
    _validate_predicates(evaluation.get("predicates", []), "evaluation.predicates", errors)
    suites = _required_mapping(evaluation, "suites", errors, prefix="evaluation.")
    if not suites:
        errors.append("evaluation.suites must declare at least one suite")
    for suite_name, suite_value in suites.items():
        _validate_suite(str(suite_name), suite_value, errors, warnings)

    compute = _required_mapping(data, "compute", errors)
    for key, integer in (
        ("max_timesteps", True),
        ("max_wall_time_sec", False),
        ("max_gpu_hours", False),
        ("gpu_hourly_cost", False),
    ):
        value = compute.get(key)
        if value is None and key != "max_timesteps":
            continue
        valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid_number or value <= 0 or (integer and not isinstance(value, int)):
            kind = "positive integer" if integer else "positive number or null"
            errors.append(f"compute.{key} must be a {kind}")
    abort_rules = compute.get("abort_rules", [])
    if not isinstance(abort_rules, list):
        errors.append("compute.abort_rules must be a list")
    else:
        for index, rule in enumerate(abort_rules):
            _validate_abort_rule(rule, index, errors)

    _find_placeholders(data, "", errors)
    return ContractValidation(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        sha256=contract_sha256(data) if not errors else None,
    )


def freeze_task_contract(
    contract: Mapping[str, Any],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Validate and wrap a contract in a deterministic hash envelope."""
    data = dict(contract)
    result = validate_task_contract(data)
    if not result.passed:
        details = "\n".join(f"- {item}" for item in result.errors)
        raise ValueError(f"Task contract is not freezable:\n{details}")
    return {
        "kind": FROZEN_KIND,
        "schema_version": SCHEMA_VERSION,
        "sha256": result.sha256,
        "source": source,
        "contract": data,
    }


def load_task_contract(
    path: Path | str,
    *,
    require_frozen: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load a draft or frozen contract and verify frozen content hashes."""
    contract_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Task contract must be JSON: {contract_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Task contract root must be an object.")
    if payload.get("kind") == FROZEN_KIND:
        contract = payload.get("contract")
        if not isinstance(contract, dict):
            raise ValueError("Frozen task contract is missing its contract object.")
        expected = payload.get("sha256")
        actual = contract_sha256(contract)
        if expected != actual:
            raise ValueError(
                f"Frozen task contract hash mismatch: recorded={expected} actual={actual}"
            )
        return contract, payload
    if require_frozen:
        raise ValueError("A frozen task contract is required for this operation.")
    return payload, None


def save_frozen_task_contract(
    source_path: Path | str,
    *,
    output: Path | str | None = None,
) -> Path:
    """Freeze a draft file and save it beside the source by default."""
    source = Path(source_path).expanduser().resolve()
    contract, envelope = load_task_contract(source)
    if envelope is not None:
        raise ValueError("Task contract is already frozen.")
    frozen = freeze_task_contract(contract, source=str(source))
    destination = (
        Path(output).expanduser()
        if output is not None
        else source.with_name(f"{source.stem}.frozen.json")
    )
    return save_json(destination, frozen)


def migrate_task_contract(
    contract: Mapping[str, Any],
    *,
    target_version: int = SCHEMA_VERSION,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Migrate an editable contract explicitly; frozen inputs must be re-frozen."""
    data = json.loads(json.dumps(dict(contract)))
    version = data.get("schema_version")
    if not isinstance(version, int):
        raise ValueError("Task contract schema_version must be an integer before migration.")
    if version > target_version:
        raise ValueError(
            f"Cannot migrate schema {version} down to older schema {target_version}."
        )
    steps: list[dict[str, Any]] = []
    while version < target_version:
        if version == 1:
            evaluation = data.setdefault("evaluation", {})
            evaluation.setdefault("evaluator", {"plugin": None, "config": {}})
            evaluation.setdefault("predicates", [])
            outcomes = data.setdefault("outcomes", {})
            outcomes.setdefault("failure_taxonomy", sorted(FAILURE_CATEGORIES))
            data["schema_version"] = 2
            steps.append(
                {
                    "from": 1,
                    "to": 2,
                    "changes": [
                        "add evaluation.evaluator declaration",
                        "add evaluation.predicates",
                        "add stable outcomes.failure_taxonomy",
                    ],
                }
            )
            version = 2
            continue
        raise ValueError(f"No migration is defined from task contract schema {version}.")
    return data, steps


def save_migrated_task_contract(
    source_path: Path | str,
    *,
    output: Path | str,
) -> tuple[Path, list[dict[str, Any]]]:
    source = Path(source_path).expanduser().resolve()
    contract, _ = load_task_contract(source)
    migrated, steps = migrate_task_contract(contract)
    if not steps:
        raise ValueError(f"Task contract already uses schema version {SCHEMA_VERSION}.")
    return save_json(Path(output).expanduser(), migrated), steps


def contract_sha256(contract: Mapping[str, Any]) -> str:
    """Hash canonical JSON so formatting and key order do not affect identity."""
    encoded = json.dumps(
        dict(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def diff_task_contracts(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic leaf-level differences between two contracts."""
    changes: list[dict[str, Any]] = []
    _diff_values(dict(left), dict(right), "", changes)
    return changes


def compare_task_contracts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    policy: str,
) -> dict[str, Any]:
    """Apply an explicit compatibility policy; never silently migrate semantics."""
    if policy not in COMPATIBILITY_POLICIES:
        raise ValueError(
            f"Unknown compatibility policy {policy!r}; expected one of "
            f"{sorted(COMPATIBILITY_POLICIES)}"
        )
    changes = diff_task_contracts(left, right)
    if policy == "exact":
        incompatible = list(changes)
    else:
        prefixes = _COMPATIBILITY_FIELDS[policy]
        incompatible = [
            change
            for change in changes
            if any(
                change["path"] == prefix or change["path"].startswith(prefix + ".")
                for prefix in prefixes
            )
        ]
    return {
        "policy": policy,
        "compatible": not incompatible,
        "left_sha256": contract_sha256(left),
        "right_sha256": contract_sha256(right),
        "changes": changes,
        "incompatible_changes": incompatible,
    }


def enforce_compute_budget(
    contract: Mapping[str, Any],
    *,
    requested_timesteps: int,
    estimated_timesteps: int,
) -> None:
    """Reject a run whose resolved rollout quantum exceeds the frozen budget."""
    compute = contract.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("Task contract compute section is missing.")
    limit = compute.get("max_timesteps")
    if isinstance(limit, int) and estimated_timesteps > limit:
        raise ValueError(
            "Resolved training budget exceeds task contract: "
            f"requested={requested_timesteps:,} estimated={estimated_timesteps:,} "
            f"max={limit:,}. Update and re-freeze the contract intentionally."
        )


def enforce_runtime_budget(
    contract: Mapping[str, Any],
    *,
    elapsed_sec: float,
    gpu_count: int,
) -> None:
    """Raise at a progress boundary when wall-time or GPU-hour caps are exceeded."""
    compute = contract.get("compute")
    if not isinstance(compute, Mapping):
        return
    wall_limit = compute.get("max_wall_time_sec")
    if isinstance(wall_limit, (int, float)) and elapsed_sec >= float(wall_limit):
        raise TrainingBudgetExceeded(
            f"Wall-time budget exhausted: elapsed={elapsed_sec:.1f}s max={float(wall_limit):.1f}s"
        )
    gpu_limit = compute.get("max_gpu_hours")
    gpu_hours = elapsed_sec * max(gpu_count, 0) / 3600.0
    if isinstance(gpu_limit, (int, float)) and gpu_hours >= float(gpu_limit):
        raise TrainingBudgetExceeded(
            f"GPU-hour budget exhausted: used={gpu_hours:.3f} max={float(gpu_limit):.3f}"
        )


def _validate_suite(
    name: str,
    value: Any,
    errors: list[str],
    warnings: list[str],
) -> None:
    prefix = f"evaluation.suites.{name}"
    if not isinstance(value, Mapping):
        errors.append(f"{prefix} must be an object")
        return
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append(f"{prefix}.scenarios must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, scenario in enumerate(scenarios):
            item = f"{prefix}.scenarios[{index}]"
            if not isinstance(scenario, Mapping):
                errors.append(f"{item} must be an object")
                continue
            scenario_name = scenario.get("name")
            if not isinstance(scenario_name, str) or not scenario_name.strip():
                errors.append(f"{item}.name must be non-empty text")
            elif scenario_name in seen:
                errors.append(f"{item}.name duplicates {scenario_name!r}")
            else:
                seen.add(scenario_name)
            seeds = scenario.get("seeds")
            if not isinstance(seeds, list) or not seeds or not all(
                isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
            ):
                errors.append(f"{item}.seeds must be a non-empty integer list")
            elif len(seeds) != len(set(seeds)):
                errors.append(f"{item}.seeds must not contain duplicates")
            if not isinstance(scenario.get("parameters", {}), Mapping):
                errors.append(f"{item}.parameters must be an object")
            max_steps = scenario.get("max_steps")
            if max_steps is not None and (
                not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1
            ):
                errors.append(f"{item}.max_steps must be a positive integer when present")
            _validate_predicates(scenario.get("predicates", []), f"{item}.predicates", errors)

    _validate_predicates(value.get("predicates", []), f"{prefix}.predicates", errors)

    group_by = value.get("group_by", [])
    if not isinstance(group_by, list) or not all(isinstance(key, str) for key in group_by):
        errors.append(f"{prefix}.group_by must be a list of field names")

    requirements = value.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        errors.append(f"{prefix}.requirements must be a non-empty list")
        return
    for index, requirement in enumerate(requirements):
        item = f"{prefix}.requirements[{index}]"
        if not isinstance(requirement, Mapping):
            errors.append(f"{item} must be an object")
            continue
        metric = requirement.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            errors.append(f"{item}.metric must be non-empty text")
        aggregate = requirement.get("aggregate", "mean")
        if aggregate not in _AGGREGATES:
            errors.append(f"{item}.aggregate must be one of {sorted(_AGGREGATES)}")
        operator = requirement.get("operator", ">=")
        if operator not in _OPERATORS:
            errors.append(f"{item}.operator must be one of {sorted(_OPERATORS)}")
        threshold = requirement.get("value")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            errors.append(f"{item}.value must be numeric")
        where = requirement.get("where", {})
        if not isinstance(where, Mapping):
            errors.append(f"{item}.where must be an object when present")
    if "reward" in {str(item.get("metric")) for item in requirements if isinstance(item, Mapping)}:
        warnings.append(f"{prefix} gates directly on reward; prefer an outcome metric")


def _validate_predicates(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list")
        return
    seen: set[str] = set()
    for index, predicate in enumerate(value):
        item = f"{path}[{index}]"
        if not isinstance(predicate, Mapping):
            errors.append(f"{item} must be an object")
            continue
        errors.extend(validate_predicate(predicate, path=item))
        predicate_id = predicate.get("id")
        if isinstance(predicate_id, str):
            if predicate_id in seen:
                errors.append(f"{item}.id duplicates {predicate_id!r}")
            seen.add(predicate_id)


def _validate_abort_rule(value: Any, index: int, errors: list[str]) -> None:
    prefix = f"compute.abort_rules[{index}]"
    if not isinstance(value, Mapping):
        errors.append(f"{prefix} must be an object")
        return
    metric = value.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        errors.append(f"{prefix}.metric must be non-empty text")
    operator = value.get("operator", "<=")
    if operator not in _OPERATORS:
        errors.append(f"{prefix}.operator must be one of {sorted(_OPERATORS)}")
    threshold = value.get("value")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        errors.append(f"{prefix}.value must be numeric")
    for key, minimum in (("after_steps", 0), ("patience", 1)):
        item = value.get(key, minimum)
        if not isinstance(item, int) or isinstance(item, bool) or item < minimum:
            errors.append(f"{prefix}.{key} must be an integer >= {minimum}")


def _metric_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (TypeError, ValueError):
            return None
    return None


def _compare_number(actual: float, operator: str, expected: float) -> bool:
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    if operator == "==":
        return actual == expected
    if operator == ">=":
        return actual >= expected
    if operator == ">":
        return actual > expected
    if operator == "!=":
        return actual != expected
    raise ValueError(f"Unsupported operator: {operator}")


def _required_mapping(
    value: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str = "",
) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        errors.append(f"{prefix}{key} must be an object")
        return {}
    return item


def _required_text(
    value: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    prefix: str = "",
) -> None:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        errors.append(f"{prefix}{key} must be non-empty text")


def _find_placeholders(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            _find_placeholders(item, child, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _find_placeholders(item, f"{path}[{index}]", errors)
    elif isinstance(value, str) and value.strip().upper().startswith("TODO"):
        errors.append(f"{path} still contains a TODO placeholder")


def _diff_values(left: Any, right: Any, path: str, changes: list[dict[str, Any]]) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                changes.append({"path": child, "left": None, "right": right[key]})
            elif key not in right:
                changes.append({"path": child, "left": left[key], "right": None})
            else:
                _diff_values(left[key], right[key], child, changes)
        return
    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            changes.append({"path": path, "left": left, "right": right})
        return
    if left != right:
        changes.append({"path": path, "left": left, "right": right})
