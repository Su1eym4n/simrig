"""Scenario-by-seed execution for independent evaluator plugins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from simrig.evaluator import (
    EvaluationRequest,
    load_evaluator,
    resolve_evaluator_path,
    run_evaluator,
)
from simrig.failures import SAFETY_FAILURE_CATEGORIES
from simrig.gates import adversarial_reward_probes, evaluate_gate
from simrig.task_contract import load_task_contract
from simrig.io import save_json, slugify
from simrig.run_manifest import record_independent_evaluation
from simrig.task_contract import validate_task_contract


EVALUATION_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationLimits:
    """Hard caps for bounded checkpoint evaluation, including live-run checks."""

    max_scenarios: int | None = None
    max_seeds_per_scenario: int | None = None
    max_evaluations: int | None = None

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")


def run_evaluation_suite(
    checkpoint: Path | str,
    *,
    contract_path: Path | str,
    suite_name: str,
    evaluator_path: Path | str | None = None,
    limits: EvaluationLimits | None = None,
) -> dict[str, Any]:
    """Execute a frozen contract's complete or explicitly bounded matrix."""
    frozen_path = Path(contract_path).expanduser().resolve()
    contract, envelope = load_task_contract(frozen_path, require_frozen=True)
    assert envelope is not None
    validation = validate_task_contract(contract)
    if not validation.passed:
        details = "\n".join(f"- {item}" for item in validation.errors)
        raise ValueError(
            "Evaluation requires a current valid task contract. Migrate and re-freeze it first:\n"
            + details
        )
    evaluation = contract.get("evaluation")
    suites = evaluation.get("suites") if isinstance(evaluation, Mapping) else None
    suite = suites.get(suite_name) if isinstance(suites, Mapping) else None
    if not isinstance(suite, Mapping):
        raise ValueError(f"Task contract has no evaluation suite {suite_name!r}.")

    evaluator_decl = evaluation.get("evaluator") if isinstance(evaluation, Mapping) else None
    evaluator_decl = evaluator_decl if isinstance(evaluator_decl, Mapping) else {}
    plugin_ref = str(evaluator_path or evaluator_decl.get("plugin") or "")
    if not plugin_ref:
        raise ValueError(
            "No evaluator plugin was provided. Declare evaluation.evaluator.plugin "
            "or pass --evaluator."
        )
    source_path = (
        Path(str(envelope["source"])).expanduser()
        if envelope.get("source")
        else None
    )
    plugin_path = resolve_evaluator_path(
        plugin_ref,
        contract_path=frozen_path,
        source_path=source_path,
    )
    evaluator = load_evaluator(
        plugin_path,
        config=evaluator_decl.get("config")
        if isinstance(evaluator_decl.get("config"), Mapping)
        else None,
    )
    active_limits = limits or EvaluationLimits()
    active_limits.validate()
    scenarios = list(suite.get("scenarios") or [])
    if active_limits.max_scenarios is not None:
        scenarios = scenarios[: active_limits.max_scenarios]
    environment = contract.get("environment") or {}
    episode = contract.get("episode") or {}
    common_predicates = _predicate_list(
        evaluation.get("predicates") if isinstance(evaluation, Mapping) else None
    )
    suite_predicates = _predicate_list(suite.get("predicates"))
    checkpoint_ref = _checkpoint_ref(checkpoint)
    records: list[dict[str, Any]] = []
    stopped_by_limit = False
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            continue
        seeds = list(scenario.get("seeds") or [])
        if active_limits.max_seeds_per_scenario is not None:
            seeds = seeds[: active_limits.max_seeds_per_scenario]
        for seed in seeds:
            if (
                active_limits.max_evaluations is not None
                and len(records) >= active_limits.max_evaluations
            ):
                stopped_by_limit = True
                break
            request = EvaluationRequest(
                checkpoint=checkpoint_ref,
                environment=str(environment.get("ref")),
                backend=str(environment.get("backend")),
                suite=suite_name,
                scenario=str(scenario.get("name")),
                parameters=dict(scenario.get("parameters") or {}),
                seed=int(seed),
                max_steps=int(
                    scenario.get("max_steps")
                    or suite.get("max_steps")
                    or episode.get("horizon_steps")
                ),
                task_contract_sha256=str(envelope["sha256"]),
            )
            predicates = (
                common_predicates
                + suite_predicates
                + _predicate_list(scenario.get("predicates"))
            )
            records.append(run_evaluator(evaluator, request, predicates=predicates))
        if stopped_by_limit:
            break

    gate = evaluate_gate(contract, suite_name=suite_name, records=records)
    reward_probes = adversarial_reward_probes(records)
    report = {
        "kind": "simrig.evaluation-suite-report",
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "task_contract": {
            "path": str(frozen_path),
            "sha256": envelope["sha256"],
            "schema_version": contract.get("schema_version"),
        },
        "evaluator": evaluator.manifest,
        "checkpoint": {
            "path": checkpoint_ref,
            "sha256": artifact_sha256(checkpoint),
        },
        "suite": suite_name,
        "bounded": any(value is not None for value in active_limits.__dict__.values()),
        "limits": dict(active_limits.__dict__),
        "records": records,
        "conditions": condition_summaries(records, gate=gate),
        "gate": gate,
        "reward_probes": reward_probes,
        "passed": bool(gate["passed"]),
    }
    report["report_sha256"] = _mapping_sha256(report)
    return report


def evaluate_checkpoint_directory(
    run_dir: Path | str,
    *,
    contract_path: Path | str,
    suite_name: str,
    evaluator_path: Path | str | None = None,
    max_checkpoints: int = 1,
    limits: EvaluationLimits | None = None,
) -> dict[str, Any]:
    """Bounded, one-shot evaluation of checkpoints visible in a live run directory."""
    if max_checkpoints < 1:
        raise ValueError("max_checkpoints must be a positive integer")
    root = Path(run_dir).expanduser().resolve()
    checkpoint_dir = root / "checkpoints"
    candidates: list[Path] = []
    if checkpoint_dir.is_dir():
        candidates.extend(
            sorted(
                (path for path in checkpoint_dir.iterdir() if path.is_dir()),
                key=_checkpoint_sort_key,
            )
        )
    if (root / "policy.params").is_file():
        candidates.append(root / "policy.params")
    if not candidates:
        raise FileNotFoundError(f"No checkpoints were found in run directory: {root}")
    selected = candidates[-max_checkpoints:]
    output_dir = root / "independent_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    report_paths: list[str] = []
    for checkpoint in selected:
        report = run_evaluation_suite(
            checkpoint,
            contract_path=contract_path,
            suite_name=suite_name,
            evaluator_path=evaluator_path,
            limits=limits or EvaluationLimits(max_scenarios=1, max_seeds_per_scenario=1),
        )
        path = output_dir / f"{slugify(checkpoint.name)}-{slugify(suite_name)}.json"
        save_json(path, report)
        reports.append(report)
        report_paths.append(str(path))
    if reports:
        record_independent_evaluation(
            root,
            evaluator=reports[0]["evaluator"],
            report_paths=report_paths,
            bounded=True,
        )
    return {
        "run_dir": str(root),
        "selected_checkpoints": [str(path) for path in selected],
        "reports": report_paths,
        "evaluator": reports[0]["evaluator"] if reports else None,
        "bounded": True,
    }


def condition_summaries(
    records: list[Mapping[str, Any]],
    *,
    gate: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("scenario")), []).append(record)
    summaries: list[dict[str, Any]] = []
    for condition in sorted(groups):
        items = groups[condition]
        successes = sum(record.get("task_success") is True for record in items)
        categories: dict[str, int] = {}
        for record in items:
            reason = record.get("terminal_reason")
            category = str(reason.get("category")) if isinstance(reason, Mapping) else "unknown"
            categories[category] = categories.get(category, 0) + 1
        group_by = list((gate or {}).get("group_by") or [])
        scenario_group_index = group_by.index("scenario") if "scenario" in group_by else None
        requirements = [
            item
            for item in (gate or {}).get("requirements", [])
            if isinstance(item, Mapping)
            and item.get("group")
            and scenario_group_index is not None
            and len(item["group"]) > scenario_group_index
            and str(item["group"][scenario_group_index]) == condition
        ]
        missing = [
            item
            for item in ((gate or {}).get("coverage") or {}).get("missing", [])
            if isinstance(item, Mapping) and str(item.get("scenario")) == condition
        ]
        promotion_passed = bool(requirements) and not missing and all(
            item.get("passed") is True for item in requirements
        )
        summaries.append(
            {
                "condition": condition,
                "episodes": len(items),
                "successes": successes,
                "success_rate": successes / len(items) if items else 0.0,
                "safety_failures": sum(
                    count for category, count in categories.items()
                    if category in SAFETY_FAILURE_CATEGORIES
                ),
                "failure_categories": categories,
                "all_successful": bool(items) and successes == len(items),
                "promotion_passed": promotion_passed,
                "missing_coverage": missing,
            }
        )
    return summaries


def artifact_sha256(path: Path | str, *, max_files: int = 512) -> str:
    """Hash a checkpoint file or bounded directory tree deterministically."""
    target = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
        return digest.hexdigest()
    if not target.is_dir():
        digest.update(str(target).encode("utf-8"))
        return digest.hexdigest()
    files = sorted(item for item in target.rglob("*") if item.is_file())[:max_files]
    for item in files:
        digest.update(str(item.relative_to(target)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_ref(value: Path | str) -> str:
    text = str(value)
    if "://" in text:
        return text
    path = Path(text).expanduser()
    return str(path.resolve()) if path.exists() else text


def _checkpoint_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def _predicate_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
