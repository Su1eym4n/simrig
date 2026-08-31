"""Independent checkpoint ranking from comparable evaluation-suite reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from simrig.failures import SAFETY_FAILURE_CATEGORIES


def load_suite_reports(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Evaluation-suite report must be JSON: {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("kind") != "simrig.evaluation-suite-report":
            raise ValueError(f"Not a SimRig evaluation-suite report: {path}")
        recorded_hash = payload.get("report_sha256")
        if recorded_hash is not None:
            canonical = dict(payload)
            canonical.pop("report_sha256", None)
            actual_hash = _mapping_sha256(canonical)
            if recorded_hash != actual_hash:
                raise ValueError(
                    f"Evaluation-suite report hash mismatch: {path}: "
                    f"recorded={recorded_hash} actual={actual_hash}"
                )
        payload.setdefault("source_report", str(path))
        reports.append(payload)
    return reports


def rank_checkpoints(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank only comparable reports; reward is explicitly excluded."""
    items = [dict(report) for report in reports]
    if not items:
        raise ValueError("At least one evaluation-suite report is required.")
    identities = {
        (
            _nested(report, "task_contract", "sha256"),
            _nested(report, "evaluator", "sha256"),
            report.get("suite"),
            report.get("schema_version"),
        )
        for report in items
    }
    if len(identities) != 1:
        raise ValueError(
            "Checkpoint reports are not comparable: contract hash, evaluator hash, "
            "suite, and report schema must match."
        )
    scored = [_score_report(report) for report in items]
    scored.sort(
        key=lambda item: (
            -int(item["promotion_passed"]),
            -item["worst_condition_success_rate"],
            -item["success_rate"],
            item["safety_failure_rate"],
            str(item["checkpoint"]),
        )
    )
    for index, item in enumerate(scored, start=1):
        item["rank"] = index
    identity = next(iter(identities))
    return {
        "kind": "simrig.checkpoint-ranking",
        "schema_version": 1,
        "task_contract_sha256": identity[0],
        "evaluator_sha256": identity[1],
        "suite": identity[2],
        "ranking_basis": [
            "promotion_passed",
            "worst_condition_success_rate",
            "overall_success_rate",
            "safety_failure_rate",
        ],
        "reward_used_for_ranking": False,
        "checkpoints": scored,
    }


def _score_report(report: Mapping[str, Any]) -> dict[str, Any]:
    records = report.get("records")
    records = records if isinstance(records, list) else []
    successes = sum(
        isinstance(record, Mapping) and record.get("task_success") is True
        for record in records
    )
    safety_failures = 0
    for record in records:
        if not isinstance(record, Mapping):
            continue
        reason = record.get("terminal_reason")
        category = reason.get("category") if isinstance(reason, Mapping) else None
        safety_failures += category in SAFETY_FAILURE_CATEGORIES
    conditions = report.get("conditions")
    rates = [
        float(item.get("success_rate", 0.0))
        for item in conditions or []
        if isinstance(item, Mapping)
    ]
    checkpoint = report.get("checkpoint")
    checkpoint_path = checkpoint.get("path") if isinstance(checkpoint, Mapping) else None
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None,
        "source_report": report.get("source_report"),
        "promotion_passed": report.get("passed") is True,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "worst_condition_success_rate": min(rates) if rates else 0.0,
        "safety_failures": safety_failures,
        "safety_failure_rate": safety_failures / len(records) if records else 1.0,
    }


def _nested(value: Mapping[str, Any], outer: str, inner: str) -> Any:
    child = value.get(outer)
    return child.get(inner) if isinstance(child, Mapping) else None


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
