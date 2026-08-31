"""Backend-neutral evaluator plugin loading and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from simrig.failures import FailureCategory, terminal_reason
from simrig.predicates import apply_predicates
from simrig.runtime import source_closure_manifest
from simrig.rollout import InvalidRolloutState


EVALUATOR_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class EvaluationRequest:
    checkpoint: str
    environment: str
    backend: str
    suite: str
    scenario: str
    parameters: dict[str, Any]
    seed: int
    max_steps: int
    task_contract_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class EvaluatorPlugin(Protocol):
    """Minimum interface implemented by task-owned independent evaluators."""

    def evaluate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LoadedEvaluator:
    path: Path
    evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    spec: dict[str, Any]
    config: dict[str, Any]
    manifest: dict[str, Any]
    close: Callable[[], None] = lambda: None


def load_evaluator(
    path: Path | str,
    *,
    config: Mapping[str, Any] | None = None,
) -> LoadedEvaluator:
    """Load a Python evaluator without coupling it to a simulator backend."""
    evaluator_path = Path(path).expanduser().resolve()
    if not evaluator_path.is_file() or evaluator_path.suffix != ".py":
        raise FileNotFoundError(f"Evaluator plugin must be a Python file: {evaluator_path}")
    module = _import_module(evaluator_path)
    evaluate = getattr(module, "evaluate", None)
    factory = getattr(module, "make_evaluator", None)
    if not callable(evaluate) and not callable(factory):
        raise AttributeError(f"Evaluator plugin must define evaluate(request) or make_evaluator(config): {evaluator_path}")
    raw_spec = getattr(module, "EVALUATOR_SPEC", None)
    if callable(getattr(module, "evaluator_spec", None)):
        raw_spec = module.evaluator_spec()
    if not isinstance(raw_spec, Mapping):
        raise TypeError("Evaluator plugin must declare EVALUATOR_SPEC or evaluator_spec().")
    spec = dict(raw_spec)
    spec.setdefault("protocol_version", EVALUATOR_PROTOCOL_VERSION)
    if spec.get("protocol_version") != EVALUATOR_PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported evaluator protocol version: {spec.get('protocol_version')!r}"
        )
    for key in ("name", "version"):
        if not isinstance(spec.get(key), str) or not str(spec[key]).strip():
            raise ValueError(f"Evaluator spec {key!r} must be non-empty text.")
    normalized_config = dict(config or {})
    close = lambda: None
    if callable(factory):
        instance = factory(normalized_config)
        evaluate = getattr(instance, "evaluate", None)
        if not callable(evaluate):
            raise TypeError("make_evaluator(config) must return an object with evaluate(request)")
        close = getattr(instance, "close", close)
        if not callable(close):
            raise TypeError("Evaluator close must be callable when present")
    closure = source_closure_manifest(evaluator_path, None)
    portable_closure = [
        {
            "relative_path": item.get("relative_path"),
            "kind": item.get("kind"),
            "sha256": item.get("sha256"),
        }
        for item in closure
    ]
    identity = {
        "protocol_version": EVALUATOR_PROTOCOL_VERSION,
        "spec": spec,
        "config": normalized_config,
        "source_closure": portable_closure,
    }
    evaluator_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "name": spec["name"],
        "version": spec["version"],
        "protocol_version": EVALUATOR_PROTOCOL_VERSION,
        "path": str(evaluator_path),
        "sha256": evaluator_hash,
        "config": normalized_config,
        "source_closure": closure,
    }
    return LoadedEvaluator(
        path=evaluator_path,
        evaluate=evaluate,
        spec=spec,
        config=normalized_config,
        manifest=manifest,
        close=close,
    )


def run_evaluator(
    evaluator: LoadedEvaluator,
    request: EvaluationRequest,
    *,
    predicates: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one matrix cell and normalize errors into stable failure records."""
    payload = request.to_dict()
    payload["evaluator_config"] = evaluator.config
    try:
        raw = evaluator.evaluate(payload)
        if not isinstance(raw, Mapping):
            raise TypeError("evaluate(request) must return a mapping")
        record = _normalize_record(raw, request)
        return apply_predicates(record, predicates)
    except Exception as exc:
        return {
            "checkpoint": request.checkpoint,
            "scenario": request.scenario,
            "seed": request.seed,
            "parameters": dict(request.parameters),
            "metrics": {},
            "events": [],
            "total_reward": None,
            "task_success": False,
            "predicate_results": [],
            "terminal_reason": terminal_reason(
                FailureCategory.INVALID_STATE if isinstance(exc, InvalidRolloutState) else FailureCategory.EVALUATOR_ERROR,
                "invalid_rollout_state" if isinstance(exc, InvalidRolloutState) else "evaluator_exception",
                f"Evaluator raised {type(exc).__name__}: {exc}",
            ),
        }


def resolve_evaluator_path(
    plugin_ref: str,
    *,
    contract_path: Path,
    source_path: Path | None = None,
) -> Path:
    path = Path(plugin_ref).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [contract_path.parent / path]
    if source_path is not None:
        candidates.append(source_path.parent / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _normalize_record(raw: Mapping[str, Any], request: EvaluationRequest) -> dict[str, Any]:
    # Reject invalid JSON numerics before predicates can silently ignore them.
    # Missing measurements remain missing and are reported as insufficient evidence.
    json.dumps(dict(raw), allow_nan=False)
    metrics = raw.get("metrics")
    events = raw.get("events")
    record = dict(raw)
    record.update(
        {
            "checkpoint": request.checkpoint,
            "scenario": request.scenario,
            "seed": request.seed,
            "parameters": dict(request.parameters),
            "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
        }
    )
    if events is not None:
        if not isinstance(events, list) or not all(isinstance(event, Mapping) for event in events):
            raise TypeError("Evaluator events must be a list of mappings")
        signals: set[tuple[str, int]] = set()
        for event in events:
            step = event.get("step")
            if type(step) is not int or not 0 <= step < request.max_steps:
                raise ValueError("Event step must be a control tick inside the requested horizon")
            if "active" in event and type(event["active"]) is not bool:
                raise TypeError("Event active must be a boolean")
            kind = event.get("kind", "event")
            if kind not in {"event", "signal", "contact"}:
                raise ValueError("Unsupported event kind")
            if kind == "contact":
                if not all(isinstance(event.get(key), str) and event[key] for key in ("body_a", "body_b")):
                    raise ValueError("Contact events must identify both bodies")
                if "force" in event and (type(event["force"]) not in (int, float) or event["force"] < 0):
                    raise ValueError("Contact force must be a non-negative finite number")
            elif not isinstance(event.get("name"), str) or not event["name"]:
                raise ValueError("Events/signals must have a name")
            if kind == "signal":
                cell = (event["name"], step)
                if cell in signals:
                    raise ValueError("Duplicate signal samples at one control tick")
                signals.add(cell)
        record["events"] = list(events)
    record["task_contract_sha256"] = request.task_contract_sha256
    record["suite"] = request.suite
    record["environment"] = request.environment
    reward = raw.get("total_reward", raw.get("reward"))
    record["total_reward"] = float(reward) if isinstance(reward, (int, float)) else None
    return record


def _import_module(path: Path) -> ModuleType:
    name = f"simrig_evaluator_{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    # Match normal import semantics so decorators and relative module metadata
    # can resolve the module while its top-level code executes.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
