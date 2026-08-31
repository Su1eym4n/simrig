"""Reusable Playground rollout adapter for task-owned physical measurements."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from simrig.rollout import PolicyRuntime, validate_state


class Measurements(Protocol):
    """One fresh observer per episode; never infer success from training reward.

    reset applies explicit scenario parameters to the reset state, including
    rebuilding observations if a target/command changes. observe measures the
    resulting physical state after each control step (steps are zero-indexed).
    result returns raw metrics/events and evidence coverage, not a success label.
    """

    def reset(self, state: Any) -> Any: ...
    def observe(self, state: Any, step: int) -> None: ...
    def result(self) -> Mapping[str, Any]: ...


class PlaygroundEvaluator:
    """Cache compiled runtime across a suite, while isolating episode observers.

    A .py input explicitly means a trusted scripted control module exposing
    make_controller(env) -> callable(state, rng). Other inputs are learned
    parameters, a SimRig run directory, or an Orbax checkpoint directory.
    """

    def __init__(self, make_measurements: Callable[[Any, Mapping[str, Any]], Measurements]):
        self.make_measurements = make_measurements
        self._runtime: PolicyRuntime | None = None
        self._identity: tuple[str, str, str] | None = None

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        identity = tuple(str(request[key]) for key in ("checkpoint", "environment", "backend"))
        max_steps = request["max_steps"]
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if identity != self._identity:
            self.close()
            checkpoint, environment, backend = identity
            if Path(checkpoint).suffix == ".py":
                from simrig.custom_env import import_env_module

                module = import_env_module(checkpoint)
                factory = getattr(module, "make_controller", None)
                if not callable(factory):
                    raise ValueError("A scripted control must define make_controller(env)")
                runtime = PolicyRuntime(
                    None, env_name=environment, backend=backend, controller_factory=factory,
                )
            else:
                runtime = PolicyRuntime(checkpoint, env_name=environment, backend=backend)
                if not runtime.config or runtime.runtime_compatibility.get("compatible") is not True:
                    runtime.close()
                    raise ValueError("Independent learned-policy evaluation requires recorded, compatible runtime metadata")
            self._runtime = runtime
            self._identity = identity
        runtime = self._runtime
        assert runtime is not None
        state, rng = runtime.reset(request["seed"])
        observer = self.make_measurements(runtime.env, request)
        state = observer.reset(state)
        validate_state(state, runtime.env.observation_size)
        reward = 0.0
        completed = 0
        for tick in range(max_steps):
            state, rng, _ = runtime.advance(state, rng)
            completed = tick + 1
            observer.observe(state, tick)
            reward += float(state.reward)
            if bool(state.done):
                break
        raw = dict(observer.result())
        if "task_success" in raw:
            raise ValueError("Measurements must return physical evidence, not task_success")
        raw.update(
            total_reward=reward,
            steps_completed=completed,
            steps_requested=max_steps,
            terminated=bool(state.done),
            runtime_compatibility=runtime.runtime_compatibility,
            artifact_type="scripted_controller" if runtime.checkpoint is None else "learned_policy",
        )
        return raw

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        self._identity = None
