"""Independent analytic evaluator used to acceptance-test SimRig Phase 1."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any, Mapping


EVALUATOR_SPEC = {
    "name": "planar-reach-independent",
    "version": "1.0.0",
    "protocol_version": 1,
    "description": "Analytic two-link reaching evaluator independent of training reward.",
}


def evaluate(request: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a small controller artifact across deterministic reach scenarios."""
    checkpoint = Path(str(request["checkpoint"]))
    controller = json.loads(checkpoint.read_text(encoding="utf-8"))["controller"]
    target = tuple(float(value) for value in request["parameters"]["target"])
    tolerance = float(request["parameters"].get("tolerance", 0.025))
    max_steps = int(request["max_steps"])
    rng = random.Random(int(request["seed"]))
    start = (-0.25 + rng.uniform(-0.03, 0.03), 0.55 + rng.uniform(-0.03, 0.03))
    goal = _inverse_kinematics(target)
    events: list[dict[str, Any]] = []
    errors: list[float] = []
    total_reward = 0.0

    for step in range(max_steps):
        if controller == "valid":
            alpha = min((step + 1) / 18.0, 1.0)
            joints = tuple(start[i] + alpha * (goal[i] - start[i]) for i in range(2))
        elif controller == "reward_trap":
            # Deliberately earns a large proxy reward while moving into a forbidden region.
            alpha = min((step + 1) / 5.0, 1.0)
            joints = (
                start[0] + alpha * (-math.pi / 2 - start[0]),
                start[1] + alpha * (0.0 - start[1]),
            )
        else:
            raise ValueError(f"Unknown example controller: {controller!r}")

        end_effector = _forward_kinematics(joints)
        error = math.dist(end_effector, target)
        errors.append(error)
        events.append(
            {
                "kind": "signal",
                "name": "target_within_tolerance",
                "step": step,
                "active": error <= tolerance,
            }
        )
        if end_effector[1] < -0.25:
            events.append(
                {
                    "kind": "contact",
                    "name": "forbidden_contact",
                    "step": step,
                    "active": True,
                    "body_a": "end_effector",
                    "body_b": "forbidden_zone",
                    "force": 1.0,
                }
            )
        total_reward += 1.0 / (1.0 + error)

    if controller == "reward_trap":
        total_reward += 1_000.0
    contacts = sum(event["kind"] == "contact" for event in events)
    return {
        "total_reward": total_reward,
        "metrics": {
            "final_target_error": errors[-1],
            "minimum_target_error": min(errors),
            "forbidden_contact_count": contacts,
        },
        "events": events,
    }


def _forward_kinematics(joints: tuple[float, float]) -> tuple[float, float]:
    first, second = 0.6, 0.45
    q1, q2 = joints
    return (
        first * math.cos(q1) + second * math.cos(q1 + q2),
        first * math.sin(q1) + second * math.sin(q1 + q2),
    )


def _inverse_kinematics(target: tuple[float, float]) -> tuple[float, float]:
    first, second = 0.6, 0.45
    x, y = target
    cosine = (x * x + y * y - first * first - second * second) / (2 * first * second)
    cosine = max(-1.0, min(1.0, cosine))
    q2 = math.acos(cosine)
    q1 = math.atan2(y, x) - math.atan2(second * math.sin(q2), first + second * math.cos(q2))
    return q1, q2
