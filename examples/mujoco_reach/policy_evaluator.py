"""Independent native-MuJoCo measurements for the learned orbit-arm task."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import mujoco
import numpy as np

from simrig.playground_evaluator import PlaygroundEvaluator


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
from targets import target_for_case


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "orbit_arm.xml"
EVALUATOR_SPEC = {
    "name": "orbit-arm-sustained-reach",
    "version": "1.0.0",
    "protocol_version": 1,
    "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
}


def make_evaluator(config):
    config = dict(config or {})
    return PlaygroundEvaluator(
        lambda env, request: OrbitReachMeasurements(env, request, config)
    )


class OrbitReachMeasurements:
    def __init__(self, env, request, config):
        if Path(env.xml_path).resolve() != MODEL_PATH.resolve():
            raise ValueError("This evaluator measures only orbit_arm.xml")
        if set(request["parameters"]) != {"target_set"}:
            raise ValueError("Each scenario must declare exactly one target_set")
        self.env = env
        self.target = target_for_case(
            config, str(request["parameters"]["target_set"]), int(request["seed"])
        )
        self.native = mujoco.MjData(env.mj_model)
        self.ee_site = env.mj_model.site("ee_site").id
        self.errors = []
        self.events = []
        self.max_hold = 0
        self.current_hold = 0
        self.collision_count = 0
        self.contact_pairs = (
            ("floor", "link1"),
            ("floor", "link2"),
            ("base", "link1"),
            ("base", "link2"),
        )

    def reset(self, state):
        return self.env.set_target(state, self.target, reset_counters=True)

    def observe(self, state, step):
        self.native.qpos[:] = np.asarray(state.data.qpos)
        self.native.qvel[:] = np.asarray(state.data.qvel)
        self.native.mocap_pos[:] = np.asarray(state.data.mocap_pos)
        mujoco.mj_forward(self.env.mj_model, self.native)
        error = float(np.linalg.norm(self.native.site_xpos[self.ee_site] - self.target))
        self.errors.append(error)
        within = error < 0.05
        self.current_hold = self.current_hold + 1 if within else 0
        self.max_hold = max(self.max_hold, self.current_hold)
        self.events.append({
            "kind": "signal",
            "name": "target_within_tolerance",
            "step": step,
            "active": within,
        })
        for contact_index in range(self.native.ncon):
            contact = self.native.contact[contact_index]
            if contact.dist >= 0:
                continue
            pair = self._contact_pair(int(contact.geom1), int(contact.geom2))
            if pair in self.contact_pairs or pair[::-1] in self.contact_pairs:
                self.collision_count += 1
                self.events.append({
                    "kind": "contact",
                    "body_a": pair[0],
                    "body_b": pair[1],
                    "step": step,
                })

    def _contact_pair(self, geom_a, geom_b):
        return self._contact_name(geom_a), self._contact_name(geom_b)

    def _contact_name(self, geom_id):
        geom_name = mujoco.mj_id2name(
            self.env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if geom_name == "floor":
            return "floor"
        body_id = int(self.env.mj_model.geom_bodyid[geom_id])
        return mujoco.mj_id2name(
            self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ) or "world"

    def result(self):
        return {
            "metrics": {
                "minimum_target_error": min(self.errors),
                "final_target_error": self.errors[-1],
                "maximum_hold_steps": self.max_hold,
                "forbidden_contact_count": self.collision_count,
                "control_steps": len(self.errors),
                "simulation_time_sec": len(self.errors) * self.env.dt,
                "mujoco_warning_count": int(sum(self.native.warning.number)),
                "target_x": float(self.target[0]),
                "target_y": float(self.target[1]),
                "target_z": float(self.target[2]),
            },
            "events": self.events,
            "evidence": {
                "events": ["target_within_tolerance"],
                "contacts": [
                    {"body_a": a, "body_b": b, "complete": True}
                    for a, b in self.contact_pairs
                ],
            },
        }
