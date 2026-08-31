"""Independent physical measurements for the unchanged demo_reach task.

The agent-owned code below applies fixed targets and measures ee_site using
native MuJoCo forward kinematics. It never reads the environment's reward or
success metric. Policy/controller loading and stepping belong to SimRig.
"""

from pathlib import Path
import hashlib

import mujoco
import numpy as np

from simrig.playground_evaluator import PlaygroundEvaluator


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "simple_arm.xml"
EVALUATOR_SPEC = {
    "name": "learned-demo-reach",
    "version": "1.0.0",
    "protocol_version": 1,
    "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
}


def make_evaluator(config):
    if config:
        raise ValueError("This reference has no evaluator configuration overrides")
    return PlaygroundEvaluator(ReachMeasurements)


class ReachMeasurements:
    def __init__(self, env, request):
        if Path(env.xml_path).resolve() != MODEL_PATH.resolve():
            raise ValueError("This evaluator measures only the supplied simple arm")
        if set(request["parameters"]) != {"target"}:
            raise ValueError("The reaching scenario must specify only target XYZ")
        self.env = env
        self.target = np.asarray(request["parameters"]["target"], dtype=float)
        if self.target.shape != (3,) or not np.isfinite(self.target).all() or self.target[1] != 0:
            raise ValueError("Target must be finite world-frame XYZ in the arm's X-Z plane")
        self.native = mujoco.MjData(env.mj_model)
        self.site = env.mj_model.site("ee_site").id
        self.marker = env.mj_model.body("target").mocapid[0]
        self.errors = []
        self.events = []

    def reset(self, state):
        import jax.numpy as jp

        info = dict(state.info)
        info["target_pos"] = jp.asarray(self.target)
        data = state.data.replace(mocap_pos=state.data.mocap_pos.at[self.marker].set(self.target))
        # This task's observations explicitly contain the target delta. Keep the
        # policy input and visible target synchronized when applying a scenario.
        return state.replace(data=data, info=info, obs=self.env._get_obs(data, info))

    def observe(self, state, step):
        self.native.qpos[:] = np.asarray(state.data.qpos)
        self.native.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(self.env.mj_model, self.native)
        error = float(np.linalg.norm(self.native.site_xpos[self.site] - self.target))
        self.errors.append(error)
        self.events.append({"kind": "signal", "name": "target_arrival", "step": step, "active": error < 0.05})

    def result(self):
        return {
            "metrics": {
                "minimum_target_error": min(self.errors),
                "final_target_error": self.errors[-1],
                "control_steps": len(self.errors),
                "simulation_time_sec": len(self.errors) * self.env.dt,
            },
            "events": self.events,
            "evidence": {"events": ["target_arrival"]},
        }
