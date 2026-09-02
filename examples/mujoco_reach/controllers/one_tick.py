"""Reach the goal, then immediately command a far pose instead of holding it."""

import importlib.util
import numpy as np
from pathlib import Path

CONTROLLERS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("orbit_reach_ik", CONTROLLERS_DIR / "ik.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load the adjacent IK controller")
ik_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ik_module)
ik_action = ik_module.action


def make_controller(env):
    import mujoco

    native = mujoco.MjData(env.mj_model)
    low, high = env.mj_model.actuator_ctrlrange.T
    leaving = False
    previous = None

    def control(state, rng):
        nonlocal leaving, previous
        del rng
        native.qpos[:] = np.asarray(state.data.qpos)
        native.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(env.mj_model, native)
        if previous is None or int(state.info["steps"]) == 0:
            leaving = False
            previous = 2 * (np.asarray(state.data.qpos[: env.action_size]) - low) / (
                high - low
            ) - 1
        if float(state.metrics["within_tolerance"]) > 0.5:
            leaving = True
        if leaving:
            target = np.asarray(state.info["target_pos"])
            opposite_yaw = (np.arctan2(target[1], target[0]) + 2 * np.pi) % (
                2 * np.pi
            ) - np.pi
            safe_pose = np.asarray([opposite_yaw, -0.8, 1.4])
            goal = 2 * (safe_pose - low) / (high - low) - 1
            delta = 0.2
        else:
            goal = ik_action(
                env.mj_model, native, np.asarray(state.info["target_pos"])
            )
            delta = 0.035
        previous = np.clip(goal, previous - delta, previous + delta)
        return previous

    return control
