"""Scripted joint-position controller for the included reaching arms.

This is ordinary inverse kinematics, not a trained policy. MuJoCo's position
actuators execute the targets; the controller never writes simulator state.
"""

import math

import numpy as np


def make_controller(env):
    """Rate-limit IK targets so the positive control follows a safe joint path."""
    import mujoco

    native = mujoco.MjData(env.mj_model)
    low, high = env.mj_model.actuator_ctrlrange.T
    previous = None

    def control(state, rng):
        nonlocal previous
        del rng
        native.qpos[:] = np.asarray(state.data.qpos)
        native.qvel[:] = np.asarray(state.data.qvel)
        mujoco.mj_forward(env.mj_model, native)
        if previous is None or int(state.info["steps"]) == 0:
            previous = 2 * (np.asarray(state.data.qpos[: env.action_size]) - low) / (
                high - low
            ) - 1
        goal = action(env.mj_model, native, np.asarray(state.info["target_pos"]))
        previous = np.clip(goal, previous - 0.035, previous + 0.035)
        return previous

    return control


def action(model, data, target):
    """Return normalized position targets in the model's actuator order."""
    origin = data.xanchor[model.joint("shoulder").id]
    x, y, z = target - origin
    has_base_yaw = model.nu == 3
    yaw = math.atan2(y, x) if has_base_yaw else None
    planar_x = math.hypot(x, y) if has_base_yaw else x
    first = float(model.body("link2").pos[0])
    second = float(model.site("ee_site").pos[0])
    cosine = (
        planar_x * planar_x + z * z - first * first - second * second
    ) / (2 * first * second)
    if not -1 <= cosine <= 1:
        raise ValueError("Target is outside the arm's geometric reach")
    # Use the elbow-up solution so link1 lifts away from the floor/base.
    elbow = math.acos(cosine)
    # Both hinges rotate around +Y: positive angles turn +X toward -Z.
    shoulder = -math.atan2(z, planar_x) - math.atan2(
        second * math.sin(elbow), first + second * math.cos(elbow)
    )
    positions = (
        np.array([yaw, shoulder, elbow])
        if has_base_yaw
        else np.array([shoulder, elbow])
    )
    low, high = model.actuator_ctrlrange.T
    if np.any(positions < low) or np.any(positions > high):
        raise ValueError("IK target exceeds this model's actuator limits")
    return 2 * (positions - low) / (high - low) - 1
