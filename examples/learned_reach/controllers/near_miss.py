"""Plausible shortcut: aim 9 cm short of the target, toward the straight pose.

A loose proximity metric might accept this; the frozen 5 cm arrival test must
judge the actual trajectory. This is a scripted control, not a reward exploit.
"""

import mujoco
import numpy as np

import math


def action(model, data, target):
    """Return normalized position targets in the model's actuator order."""
    origin = data.xanchor[model.joint("shoulder").id]
    x, _, z = target - origin
    first = float(model.body("link2").pos[0])
    second = float(model.site("ee_site").pos[0])
    cosine = (x * x + z * z - first * first - second * second) / (2 * first * second)
    if not -1 <= cosine <= 1:
        raise ValueError("Target is outside the arm's geometric reach")
    # Use the elbow-up solution so link1 lifts away from the floor/base.
    elbow = math.acos(cosine)
    # Both hinges rotate around +Y: positive angles turn +X toward -Z.
    shoulder = -math.atan2(z, x) - math.atan2(
        second * math.sin(elbow), first + second * math.cos(elbow)
    )
    positions = np.array([shoulder, elbow])
    low, high = model.actuator_ctrlrange.T
    if np.any(positions < low) or np.any(positions > high):
        raise ValueError("IK target exceeds this model's actuator limits")
    return 2 * (positions - low) / (high - low) - 1


def make_controller(env):
    data = mujoco.MjData(env.mj_model)

    def control(state, rng):
        del rng
        data.qpos[:] = np.asarray(state.data.qpos)
        mujoco.mj_forward(env.mj_model, data)
        target = np.asarray(state.info["target_pos"])
        toward_start = np.array([0.38, 0, 0.10]) - target
        near_target = target + 0.09 * toward_start / np.linalg.norm(toward_start)
        return action(env.mj_model, data, near_target)

    return control
