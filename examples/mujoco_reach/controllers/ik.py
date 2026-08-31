"""Scripted joint-position controller for examples/models/simple_arm.xml.

This is ordinary inverse kinematics, not a trained policy. MuJoCo's position
actuators execute the targets; the controller never writes simulator state.
"""

import math

import numpy as np


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
