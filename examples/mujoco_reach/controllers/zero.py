"""Zero normalized action baseline for the same MuJoCo arm.

Zero maps to each position actuator's range midpoint, not zero torque.
This is a comparison controller, not a checkpoint or a simulated failure flag.
"""

import numpy as np


def make_controller(env):
    import jax.numpy as jp

    return lambda state, rng: jp.zeros(env.action_size)


def action(model, data, target):
    return np.zeros(model.nu)
