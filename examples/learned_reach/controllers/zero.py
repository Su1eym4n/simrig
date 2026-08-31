"""Zero normalized action means midpoint position targets, not zero torque."""


def make_controller(env):
    import jax.numpy as jp

    return lambda state, rng: jp.zeros(env.action_size)
