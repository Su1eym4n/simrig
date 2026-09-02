"""Seeded random normalized actions for the learned-policy negative control."""


def make_controller(env):
    import jax

    return lambda state, rng: jax.random.uniform(
        rng, (env.action_size,), minval=-1.0, maxval=1.0
    )
