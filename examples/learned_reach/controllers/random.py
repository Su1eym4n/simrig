"""Seeded uniform normalized actions; no learned parameters."""


def make_controller(env):
    import jax

    return lambda state, rng: jax.random.uniform(rng, (env.action_size,), minval=-1, maxval=1)
