from __future__ import annotations

import unittest


class JaxBraxCompatTests(unittest.TestCase):
    def test_patch_restores_device_put_replicated(self) -> None:
        try:
            import jax
        except ImportError:
            self.skipTest("jax is not installed")

        from simrig.playground_backend import _patch_jax_brax_compat

        # Simulate JAX 0.10 public surface where the name raises AttributeError.
        jax.__dict__.pop("device_put_replicated", None)
        _patch_jax_brax_compat(jax)
        self.assertIn("device_put_replicated", jax.__dict__)
        self.assertTrue(callable(jax.device_put_replicated))


if __name__ == "__main__":
    unittest.main()
