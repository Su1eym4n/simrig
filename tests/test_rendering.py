from __future__ import annotations

import unittest

from simrig.cli import build_parser
from simrig.rendering import CameraState, ensure_offscreen_framebuffer, preferred_gl_backends


class RenderingTests(unittest.TestCase):
    def test_preview_defaults_to_mujoco_render_mode(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "preview",
                "policy.params",
                "--env",
                "Go1JoystickFlatTerrain",
            ]
        )

        self.assertEqual(args.render_mode, "mujoco")

    def test_preferred_gl_backends_on_darwin(self) -> None:
        backends = preferred_gl_backends()
        self.assertTrue(backends)

    def test_ensure_offscreen_framebuffer_grows_model_buffer(self) -> None:
        class Global:
            offwidth = 640
            offheight = 480

        class Vis:
            global_ = Global()

        class Model:
            vis = Vis()

        model = Model()
        ensure_offscreen_framebuffer(model, width=960, height=540)

        self.assertEqual(model.vis.global_.offwidth, 960)
        self.assertEqual(model.vis.global_.offheight, 540)

    def test_camera_state_updates_from_query(self) -> None:
        state = CameraState()
        state.update_from_query(
            {
                "azimuth": ["140"],
                "elevation": ["-10"],
                "distance": ["3.5"],
            }
        )

        self.assertEqual(state.azimuth, 140.0)
        self.assertEqual(state.elevation, -10.0)
        self.assertEqual(state.distance, 3.5)


if __name__ == "__main__":
    unittest.main()
