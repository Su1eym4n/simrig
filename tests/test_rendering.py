from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from simrig.cli import build_parser
from simrig.browser_render import named_camera_names
from simrig.rendering import (
    CameraState,
    configure_headless_mujoco_gl,
    ensure_offscreen_framebuffer,
    preferred_gl_backends,
)


class RenderingTests(unittest.TestCase):
    def test_preview_defaults_to_threejs_render_mode(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "preview",
                "policy.params",
                "--env",
                "Go1JoystickFlatTerrain",
            ]
        )

        self.assertEqual(args.render_mode, "threejs")

    def test_threejs_preview_uses_live_scene_state(self) -> None:
        from simrig.preview import _html

        page = _html("threejs")

        self.assertIn("three@0.184.0", page)
        self.assertIn("/scene.json", page)
        self.assertIn("/state.json", page)
        self.assertIn("/agent-cameras.json", page)
        self.assertIn("/agent-frame.jpg", page)
        self.assertIn('id="agent-camera-three"', page)
        self.assertIn("updateAuthoredCameras(status.authored_cameras)", page)
        self.assertIn("renderAgentCameraEmulation()", page)
        self.assertIn("followRobot", page)
        self.assertIn('id="three-view"', page)

    def test_preview_retains_streamed_mujoco_mode(self) -> None:
        from simrig.preview import _html

        parser = build_parser()
        args = parser.parse_args(
            [
                "preview",
                "policy.params",
                "--env",
                "Go1JoystickFlatTerrain",
                "--render-mode",
                "mujoco",
            ]
        )

        self.assertEqual(args.render_mode, "mujoco")
        self.assertIn("/frame.jpg", _html("mujoco"))

    def test_preferred_gl_backends_on_darwin(self) -> None:
        backends = preferred_gl_backends()
        self.assertTrue(backends)

    def test_headless_linux_selects_egl_before_mujoco_import(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "simrig.rendering.sys.platform",
            "linux",
        ):
            self.assertEqual(configure_headless_mujoco_gl(), "egl")
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")

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

    def test_named_camera_names_reads_authored_mujoco_cameras(self) -> None:
        try:
            import mujoco
        except ImportError as exc:
            self.skipTest(str(exc))

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><camera name="front"/><camera name="wrist"/>'
            '<geom type="sphere" size="0.1"/></worldbody></mujoco>'
        )

        self.assertEqual(named_camera_names(mujoco, model), ["front", "wrist"])


if __name__ == "__main__":
    unittest.main()
