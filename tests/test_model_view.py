from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from simrig.cli import build_parser


ARM_XML = """\
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="shoulder" type="hinge" range="-1.5 1.5"/>
      <geom type="capsule" size="0.05 0.2"/>
      <body name="link2" pos="0 0 0.4">
        <joint name="elbow" type="hinge" range="0 2"/>
        <geom type="capsule" size="0.05 0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

KEYFRAME_XML = """\
<mujoco model="keyframed_arm">
  <worldbody>
    <body name="link">
      <joint name="shoulder" type="hinge" range="-1 1"/>
      <geom type="capsule" size="0.05 0.2"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0.75"/>
  </keyframe>
</mujoco>
"""

CAMERA_XML = """\
<mujoco model="camera_test">
  <worldbody>
    <camera name="fixed" pos="0 -2 1" zaxis="0 -1 0"/>
    <camera name="wrist" pos="1 -1 1" zaxis="1 -1 0"/>
    <body name="body"><geom type="sphere" size="0.1"/></body>
  </worldbody>
</mujoco>
"""


class ModelViewTests(unittest.TestCase):
    def test_view_model_command_parses_browser_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "view-model",
                "unitree_go1",
                "--menagerie",
                "/tmp/menagerie",
                "--port",
                "8770",
                "--render-mode",
                "topdown",
            ]
        )

        self.assertEqual(args.model_or_xml, "unitree_go1")
        self.assertEqual(str(args.menagerie), "/tmp/menagerie")
        self.assertEqual(args.port, 8770)
        self.assertEqual(args.render_mode, "topdown")

    def test_view_model_defaults_to_threejs(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["view-model", "unitree_go1"])

        self.assertEqual(args.render_mode, "threejs")

    def test_threejs_view_uses_webgl_scene_endpoint(self) -> None:
        from simrig.model_view import _html

        page = _html("threejs")

        self.assertIn("three@0.184.0", page)
        self.assertIn("OrbitControls", page)
        self.assertIn("/scene.json", page)
        self.assertIn("/agent-cameras.json", page)
        self.assertIn("/agent-frame.jpg", page)
        self.assertIn('id="agent-camera-panel"', page)
        self.assertIn('id="agent-camera-three"', page)
        self.assertIn('<option value="emulated" selected>Emulated</option>', page)
        self.assertIn('<option value="sensor">Sensor</option>', page)
        self.assertIn("new THREE.PerspectiveCamera(45, 4 / 3", page)
        self.assertIn("authored.fovy", page)
        self.assertIn('id="three-view"', page)

    def test_threejs_view_exposes_named_agent_cameras(self) -> None:
        from simrig.model_view import ModelViewSession

        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "camera.xml"
            xml.write_text(CAMERA_XML, encoding="utf-8")
            pump = MagicMock()
            pump.stats.return_value = {"fps_target": 12, "renderer_error": None}
            pump.get_jpeg.return_value = b"jpeg"
            with patch("simrig.model_view.MujocoFramePump", return_value=pump):
                try:
                    session = ModelViewSession(xml, render_mode="threejs", camera="fixed")
                except (ImportError, RuntimeError) as exc:
                    self.skipTest(str(exc))
                try:
                    payload = session.agent_cameras_payload()
                    self.assertEqual(payload["cameras"], ["fixed", "wrist"])
                    self.assertEqual(payload["selected"], "fixed")
                    self.assertEqual(session.agent_frame_jpeg(), b"jpeg")

                    scene = session.scene_payload()
                    authored = scene["authored_cameras"]
                    self.assertEqual([item["name"] for item in authored], ["fixed", "wrist"])
                    self.assertEqual(authored[0]["position"], [0.0, -2.0, 1.0])
                    self.assertAlmostEqual(authored[0]["fovy"], 45.0)
                    self.assertEqual(len(authored[0]["matrix"]), 9)

                    selected = session.select_agent_camera("wrist")
                    self.assertEqual(selected["selected"], "wrist")
                    pump.select_fixed_camera.assert_called_once_with("wrist")
                finally:
                    session.close()

    def test_joint_controls_use_joint_names_when_mujoco_available(self) -> None:
        try:
            import numpy  # noqa: F401
            from simrig.model_view import joint_control_specs
            from simrig.mujoco_backend import _import_mujoco

            mujoco = _import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "arm.xml"
            xml.write_text(ARM_XML, encoding="utf-8")

            model = mujoco.MjModel.from_xml_path(str(xml))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            controls = joint_control_specs(mujoco, model, data)
            labels = [control.label for control in controls]

            self.assertEqual(labels, ["shoulder", "elbow"])
            self.assertEqual(controls[0].joint_name, "shoulder")
            self.assertEqual(controls[1].joint_name, "elbow")
            self.assertTrue(controls[0].limited)
            self.assertEqual(controls[0].min, -1.5)
            self.assertEqual(controls[0].max, 1.5)

    def test_model_view_starts_from_first_authored_keyframe(self) -> None:
        from simrig.model_view import ModelViewSession

        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "keyframed_arm.xml"
            xml.write_text(KEYFRAME_XML, encoding="utf-8")
            try:
                session = ModelViewSession(xml)
            except (ImportError, RuntimeError) as exc:
                self.skipTest(str(exc))
            try:
                payload = session.joints_payload()
                self.assertEqual(payload["render_mode"], "threejs")
                self.assertEqual(payload["initial_keyframe"], "home")
                self.assertAlmostEqual(payload["controls"][0]["value"], 0.75)
                session.set_joint_value(0, 0, -0.25)
                session.reset()
                self.assertAlmostEqual(session.joints_payload()["controls"][0]["value"], 0.75)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
