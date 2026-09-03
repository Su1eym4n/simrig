"""Browser viewer for MuJoCo models with per-joint controls."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from simrig.browser_render import MujocoFramePump, named_camera_names
from simrig.browser_shell import (
    agent_camera_panel,
    agent_camera_script,
    agent_camera_styles,
    camera_interaction_script,
    frame_poll_script,
    threejs_agent_camera_script,
    viewer_chrome_script,
    viewer_styles,
)
from simrig.mujoco_backend import _import_mujoco, resolve_model_path
from simrig.rendering import (
    CameraState,
    configure_headless_mujoco_gl,
    make_tracking_camera,
    tracking_body_id,
)
from simrig.three_scene import camera_transforms, geom_transforms, scene_payload


@dataclass(frozen=True)
class JointControl:
    joint_id: int
    joint_name: str
    joint_type: str
    component: int
    label: str
    value: float
    min: float
    max: float
    limited: bool


def joint_control_specs(mujoco: Any, model: Any, data: Any) -> list[JointControl]:
    """Build browser controls for every joint, using joint names as labels."""

    specs: list[JointControl] = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        joint_type = int(model.jnt_type[joint_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        limited = bool(model.jnt_limited[joint_id])
        joint_range = model.jnt_range[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            specs.append(
                _scalar_control(
                    mujoco,
                    model,
                    data,
                    joint_id=joint_id,
                    joint_name=joint_name,
                    joint_type="hinge",
                    component=0,
                    qpos_index=qpos_adr,
                    label=joint_name,
                    limited=limited,
                    joint_range=joint_range,
                    default_min=-3.14159,
                    default_max=3.14159,
                )
            )
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            specs.append(
                _scalar_control(
                    mujoco,
                    model,
                    data,
                    joint_id=joint_id,
                    joint_name=joint_name,
                    joint_type="slide",
                    component=0,
                    qpos_index=qpos_adr,
                    label=joint_name,
                    limited=limited,
                    joint_range=joint_range,
                    default_min=-1.0,
                    default_max=1.0,
                )
            )
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            axis_labels = ("x", "y", "z", "qw", "qx", "qy", "qz")
            defaults = (
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-0.5, 2.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
            )
            for component, axis in enumerate(axis_labels):
                min_value, max_value = defaults[component]
                specs.append(
                    JointControl(
                        joint_id=joint_id,
                        joint_name=joint_name,
                        joint_type="free",
                        component=component,
                        label=f"{joint_name}_{axis}",
                        value=float(data.qpos[qpos_adr + component]),
                        min=min_value,
                        max=max_value,
                        limited=False,
                    )
                )
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            axis_labels = ("qw", "qx", "qy", "qz")
            for component, axis in enumerate(axis_labels):
                specs.append(
                    JointControl(
                        joint_id=joint_id,
                        joint_name=joint_name,
                        joint_type="ball",
                        component=component,
                        label=f"{joint_name}_{axis}",
                        value=float(data.qpos[qpos_adr + component]),
                        min=-1.0,
                        max=1.0,
                        limited=False,
                    )
                )
    return specs


def _scalar_control(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    joint_id: int,
    joint_name: str,
    joint_type: str,
    component: int,
    qpos_index: int,
    label: str,
    limited: bool,
    joint_range: Any,
    default_min: float,
    default_max: float,
) -> JointControl:
    del model, mujoco
    min_value = float(joint_range[0]) if limited else default_min
    max_value = float(joint_range[1]) if limited else default_max
    return JointControl(
        joint_id=joint_id,
        joint_name=joint_name,
        joint_type=joint_type,
        component=component,
        label=label,
        value=float(data.qpos[qpos_index]),
        min=min_value,
        max=max_value,
        limited=limited,
    )


class ModelViewSession:
    """Owns a MuJoCo model, joint state, and browser renderer."""

    def __init__(
        self,
        model_or_xml: str | Path,
        *,
        menagerie: Path | str | None = None,
        width: int = 960,
        height: int = 540,
        render_mode: str = "threejs",
        camera: str | int | None = None,
        fps: int = 24,
    ) -> None:
        self.model_path = resolve_model_path(model_or_xml, menagerie)
        configure_headless_mujoco_gl()
        self.mujoco = _import_mujoco()
        try:
            from PIL import Image  # type: ignore
            from PIL import ImageDraw  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Browser model view requires Pillow.") from exc

        self.Image = Image
        self.ImageDraw = ImageDraw
        self.model = self.mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = self.mujoco.MjData(self.model)
        self.initial_keyframe: str | None = None
        if self.model.nkey:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.initial_keyframe = self.mujoco.mj_id2name(
                self.model,
                self.mujoco.mjtObj.mjOBJ_KEY,
                0,
            ) or "keyframe_0"
        self.mujoco.mj_forward(self.model, self.data)
        self._initial_qpos = np.array(self.data.qpos, copy=True)

        self.width = width
        self.height = height
        self.render_mode = render_mode.lower()
        self.renderer_error: str | None = None
        self._lock = threading.Lock()
        self._last_frame_jpeg: bytes | None = None
        self._scene_payload: dict[str, Any] | None = None

        self.renderer = None
        if self.render_mode not in ("threejs", "topdown", "mujoco"):
            raise ValueError("render_mode must be 'threejs', 'mujoco', or 'topdown'.")
        self.camera, self.camera_state = make_tracking_camera(
            self.mujoco,
            self.model,
            self.data,
            camera,
        )
        self._frame_pump: MujocoFramePump | None = None
        self.agent_camera_names = named_camera_names(self.mujoco, self.model)
        self.agent_camera_name = self._initial_agent_camera(camera)
        self._agent_frame_pump: MujocoFramePump | None = None
        self._agent_pump_lock = threading.Lock()
        self._agent_width = min(360, max(160, width))
        self._agent_height = max(120, round(self._agent_width * 3 / 4))
        self._agent_fps = min(12, max(1, int(fps)))
        if self.render_mode != "threejs":
            self._frame_pump = MujocoFramePump(
                self.mujoco,
                self.model,
                self.data,
                width=width,
                height=height,
                camera=self.camera,
                camera_state=self.camera_state,
                image_module=self.Image,
                fps=fps,
                render_mode=self.render_mode,
                scene_lock=self._lock,
                fallback_frame=self._fallback_frame,
                error_frame=self._error_frame,
            )

    def reset(self) -> None:
        with self._lock:
            self.data.qpos[:] = self._initial_qpos
            self.mujoco.mj_forward(self.model, self.data)

    def set_joint_value(self, joint_id: int, component: int, value: float) -> None:
        with self._lock:
            if joint_id < 0 or joint_id >= self.model.njnt:
                raise ValueError(f"Invalid joint id: {joint_id}")
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type in (
                self.mujoco.mjtJoint.mjJNT_HINGE,
                self.mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                if component != 0:
                    raise ValueError(f"Joint {joint_id} only has one position coordinate.")
                if bool(self.model.jnt_limited[joint_id]):
                    low, high = self.model.jnt_range[joint_id]
                    value = float(np.clip(value, low, high))
                self.data.qpos[qpos_adr] = value
            elif joint_type == self.mujoco.mjtJoint.mjJNT_FREE:
                if component < 0 or component > 6:
                    raise ValueError(f"Invalid free-joint component: {component}")
                self.data.qpos[qpos_adr + component] = value
                quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
                norm = float(np.linalg.norm(quat))
                if norm > 1e-8:
                    self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat / norm
            elif joint_type == self.mujoco.mjtJoint.mjJNT_BALL:
                if component < 0 or component > 3:
                    raise ValueError(f"Invalid ball-joint component: {component}")
                self.data.qpos[qpos_adr + component] = value
                quat = self.data.qpos[qpos_adr : qpos_adr + 4]
                norm = float(np.linalg.norm(quat))
                if norm > 1e-8:
                    self.data.qpos[qpos_adr : qpos_adr + 4] = quat / norm
            else:
                raise ValueError(f"Unsupported joint type for joint {joint_id}.")
            self.mujoco.mj_forward(self.model, self.data)

    def set_camera_from_query(self, query: dict[str, list[str]]) -> None:
        if self._frame_pump is not None:
            self._frame_pump.set_camera_from_query(query)

    def joints_payload(self) -> dict[str, Any]:
        acquired = self._lock.acquire(blocking=False)
        try:
            controls = joint_control_specs(self.mujoco, self.model, self.data)
            pump_stats = (
                self._frame_pump.stats()
                if self._frame_pump is not None
                else {
                    "renderer_error": None,
                    "fps_target": 60,
                    "camera": {
                        "interactive": True,
                        "renderer": "threejs-orbit-controls",
                    },
                }
            )
            return {
                "model_name": self.model_path.parent.name or self.model_path.stem,
                "model_path": str(self.model_path),
                "joint_count": self.model.njnt,
                "control_count": len(controls),
                "render_mode": self.render_mode,
                "initial_keyframe": self.initial_keyframe,
                "renderer_error": pump_stats["renderer_error"],
                "fps_target": pump_stats["fps_target"],
                "camera": pump_stats["camera"],
                "transforms": self.geom_transforms(),
                "authored_cameras": camera_transforms(
                    self.mujoco,
                    self.model,
                    self.data,
                ),
                "controls": [
                    {
                        "joint_id": control.joint_id,
                        "joint_name": control.joint_name,
                        "joint_type": control.joint_type,
                        "component": control.component,
                        "label": control.label,
                        "value": control.value,
                        "min": control.min,
                        "max": control.max,
                        "limited": control.limited,
                    }
                    for control in controls
                ],
            }
        finally:
            if acquired:
                self._lock.release()

    def scene_payload(self) -> dict[str, Any]:
        """Return static render geometry plus the current world transforms."""

        if self._scene_payload is None:
            self._scene_payload = scene_payload(
                self.mujoco,
                self.model,
                self.data,
                model_name=self.model_path.parent.name or self.model_path.stem,
            )
            self._scene_payload.pop("transforms", None)
            self._scene_payload.pop("authored_cameras", None)
        return {
            **self._scene_payload,
            "transforms": self.geom_transforms(),
            "authored_cameras": camera_transforms(
                self.mujoco,
                self.model,
                self.data,
            ),
        }

    def geom_transforms(self) -> list[dict[str, Any]]:
        return geom_transforms(self.model, self.data)

    def frame_jpeg(self) -> bytes:
        if self._frame_pump is None:
            raise RuntimeError("Frame streaming is disabled in threejs mode.")
        return self._frame_pump.get_jpeg()

    def agent_cameras_payload(self) -> dict[str, Any]:
        stats = (
            self._agent_frame_pump.stats()
            if self._agent_frame_pump is not None
            else {"fps_target": self._agent_fps, "renderer_error": None}
        )
        return {
            "cameras": list(self.agent_camera_names),
            "selected": self.agent_camera_name,
            "fps_target": stats["fps_target"],
            "renderer_error": stats["renderer_error"],
            "source": "mujoco-offscreen",
        }

    def select_agent_camera(self, name: str) -> dict[str, Any]:
        if name not in self.agent_camera_names:
            choices = ", ".join(self.agent_camera_names) or "none"
            raise ValueError(f"Unknown agent camera: {name}. Available: {choices}")
        self.agent_camera_name = name
        if self._agent_frame_pump is not None:
            self._agent_frame_pump.select_fixed_camera(self._agent_camera_ref(name))
        return self.agent_cameras_payload()

    def agent_frame_jpeg(self) -> bytes:
        pump = self._ensure_agent_frame_pump()
        if pump is None:
            raise RuntimeError("No authored MuJoCo cameras are available.")
        return pump.get_jpeg()

    def _ensure_agent_frame_pump(self) -> MujocoFramePump | None:
        if self.agent_camera_name is None:
            return None
        with self._agent_pump_lock:
            if self._agent_frame_pump is None:
                self._agent_frame_pump = MujocoFramePump(
                    self.mujoco,
                    self.model,
                    self.data,
                    width=self._agent_width,
                    height=self._agent_height,
                    camera=self._agent_camera_ref(self.agent_camera_name),
                    camera_state=CameraState(interactive=False),
                    image_module=self.Image,
                    fps=self._agent_fps,
                    render_mode="mujoco",
                    scene_lock=self._lock,
                )
            return self._agent_frame_pump

    def close(self) -> None:
        if self._frame_pump is not None:
            self._frame_pump.close()
        if self._agent_frame_pump is not None:
            self._agent_frame_pump.close()

    def _initial_agent_camera(self, requested: str | int | None) -> str | None:
        if not self.agent_camera_names:
            return None
        if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
            index = int(requested)
            if 0 <= index < len(self.agent_camera_names):
                return self.agent_camera_names[index]
        if isinstance(requested, str) and requested in self.agent_camera_names:
            return requested
        return self.agent_camera_names[0]

    def _agent_camera_ref(self, name: str) -> str | int:
        camera_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_CAMERA,
            name,
        )
        return name if camera_id >= 0 else self.agent_camera_names.index(name)

    def _tracking_body_id(self) -> int:
        return tracking_body_id(self.mujoco, self.model, self.data)

    def _fallback_frame(self) -> np.ndarray:
        """Explicit schematic mode only; not used for default MuJoCo rendering."""
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        model = self.model
        center = np.asarray(self.data.xpos[self._tracking_body_id()])[:2]
        scale = min(self.width, self.height) / 5.0

        def project(pos: np.ndarray) -> tuple[int, int]:
            xy = (np.asarray(pos)[:2] - center) * scale
            return int(self.width / 2 + xy[0]), int(self.height / 2 - xy[1])

        grid_color = (28, 34, 38)
        for offset in np.linspace(-2.0, 2.0, 9):
            x1, y1 = project(center + np.array([offset, -2.0]))
            x2, y2 = project(center + np.array([offset, 2.0]))
            draw.line((x1, y1, x2, y2), fill=grid_color)
            x1, y1 = project(center + np.array([-2.0, offset]))
            x2, y2 = project(center + np.array([2.0, offset]))
            draw.line((x1, y1, x2, y2), fill=grid_color)

        for body_id in range(1, model.nbody):
            parent = int(model.body_parentid[body_id])
            if parent <= 0:
                continue
            x1, y1 = project(self.data.xpos[parent])
            x2, y2 = project(self.data.xpos[body_id])
            draw.line((x1, y1, x2, y2), fill=(96, 150, 255), width=3)

        for body_id in range(1, model.nbody):
            x, y = project(self.data.xpos[body_id])
            radius = 7 if body_id == self._tracking_body_id() else 4
            fill = (255, 210, 94) if body_id == self._tracking_body_id() else (205, 224, 255)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

        overlay = [
            "SimRig Model View",
            f"mode: {self.render_mode}",
            f"model: {self.model_path.name}",
            f"joints: {self.model.njnt}",
        ]
        if self.renderer_error:
            overlay.append(f"render error: {self.renderer_error[:80]}")
        draw.rectangle((16, 16, min(self.width - 16, 520), 140), fill=(0, 0, 0))
        y = 28
        for line in overlay:
            draw.text((28, y), line, fill=(236, 240, 245))
            y += 22
        return np.asarray(image)

    def _error_frame(self, message: str) -> np.ndarray:
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        lines = [
            "SimRig Model View",
            "MuJoCo rendering failed",
            message[:240],
            "Try: export MUJOCO_GL=glfw",
        ]
        y = 28
        for line in lines:
            draw.text((28, y), line, fill=(236, 240, 245))
            y += 24
        return np.asarray(image)

    def _plain_frame(self, message: str) -> np.ndarray:
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        draw.text((28, 28), "SimRig Model View", fill=(236, 240, 245))
        draw.text((28, 56), message, fill=(180, 190, 200))
        return np.asarray(image)


def serve_model_view(
    model_or_xml: str | Path,
    *,
    menagerie: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8766,
    width: int = 960,
    height: int = 540,
    render_mode: str = "threejs",
    camera: str | int | None = None,
    fps: int = 24,
) -> None:
    session = ModelViewSession(
        model_or_xml,
        menagerie=menagerie,
        width=width,
        height=height,
        render_mode=render_mode,
        camera=camera,
        fps=fps,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_html(session.render_mode))
            elif parsed.path == "/frame.jpg":
                if session.render_mode == "threejs":
                    self.send_error(HTTPStatus.NOT_FOUND, "Frame streaming disabled")
                else:
                    self._send_bytes(session.frame_jpeg(), "image/jpeg")
            elif parsed.path == "/agent-frame.jpg":
                try:
                    self._send_bytes(session.agent_frame_jpeg(), "image/jpeg")
                except RuntimeError as exc:
                    self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            elif parsed.path == "/agent-cameras.json":
                self._send_json(session.agent_cameras_payload())
            elif parsed.path == "/agent-camera":
                query = parse_qs(parsed.query)
                name = query.get("name", [""])[0]
                try:
                    self._send_json(session.select_agent_camera(name))
                except (RuntimeError, ValueError) as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/scene.json":
                self._send_json(session.scene_payload(), compress=True)
            elif parsed.path == "/joints.json":
                self._send_json(session.joints_payload())
            elif parsed.path == "/camera":
                query = parse_qs(parsed.query)
                session.set_camera_from_query(query)
                self._send_json(session.joints_payload())
            elif parsed.path == "/set":
                query = parse_qs(parsed.query)
                joint_id = int(query.get("joint_id", query.get("joint", ["-1"]))[0])
                component = int(query.get("component", ["0"])[0])
                value = float(query.get("value", ["0"])[0])
                session.set_joint_value(joint_id, component, value)
                self._send_json(session.joints_payload())
            elif parsed.path == "/reset":
                session.reset()
                self._send_json(session.joints_payload())
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, body: str) -> None:
            self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")

        def _send_json(self, value: dict[str, Any], *, compress: bool = False) -> None:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")
            if compress and "gzip" in self.headers.get("Accept-Encoding", ""):
                body = gzip.compress(body, compresslevel=5)
                self._send_bytes(
                    body,
                    "application/json; charset=utf-8",
                    content_encoding="gzip",
                )
                return
            self._send_bytes(body, "application/json; charset=utf-8")

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            *,
            content_encoding: str | None = None,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            if content_encoding is not None:
                self.send_header("Content-Encoding", content_encoding)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SimRig model view: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()


def _html(render_mode: str = "threejs") -> str:
    if render_mode == "threejs":
        return _threejs_html()
    return _frame_html()


def _threejs_html() -> str:
    return (
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SimRig Model View</title>
  <style>"""
        + viewer_styles(sidebar_width=360)
        + agent_camera_styles()
        + """
    #three-view { width: 100%; height: 100%; display: block; outline: none; }
    #loading { position: absolute; inset: 0; display: grid; place-items: center; color: #cbd5e1; background: #070b12; z-index: 2; }
    #loading.error { color: #fca5a5; padding: 28px; text-align: center; white-space: pre-wrap; }
  </style>
  <script type="importmap">
    {"imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/"
    }}
  </script>
</head>
<body>
  <main>
    <div id="viewport">
      <canvas id="three-view" aria-label="Interactive SimRig model"></canvas>
      <div id="loading">Loading WebGL scene…</div>
      <div id="hint">Drag to orbit · scroll to zoom · right-drag to pan</div>
"""
        + agent_camera_panel()
        + """
    </div>
  </main>
  <aside>
    <h1>SimRig Model View</h1>
    <div id="meta" class="meta">loading joints...</div>
    <button class="secondary" id="reset-joints">Reset Joints</button>
    <button class="secondary" id="reset-camera">Reset Camera</button>
    <div id="controls"></div>
    <details class="simrig-debug">
      <summary>Raw State</summary>
      <pre id="status">loading</pre>
    </details>
  </aside>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const canvas = document.getElementById('three-view');
    const viewport = document.getElementById('viewport');
    const loadingEl = document.getElementById('loading');
    const statusEl = document.getElementById('status');
    const metaEl = document.getElementById('meta');
    const controlsEl = document.getElementById('controls');
    const objects = new Map();
    const meshGeometries = new Map();
    let controlsRendered = false;
"""
        + agent_camera_script()
        + """

    const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.setClearColor(0x0b1220, 1);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1220);
    scene.fog = new THREE.Fog(0x0b1220, 7, 22);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
    camera.up.set(0, 0, 1);
    const orbit = new OrbitControls(camera, canvas);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.075;
    orbit.screenSpacePanning = false;
    orbit.minDistance = 0.08;
    orbit.maxDistance = 100;
    orbit.minPolarAngle = 0.08;
    orbit.maxPolarAngle = Math.PI / 2 - 0.04;

    scene.add(new THREE.HemisphereLight(0xbfdcff, 0x172033, 1.15));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.35);
    keyLight.position.set(4, -5, 8);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 30;
    keyLight.shadow.camera.left = -5;
    keyLight.shadow.camera.right = 5;
    keyLight.shadow.camera.top = 5;
    keyLight.shadow.camera.bottom = -5;
    keyLight.shadow.bias = -0.0002;
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x7aa8ff, 0.85);
    rimLight.position.set(-5, 3, 5);
    scene.add(rimLight);

    const modelRoot = new THREE.Group();
    scene.add(modelRoot);
"""
        + threejs_agent_camera_script()
        + """

    function materialFor(geom) {
      const [r, g, b, a] = geom.rgba;
      const props = geom.material || {};
      if (geom.type === 0) {
        return new THREE.MeshStandardMaterial({color: 0x182231, roughness: 0.92});
      }
      const material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(r, g, b),
        opacity: a,
        transparent: a < 0.999,
        roughness: THREE.MathUtils.clamp(0.68 - (props.shininess || 0) * 0.32, 0.22, 0.82),
        metalness: THREE.MathUtils.clamp((props.reflectance || 0) * 0.45, 0, 0.35),
        clearcoat: THREE.MathUtils.clamp((props.specular || 0) * 0.35, 0, 0.4),
        clearcoatRoughness: 0.35,
      });
      if ((props.emission || 0) > 0) {
        material.emissive.setRGB(r, g, b);
        material.emissiveIntensity = props.emission;
      }
      return material;
    }

    function primitiveGeometry(geom) {
      const [x, y, z] = geom.size;
      switch (geom.type) {
        case 0: // plane
          return new THREE.PlaneGeometry(200, 200);
        case 2: // sphere
          return new THREE.SphereGeometry(x, 32, 20);
        case 3: { // capsule, MuJoCo axis is local Z
          const geometry = new THREE.CapsuleGeometry(x, 2 * y, 10, 24);
          geometry.rotateX(Math.PI / 2);
          return geometry;
        }
        case 4: { // ellipsoid
          const geometry = new THREE.SphereGeometry(1, 32, 20);
          geometry.scale(x, y, z);
          return geometry;
        }
        case 5: { // cylinder, MuJoCo axis is local Z
          const geometry = new THREE.CylinderGeometry(x, x, 2 * y, 32);
          geometry.rotateX(Math.PI / 2);
          return geometry;
        }
        case 6:
          return new THREE.BoxGeometry(2 * x, 2 * y, 2 * z);
        case 7:
          return meshGeometries.get(geom.mesh_id) || null;
        default:
          return null;
      }
    }

    function applyTransform(object, transform) {
      if (!object || !transform) return;
      object.position.fromArray(transform.position);
      const m = transform.matrix;
      const rotation = new THREE.Matrix4();
      rotation.set(
        m[0], m[1], m[2], 0,
        m[3], m[4], m[5], 0,
        m[6], m[7], m[8], 0,
        0, 0, 0, 1,
      );
      object.quaternion.setFromRotationMatrix(rotation);
    }

    function updateTransforms(transforms) {
      for (const transform of transforms || []) {
        applyTransform(objects.get(transform.id), transform);
      }
    }

    function fitCamera() {
      const bounds = new THREE.Box3().setFromObject(modelRoot);
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.25);
      orbit.target.copy(center);
      camera.position.set(
        center.x + radius * 1.35,
        center.y - radius * 1.75,
        center.z + radius * 0.95,
      );
      camera.near = Math.max(radius / 200, 0.002);
      camera.far = Math.max(radius * 80, 100);
      camera.updateProjectionMatrix();
      orbit.update();
    }

    function resize() {
      const width = Math.max(1, viewport.clientWidth);
      const height = Math.max(1, viewport.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }

    function animate() {
      orbit.update();
      renderer.render(scene, camera);
      renderAgentCameraEmulation();
      requestAnimationFrame(animate);
    }

    async function loadScene() {
      const res = await fetch('/scene.json', {cache: 'no-store'});
      if (!res.ok) throw new Error(`scene request failed (${res.status})`);
      const payload = await res.json();
      updateAuthoredCameras(payload.authored_cameras);

      for (const mesh of payload.meshes) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(mesh.vertices, 3));
        geometry.setIndex(mesh.indices);
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        meshGeometries.set(mesh.id, geometry);
      }

      const transformById = new Map(payload.transforms.map(item => [item.id, item]));
      for (const geom of payload.geoms) {
        const geometry = primitiveGeometry(geom);
        if (!geometry || geom.rgba[3] <= 0.001) continue;
        const object = new THREE.Mesh(geometry, materialFor(geom));
        object.name = geom.name;
        object.castShadow = geom.type !== 0;
        object.receiveShadow = true;
        applyTransform(object, transformById.get(geom.id));
        if (geom.type === 0) {
          scene.add(object);
        } else {
          modelRoot.add(object);
        }
        objects.set(geom.id, object);
      }

      const grid = new THREE.GridHelper(30, 60, 0x52647a, 0x263346);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = 0.001;
      grid.material.opacity = 0.42;
      grid.material.transparent = true;
      scene.add(grid);

      fitCamera();
      loadingEl.remove();
    }

    function statusPayload(payload) {
      const copy = {...payload};
      delete copy.transforms;
      delete copy.authored_cameras;
      return copy;
    }

    function renderControls(payload) {
      metaEl.textContent = `${payload.model_name} | ${payload.joint_count} joints | ${payload.control_count} controls | mode: ${payload.render_mode} | fps: display`;
      updateTransforms(payload.transforms);
      updateAuthoredCameras(payload.authored_cameras);
      if (!controlsRendered) {
        controlsEl.innerHTML = '';
        for (const control of payload.controls) {
          const wrapper = document.createElement('div');
          wrapper.className = 'control';
          const label = document.createElement('label');
          label.textContent = control.label;
          const slider = document.createElement('input');
          slider.type = 'range';
          slider.min = control.min;
          slider.max = control.max;
          slider.step = Math.max((control.max - control.min) / 300, 0.000001);
          slider.value = control.value;
          slider.dataset.jointId = control.joint_id;
          slider.dataset.component = control.component;
          const valueEl = document.createElement('div');
          valueEl.className = 'value';
          valueEl.textContent = control.value.toFixed(4);
          slider.addEventListener('input', () => {
            valueEl.textContent = Number(slider.value).toFixed(4);
          });
          slider.addEventListener('change', async () => {
            await setJoint(control.joint_id, control.component, slider.value);
          });
          wrapper.append(label, slider, valueEl);
          controlsEl.appendChild(wrapper);
        }
        controlsRendered = true;
      }
      statusEl.textContent = JSON.stringify(statusPayload(payload), null, 2);
    }

    async function loadJoints() {
      const res = await fetch('/joints.json', {cache: 'no-store'});
      renderControls(await res.json());
    }

    async function setJoint(jointId, component, value) {
      const res = await fetch(`/set?joint_id=${jointId}&component=${component}&value=${value}`, {cache: 'no-store'});
      renderControls(await res.json());
    }

    async function resetJoints() {
      const res = await fetch('/reset', {cache: 'no-store'});
      const payload = await res.json();
      controlsRendered = false;
      renderControls(payload);
    }

    document.getElementById('reset-joints').addEventListener('click', resetJoints);
    document.getElementById('reset-camera').addEventListener('click', fitCamera);
    window.addEventListener('resize', resize);
    resize();
    animate();
    try {
      await Promise.all([loadScene(), loadJoints(), initializeAgentCamera()]);
      setInterval(loadJoints, 2000);
    } catch (err) {
      loadingEl.className = 'error';
      loadingEl.textContent = `WebGL viewer failed to load.\n${err}\n\nThree.js is loaded from jsDelivr, so an internet connection is required.`;
      statusEl.textContent = String(err);
      console.error(err);
    }
  </script>
"""
        + viewer_chrome_script()
        + """
</body>
</html>
"""
    )


def _frame_html() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SimRig Model View</title>
  <style>{viewer_styles(sidebar_width=360)}</style>
</head>
<body>
  <main>
    <div id="viewport">
      <img id="frame" alt="SimRig model frame">
      <div id="hint">Drag to orbit · scroll to zoom</div>
    </div>
  </main>
  <aside>
    <h1>SimRig Model View</h1>
    <div id="meta" class="meta">loading joints...</div>
    <button class="secondary" onclick="resetJoints()">Reset Joints</button>
    <div id="controls"></div>
    <details class="simrig-debug">
      <summary>Raw State</summary>
      <pre id="status">loading</pre>
    </details>
  </aside>
  <script>
    const frame = document.getElementById('frame');
    const statusEl = document.getElementById('status');
    const metaEl = document.getElementById('meta');
    const controlsEl = document.getElementById('controls');
    let refreshControlsTimer = null;
    {camera_interaction_script()}
    {frame_poll_script(poll_ms=33)}
    bindCameraControls(frame);

    function renderControls(payload) {{
      controlsEl.innerHTML = '';
      metaEl.textContent = `${{payload.model_name}} | ${{payload.joint_count}} joints | ${{payload.control_count}} controls | mode: ${{payload.render_mode}} | fps: ${{payload.fps_target}}`;
      applyCameraFromStatus(payload);
      for (const control of payload.controls) {{
        const wrapper = document.createElement('div');
        wrapper.className = 'control';
        const label = document.createElement('label');
        label.textContent = control.label;
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = control.min;
        slider.max = control.max;
        slider.step = (control.max - control.min) / 200;
        slider.value = control.value;
        const valueEl = document.createElement('div');
        valueEl.className = 'value';
        valueEl.textContent = control.value.toFixed(4);
        slider.addEventListener('input', () => {{
          valueEl.textContent = Number(slider.value).toFixed(4);
        }});
        slider.addEventListener('change', async () => {{
          await setJoint(control.joint_id, control.component, slider.value);
        }});
        wrapper.appendChild(label);
        wrapper.appendChild(slider);
        wrapper.appendChild(valueEl);
        controlsEl.appendChild(wrapper);
      }}
      statusEl.textContent = JSON.stringify(payload, null, 2);
    }}

    async function loadJoints() {{
      const res = await fetch('/joints.json', {{cache: 'no-store'}});
      const payload = await res.json();
      renderControls(payload);
    }}

    async function setJoint(jointId, component, value) {{
      const res = await fetch(`/set?joint_id=${{jointId}}&component=${{component}}&value=${{value}}`);
      const payload = await res.json();
      statusEl.textContent = JSON.stringify(payload, null, 2);
    }}

    async function resetJoints() {{
      const res = await fetch('/reset');
      const payload = await res.json();
      renderControls(payload);
    }}

    refreshFrame();
    loadJoints();
    refreshControlsTimer = setInterval(loadJoints, 2000);
  </script>
  {viewer_chrome_script()}
</body>
</html>
"""
