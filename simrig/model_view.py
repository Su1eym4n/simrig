"""Browser viewer for MuJoCo models with per-joint controls."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from simrig.browser_render import MujocoFramePump
from simrig.browser_shell import camera_interaction_script, frame_poll_script, viewer_styles
from simrig.mujoco_backend import _import_mujoco, resolve_model_path
from simrig.rendering import make_tracking_camera, tracking_body_id


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
        render_mode: str = "mujoco",
        camera: str | int | None = None,
        fps: int = 24,
    ) -> None:
        self.model_path = resolve_model_path(model_or_xml, menagerie)
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
        self.mujoco.mj_forward(self.model, self.data)
        self._initial_qpos = np.array(self.data.qpos, copy=True)

        self.width = width
        self.height = height
        self.render_mode = render_mode.lower()
        self.renderer_error: str | None = None
        self._lock = threading.Lock()
        self._last_frame_jpeg: bytes | None = None

        self.renderer = None
        if self.render_mode not in ("topdown", "mujoco"):
            raise ValueError("render_mode must be 'topdown' or 'mujoco'.")
        self.camera, self.camera_state = make_tracking_camera(
            self.mujoco,
            self.model,
            self.data,
            camera,
        )
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
        self._frame_pump.set_camera_from_query(query)

    def joints_payload(self) -> dict[str, Any]:
        acquired = self._lock.acquire(blocking=False)
        try:
            controls = joint_control_specs(self.mujoco, self.model, self.data)
            pump_stats = self._frame_pump.stats()
            return {
                "model_name": self.model_path.parent.name or self.model_path.stem,
                "model_path": str(self.model_path),
                "joint_count": self.model.njnt,
                "control_count": len(controls),
                "render_mode": self.render_mode,
                "renderer_error": pump_stats["renderer_error"],
                "fps_target": pump_stats["fps_target"],
                "camera": pump_stats["camera"],
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

    def frame_jpeg(self) -> bytes:
        return self._frame_pump.get_jpeg()

    def close(self) -> None:
        self._frame_pump.close()

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
    render_mode: str = "mujoco",
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
                self._send_html(_html())
            elif parsed.path == "/frame.jpg":
                self._send_bytes(session.frame_jpeg(), "image/jpeg")
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

        def _send_json(self, value: dict[str, Any]) -> None:
            self._send_bytes(
                json.dumps(value, indent=2).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
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


def _html() -> str:
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
    <h1 style="margin-top:18px">Status</h1>
    <pre id="status">loading</pre>
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
</body>
</html>
"""
