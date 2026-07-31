"""Browser preview server for policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import functools
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from simrig.browser_render import MujocoFramePump
from simrig.browser_shell import camera_interaction_script, frame_poll_script, viewer_styles
from simrig.playground_backend import (
    _apply_command,
    _import_training_deps,
    _validate_backend,
    load_env,
)
from simrig.presets import hidden_sizes, resolve_small_network
from simrig.rendering import make_tracking_camera, tracking_body_id


@dataclass
class PreviewStatus:
    env_name: str
    checkpoint: str
    step: int
    reward: float
    total_reward: float
    done: bool
    command: list[float] | None
    command_applied: bool
    paused: bool
    render_mode: str
    frame_busy: bool
    renderer_error: str | None


class PolicyPreviewSession:
    """Owns env, policy, renderer, and mutable rollout state."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        env_name: str,
        backend: str = "mujoco-playground",
        small_network: bool | None = None,
        seed: int = 0,
        command: tuple[float, ...] | None = None,
        width: int = 960,
        height: int = 540,
        frame_skip: int = 1,
        camera: str | int | None = None,
        render_mode: str = "mujoco",
        paused: bool = False,
        fps: int = 24,
    ) -> None:
        _validate_backend(backend)
        self.env_name = env_name
        self.checkpoint = str(checkpoint)
        self.backend = backend
        self.frame_skip = max(1, int(frame_skip))
        self.command = list(command) if command is not None else None
        self.command_applied = False
        self.paused = paused
        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.render_mode = render_mode.lower()
        self.renderer_error: str | None = None
        self._lock = threading.Lock()
        self._last_frame_jpeg: bytes | None = None

        (
            self.jax,
            self.jp,
            self.brax_model,
            self.running_statistics,
            self.ppo_networks,
            *_,
        ) = _import_training_deps()
        try:
            import mujoco  # type: ignore
            from PIL import Image  # type: ignore
            from PIL import ImageDraw  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Browser preview requires MuJoCo and Pillow.") from exc

        self.mujoco = mujoco
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.env = load_env(env_name)
        sizes = hidden_sizes(resolve_small_network(checkpoint, small_network=small_network))
        network_factory = functools.partial(
            self.ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=sizes,
            value_hidden_layer_sizes=sizes,
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )
        networks = network_factory(
            self.env.observation_size,
            self.env.action_size,
            preprocess_observations_fn=self.running_statistics.normalize,
        )
        params = self.brax_model.load_params(str(checkpoint))
        self.policy = self.jax.jit(
            self.ppo_networks.make_inference_fn(networks)(params, deterministic=True)
        )
        self.reset_fn = self.jax.jit(self.env.reset)
        self.step_fn = self.jax.jit(self.env.step)
        self.rng = self.jax.random.PRNGKey(seed)
        self.state = self.reset_fn(self.rng)
        self._apply_current_command()

        self.mj_data = self.mujoco.MjData(self.env.mj_model)
        self.width = width
        self.height = height
        self.renderer = None
        if self.render_mode not in ("topdown", "mujoco"):
            raise ValueError("render_mode must be 'topdown' or 'mujoco'.")
        self.camera, self.camera_state = make_tracking_camera(
            self.mujoco,
            self.env.mj_model,
            self.mj_data,
            camera,
        )
        self._frame_pump = MujocoFramePump(
            self.mujoco,
            self.env.mj_model,
            self.mj_data,
            width=width,
            height=height,
            camera=self.camera,
            camera_state=self.camera_state,
            image_module=self.Image,
            fps=fps,
            render_mode=self.render_mode,
            scene_lock=self._lock,
            before_render=self._advance_rollout,
            fallback_frame=self._fallback_frame,
            error_frame=self._error_frame,
        )

    def reset(self) -> None:
        with self._lock:
            self.rng = self.jax.random.PRNGKey(0)
            self.state = self.reset_fn(self.rng)
            self.step_count = 0
            self.total_reward = 0.0
            self.last_reward = 0.0
            self.done = False
            self._apply_current_command()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused

    def set_command(self, command: list[float] | None) -> None:
        with self._lock:
            self.command = command
            self._apply_current_command()

    def status(self) -> PreviewStatus:
        acquired = self._lock.acquire(blocking=False)
        try:
            pump_stats = self._frame_pump.stats()
            return PreviewStatus(
                env_name=self.env_name,
                checkpoint=self.checkpoint,
                step=self.step_count,
                reward=self.last_reward,
                total_reward=self.total_reward,
                done=self.done,
                command=list(self.command) if self.command is not None else None,
                command_applied=self.command_applied,
                paused=self.paused,
                render_mode=self.render_mode if acquired else "busy",
                frame_busy=not acquired,
                renderer_error=pump_stats["renderer_error"],
            )
        finally:
            if acquired:
                self._lock.release()

    def status_payload(self) -> dict[str, Any]:
        payload = self.status().__dict__
        payload.update(self._frame_pump.stats())
        return payload

    def set_camera_from_query(self, query: dict[str, list[str]]) -> None:
        self._frame_pump.set_camera_from_query(query)

    def frame_jpeg(self) -> bytes:
        return self._frame_pump.get_jpeg()

    def close(self) -> None:
        self._frame_pump.close()

    def _advance_rollout(self) -> None:
        if not self.paused and not self.done:
            for _ in range(self.frame_skip):
                self._step_once_unlocked()
                if self.done:
                    break
        self._copy_state_to_mujoco_unlocked()

    def _step_once_unlocked(self) -> None:
        self._apply_current_command()
        self.rng, action_rng = self.jax.random.split(self.rng)
        action, _ = self.policy(self.state.obs, action_rng)
        self.state = self.step_fn(self.state, action)
        self.last_reward = float(self.state.reward)
        self.total_reward += self.last_reward
        self.step_count += 1
        self.done = bool(self.state.done)

    def _apply_current_command(self) -> None:
        if self.command is None:
            self.command_applied = False
            return
        self.state, self.command_applied = _apply_command(
            self.env,
            self.state,
            self.jp.asarray(self.command),
        )

    def _copy_state_to_mujoco_unlocked(self) -> None:
        self.mj_data.qpos[:] = np.asarray(self.state.data.qpos)
        self.mj_data.qvel[:] = np.asarray(self.state.data.qvel)
        for name in ("mocap_pos", "mocap_quat"):
            source = getattr(self.state.data, name, None)
            target = getattr(self.mj_data, name, None)
            if source is not None and target is not None:
                target[:] = np.asarray(source)
        self.mujoco.mj_forward(self.env.mj_model, self.mj_data)

    def _tracking_body_id(self) -> int:
        from simrig.rendering import tracking_body_id

        return tracking_body_id(self.mujoco, self.env.mj_model, self.mj_data)

    def _fallback_frame(self) -> np.ndarray:
        """Explicit schematic mode only; not used for default MuJoCo rendering."""
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        model = self.env.mj_model
        center = np.asarray(self.mj_data.xpos[self._tracking_body_id()])[:2]
        scale = min(self.width, self.height) / 5.0

        def project(pos: np.ndarray) -> tuple[int, int]:
            xy = (np.asarray(pos)[:2] - center) * scale
            return int(self.width / 2 + xy[0]), int(self.height / 2 - xy[1])

        # Ground grid.
        grid_color = (28, 34, 38)
        for offset in np.linspace(-2.0, 2.0, 9):
            x1, y1 = project(center + np.array([offset, -2.0]))
            x2, y2 = project(center + np.array([offset, 2.0]))
            draw.line((x1, y1, x2, y2), fill=grid_color)
            x1, y1 = project(center + np.array([-2.0, offset]))
            x2, y2 = project(center + np.array([2.0, offset]))
            draw.line((x1, y1, x2, y2), fill=grid_color)

        # Parent-child body graph.
        for body_id in range(1, model.nbody):
            parent = int(model.body_parentid[body_id])
            if parent <= 0:
                continue
            x1, y1 = project(self.mj_data.xpos[parent])
            x2, y2 = project(self.mj_data.xpos[body_id])
            draw.line((x1, y1, x2, y2), fill=(96, 150, 255), width=3)

        for body_id in range(1, model.nbody):
            x, y = project(self.mj_data.xpos[body_id])
            radius = 7 if body_id == self._tracking_body_id() else 4
            fill = (255, 210, 94) if body_id == self._tracking_body_id() else (205, 224, 255)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

        command = self.command if self.command is not None else ["-", "-", "-"]
        overlay = [
            "SimRig Preview",
            f"mode: {self.render_mode}",
            f"env: {self.env_name}",
            f"step: {self.step_count}",
            f"reward: {self.last_reward:.4f}",
            f"command: {command}",
        ]
        if self.renderer_error:
            overlay.append(f"render error: {self.renderer_error[:80]}")
        draw.rectangle((16, 16, min(self.width - 16, 720), 168), fill=(0, 0, 0))
        y = 28
        for line in overlay:
            draw.text((28, y), line, fill=(236, 240, 245))
            y += 22
        return np.asarray(image)

    def _error_frame(self, message: str) -> np.ndarray:
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        lines = [
            "SimRig Preview",
            "MuJoCo rendering failed",
            message[:240],
            "Try: export MUJOCO_GL=glfw",
            "Or use: simrig demo ... for native 3D viewer",
        ]
        y = 28
        for line in lines:
            draw.text((28, y), line, fill=(236, 240, 245))
            y += 24
        return np.asarray(image)

    def _plain_frame(self, message: str) -> np.ndarray:
        image = self.Image.new("RGB", (self.width, self.height), (8, 10, 12))
        draw = self.ImageDraw.Draw(image)
        draw.text((28, 28), "SimRig Preview", fill=(236, 240, 245))
        draw.text((28, 56), message, fill=(180, 190, 200))
        return np.asarray(image)


def serve_policy_preview(
    checkpoint: Path | str,
    *,
    env_name: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    backend: str = "mujoco-playground",
    small_network: bool | None = None,
    seed: int = 0,
    command: tuple[float, ...] | None = None,
    width: int = 960,
    height: int = 540,
    frame_skip: int = 1,
    camera: str | int | None = None,
    render_mode: str = "mujoco",
    paused: bool = False,
    fps: int = 24,
) -> None:
    session = PolicyPreviewSession(
        checkpoint,
        env_name=env_name,
        backend=backend,
        small_network=small_network,
        seed=seed,
        command=command,
        width=width,
        height=height,
        frame_skip=frame_skip,
        camera=camera,
        render_mode=render_mode,
        paused=paused,
        fps=fps,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_html())
            elif parsed.path == "/frame.jpg":
                self._send_bytes(session.frame_jpeg(), "image/jpeg")
            elif parsed.path == "/status.json":
                self._send_json(session.status_payload())
            elif parsed.path == "/camera":
                query = parse_qs(parsed.query)
                session.set_camera_from_query(query)
                self._send_json(session.status_payload())
            elif parsed.path == "/command":
                query = parse_qs(parsed.query)
                command_values = _command_from_query(query)
                session.set_command(command_values)
                self._send_json(session.status_payload())
            elif parsed.path == "/pause":
                session.set_paused(True)
                self._send_json(session.status_payload())
            elif parsed.path == "/resume":
                session.set_paused(False)
                self._send_json(session.status_payload())
            elif parsed.path == "/reset":
                session.reset()
                self._send_json(session.status_payload())
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
    print(f"SimRig preview: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()


def _command_from_query(query: dict[str, list[str]]) -> list[float] | None:
    if "clear" in query:
        return None
    values = []
    for key in ("x", "y", "yaw"):
        raw = query.get(key, ["0"])[0]
        values.append(float(raw))
    return values


def _html() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SimRig Preview</title>
  <style>{viewer_styles(sidebar_width=320)}</style>
</head>
<body>
  <main>
    <div id="viewport">
      <img id="frame" alt="SimRig rendered frame">
      <div id="hint">Drag to orbit · scroll to zoom</div>
    </div>
  </main>
  <aside>
    <h1>SimRig Preview</h1>
    <label>Forward X</label><input id="x" type="number" step="0.1" value="0">
    <label>Lateral Y</label><input id="y" type="number" step="0.1" value="0">
    <label>Yaw</label><input id="yaw" type="number" step="0.1" value="0">
    <button onclick="setCommand()">Set Command</button>
    <button class="secondary" onclick="clearCommand()">Clear</button>
    <button class="secondary" onclick="call('/pause')">Pause</button>
    <button class="secondary" onclick="call('/resume')">Resume</button>
    <button class="secondary" onclick="call('/reset')">Reset</button>
    <h1 style="margin-top:18px">Status</h1>
    <pre id="status">loading</pre>
  </aside>
  <script>
    const frame = document.getElementById('frame');
    const statusEl = document.getElementById('status');
    const xInput = document.getElementById('x');
    const yInput = document.getElementById('y');
    const yawInput = document.getElementById('yaw');
    let controlsInitialized = false;
    {camera_interaction_script()}
    {frame_poll_script(poll_ms=33)}
    bindCameraControls(frame);
    async function refreshStatus() {{
      try {{
        const res = await fetch('/status.json', {{cache: 'no-store'}});
        const status = await res.json();
        statusEl.textContent = JSON.stringify(status, null, 2);
        applyCameraFromStatus(status);
        if (!controlsInitialized && Array.isArray(status.command)) {{
          xInput.value = status.command[0] ?? 0;
          yInput.value = status.command[1] ?? 0;
          yawInput.value = status.command[2] ?? 0;
          controlsInitialized = true;
        }}
      }} catch (err) {{
        statusEl.textContent = String(err);
      }}
    }}
    async function call(path) {{ await fetch(path); await refreshStatus(); }}
    async function setCommand() {{
      const x = xInput.value;
      const y = yInput.value;
      const yaw = yawInput.value;
      await call(`/command?x=${{x}}&y=${{y}}&yaw=${{yaw}}`);
    }}
    async function clearCommand() {{
      controlsInitialized = false;
      await call('/command?clear=1');
    }}
    setInterval(refreshStatus, 500);
    refreshFrame();
    refreshStatus();
  </script>
</body>
</html>
"""
