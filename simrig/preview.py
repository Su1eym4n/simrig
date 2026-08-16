"""Browser preview server for policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import functools
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from simrig.browser_render import MujocoFramePump
from simrig.browser_shell import camera_interaction_script, frame_poll_script, viewer_styles
from simrig.playground_backend import (
    _apply_command,
    _checkpoint_env_overrides,
    _import_training_deps,
    _validate_backend,
    load_env,
)
from simrig.presets import resolve_network_factory
from simrig.rendering import make_tracking_camera, tracking_body_id
from simrig.runtime import verify_checkpoint_runtime
from simrig.three_scene import geom_transforms, scene_payload


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
        render_mode: str = "threejs",
        paused: bool = False,
        fps: int = 24,
        allow_runtime_mismatch: bool = False,
    ) -> None:
        _validate_backend(backend)
        self.runtime_compatibility = verify_checkpoint_runtime(
            checkpoint,
            allow_mismatch=allow_runtime_mismatch,
        )
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
        self.fps = max(1, int(fps))
        self.renderer_error: str | None = None
        self._lock = threading.Lock()
        self._last_frame_jpeg: bytes | None = None
        self._scene_payload: dict[str, Any] | None = None

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
        self.env = load_env(
            env_name,
            config_overrides=_checkpoint_env_overrides(checkpoint),
        )
        network_config = resolve_network_factory(
            checkpoint,
            small_network=small_network,
        )
        network_factory = functools.partial(
            self.ppo_networks.make_ppo_networks,
            **network_config,
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
        self._copy_state_to_mujoco_unlocked()
        self.width = width
        self.height = height
        self.renderer = None
        if self.render_mode not in ("threejs", "topdown", "mujoco"):
            raise ValueError("render_mode must be 'threejs', 'mujoco', or 'topdown'.")
        self.camera, self.camera_state = make_tracking_camera(
            self.mujoco,
            self.env.mj_model,
            self.mj_data,
            camera,
        )
        self._frame_pump: MujocoFramePump | None = None
        self._rollout_thread: threading.Thread | None = None
        self._running = True
        if self.render_mode == "threejs":
            self._rollout_thread = threading.Thread(
                target=self._run_rollout,
                name="simrig-preview-rollout",
                daemon=True,
            )
            self._rollout_thread.start()
        else:
            self._frame_pump = MujocoFramePump(
                self.mujoco,
                self.env.mj_model,
                self.mj_data,
                width=width,
                height=height,
                camera=self.camera,
                camera_state=self.camera_state,
                image_module=self.Image,
                fps=self.fps,
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
        with self._lock:
            return self._status_unlocked()

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = self._status_unlocked().__dict__
            payload.update(self._renderer_stats())
            return payload

    def state_payload(self) -> dict[str, Any]:
        """Return current rollout metadata and world transforms for Three.js."""

        with self._lock:
            payload = self._status_unlocked().__dict__
            payload.update(self._renderer_stats())
            payload["transforms"] = geom_transforms(self.env.mj_model, self.mj_data)
            payload["tracking_position"] = np.asarray(
                self.mj_data.xpos[self._tracking_body_id()],
                dtype=float,
            ).tolist()
            return payload

    def scene_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._scene_payload is None:
                self._scene_payload = scene_payload(
                    self.mujoco,
                    self.env.mj_model,
                    self.mj_data,
                    model_name=self.env_name,
                )
                self._scene_payload.pop("transforms", None)
            return {
                **self._scene_payload,
                "transforms": geom_transforms(self.env.mj_model, self.mj_data),
                "tracking_position": np.asarray(
                    self.mj_data.xpos[self._tracking_body_id()],
                    dtype=float,
                ).tolist(),
                "fps_target": self.fps,
            }

    def _status_unlocked(self) -> PreviewStatus:
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
            render_mode=self.render_mode,
            frame_busy=False,
            renderer_error=self._renderer_stats()["renderer_error"],
        )

    def _renderer_stats(self) -> dict[str, Any]:
        if self._frame_pump is not None:
            return self._frame_pump.stats()
        return {
            "fps_target": self.fps,
            "renderer_error": self.renderer_error,
            "camera": {
                "interactive": True,
                "renderer": "threejs-orbit-controls",
            },
        }

    def set_camera_from_query(self, query: dict[str, list[str]]) -> None:
        if self._frame_pump is not None:
            self._frame_pump.set_camera_from_query(query)

    def frame_jpeg(self) -> bytes:
        if self._frame_pump is None:
            raise RuntimeError("Frame streaming is disabled in threejs mode.")
        return self._frame_pump.get_jpeg()

    def close(self) -> None:
        self._running = False
        if self._frame_pump is not None:
            self._frame_pump.close()
        if self._rollout_thread is not None:
            self._rollout_thread.join(timeout=2.0)

    def _run_rollout(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            started = time.monotonic()
            try:
                with self._lock:
                    self._advance_rollout()
            except Exception as exc:
                self.renderer_error = str(exc)
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, interval - elapsed))

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
    render_mode: str = "threejs",
    paused: bool = False,
    fps: int = 24,
    allow_runtime_mismatch: bool = False,
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
        allow_runtime_mismatch=allow_runtime_mismatch,
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
            elif parsed.path == "/scene.json":
                self._send_json(session.scene_payload(), compress=True)
            elif parsed.path == "/state.json":
                self._send_json(session.state_payload())
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
  <title>SimRig Preview</title>
  <style>"""
        + viewer_styles(sidebar_width=320)
        + """
    #three-view { width: 100%; height: 100%; display: block; outline: none; }
    #loading { position: absolute; inset: 0; display: grid; place-items: center; color: #cbd5e1; background: #070b12; z-index: 2; }
    #loading.error { color: #fca5a5; padding: 28px; text-align: center; white-space: pre-wrap; }
    #render-meta { color: #94a3b8; font-size: 12px; margin: -4px 0 12px; }
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
      <canvas id="three-view" aria-label="Interactive SimRig policy preview"></canvas>
      <div id="loading">Loading rollout scene…</div>
      <div id="hint">Drag to orbit · scroll to zoom · right-drag to pan</div>
    </div>
  </main>
  <aside>
    <h1>SimRig Preview</h1>
    <div id="render-meta">Three.js · loading rollout…</div>
    <label>Forward X</label><input id="x" type="number" step="0.1" value="0">
    <label>Lateral Y</label><input id="y" type="number" step="0.1" value="0">
    <label>Yaw</label><input id="yaw" type="number" step="0.1" value="0">
    <button id="set-command">Set Command</button>
    <button class="secondary" id="clear-command">Clear</button>
    <button class="secondary" id="pause">Pause</button>
    <button class="secondary" id="resume">Resume</button>
    <button class="secondary" id="reset">Reset</button>
    <button class="secondary" id="reset-camera">Reset Camera</button>
    <h1 style="margin-top:18px">Status</h1>
    <pre id="status">loading</pre>
  </aside>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const canvas = document.getElementById('three-view');
    const viewport = document.getElementById('viewport');
    const loadingEl = document.getElementById('loading');
    const statusEl = document.getElementById('status');
    const renderMetaEl = document.getElementById('render-meta');
    const xInput = document.getElementById('x');
    const yInput = document.getElementById('y');
    const yawInput = document.getElementById('yaw');
    const objects = new Map();
    const meshGeometries = new Map();
    let controlsInitialized = false;
    let trackingPosition = null;
    let stateTimer = null;
    let targetPollMs = 33;

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
    scene.fog = new THREE.Fog(0x0b1220, 8, 26);

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
    scene.add(keyLight, keyLight.target);
    const rimLight = new THREE.DirectionalLight(0x7aa8ff, 0.85);
    rimLight.position.set(-5, 3, 5);
    scene.add(rimLight);

    const modelRoot = new THREE.Group();
    scene.add(modelRoot);

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
        case 0:
          return new THREE.PlaneGeometry(200, 200);
        case 2:
          return new THREE.SphereGeometry(x, 32, 20);
        case 3: {
          const geometry = new THREE.CapsuleGeometry(x, 2 * y, 10, 24);
          geometry.rotateX(Math.PI / 2);
          return geometry;
        }
        case 4: {
          const geometry = new THREE.SphereGeometry(1, 32, 20);
          geometry.scale(x, y, z);
          return geometry;
        }
        case 5: {
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

    function followRobot(rawPosition) {
      if (!Array.isArray(rawPosition)) return;
      const next = new THREE.Vector3().fromArray(rawPosition);
      if (trackingPosition !== null) {
        const delta = next.clone().sub(trackingPosition);
        camera.position.add(delta);
        orbit.target.add(delta);
        keyLight.position.add(delta);
        keyLight.target.position.add(delta);
      }
      trackingPosition = next;
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
      trackingPosition = null;
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
      requestAnimationFrame(animate);
    }

    async function loadScene() {
      const res = await fetch('/scene.json', {cache: 'no-store'});
      if (!res.ok) throw new Error(`scene request failed (${res.status})`);
      const payload = await res.json();
      targetPollMs = Math.max(16, Math.round(1000 / (payload.fps_target || 24)));

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

      const grid = new THREE.GridHelper(40, 80, 0x52647a, 0x263346);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = 0.001;
      grid.material.opacity = 0.42;
      grid.material.transparent = true;
      scene.add(grid);

      fitCamera();
      followRobot(payload.tracking_position);
      loadingEl.remove();
    }

    function displayStatus(status) {
      const copy = {...status};
      delete copy.transforms;
      delete copy.tracking_position;
      statusEl.textContent = JSON.stringify(copy, null, 2);
      renderMetaEl.textContent = `${status.env_name} · step ${status.step} · ${status.fps_target || 24} Hz physics · display-rate WebGL`;
      if (!controlsInitialized && Array.isArray(status.command)) {
        xInput.value = status.command[0] ?? 0;
        yInput.value = status.command[1] ?? 0;
        yawInput.value = status.command[2] ?? 0;
        controlsInitialized = true;
      }
    }

    async function refreshState() {
      clearTimeout(stateTimer);
      try {
        const res = await fetch('/state.json', {cache: 'no-store'});
        if (!res.ok) throw new Error(`state request failed (${res.status})`);
        const status = await res.json();
        updateTransforms(status.transforms);
        followRobot(status.tracking_position);
        displayStatus(status);
      } catch (err) {
        statusEl.textContent = String(err);
      } finally {
        stateTimer = setTimeout(refreshState, targetPollMs);
      }
    }

    async function call(path) {
      const res = await fetch(path, {cache: 'no-store'});
      displayStatus(await res.json());
    }

    async function setCommand() {
      const params = new URLSearchParams({x: xInput.value, y: yInput.value, yaw: yawInput.value});
      await call('/command?' + params.toString());
    }

    async function clearCommand() {
      controlsInitialized = false;
      await call('/command?clear=1');
    }

    document.getElementById('set-command').addEventListener('click', setCommand);
    document.getElementById('clear-command').addEventListener('click', clearCommand);
    document.getElementById('pause').addEventListener('click', () => call('/pause'));
    document.getElementById('resume').addEventListener('click', () => call('/resume'));
    document.getElementById('reset').addEventListener('click', () => call('/reset'));
    document.getElementById('reset-camera').addEventListener('click', fitCamera);
    window.addEventListener('resize', resize);
    resize();
    animate();

    try {
      await loadScene();
      await refreshState();
    } catch (err) {
      loadingEl.className = 'error';
      loadingEl.textContent = `WebGL preview failed to load.\n${err}\n\nThree.js is loaded from jsDelivr, so an internet connection is required.`;
      statusEl.textContent = String(err);
      console.error(err);
    }
  </script>
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
