"""Three.js viewer for MuJoCo simulations owned by ordinary Python scripts."""

from __future__ import annotations

import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any
from urllib.parse import urlparse

import numpy as np

from simrig.browser_shell import viewer_chrome_script, viewer_styles
from simrig.three_scene import geom_transforms, scene_payload


class LiveWebViewer:
    """Serve an existing ``MjModel``/``MjData`` pair without owning its loop.

    The calling script remains responsible for control and ``mj_step``. Use
    :attr:`lock` around mutations of ``data`` so HTTP snapshots are consistent.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        name: str = "MuJoCo script",
        host: str = "127.0.0.1",
        port: int = 8767,
        fps: int = 30,
        tracking_body: str | int | None = None,
        tracking_site: str | int | None = None,
        allow_reset: bool = False,
        mujoco_module: Any | None = None,
    ) -> None:
        if fps < 1:
            raise ValueError("fps must be at least 1")
        if port < 0 or port > 65535:
            raise ValueError("port must be between 0 and 65535")

        if mujoco_module is None:
            try:
                import mujoco as mujoco_module  # type: ignore
            except ImportError as exc:
                raise RuntimeError("LiveWebViewer requires MuJoCo.") from exc

        self.mujoco = mujoco_module
        self.model = model
        self.data = data
        self.name = name
        self.host = host
        self.port = port
        self.fps = int(fps)
        self.allow_reset = bool(allow_reset)
        self.lock = threading.RLock()
        if tracking_body is not None and tracking_site is not None:
            raise ValueError("Choose tracking_body or tracking_site, not both")
        self._tracking_body_id, self._tracking_body_name = self._resolve_tracking_body(tracking_body)
        self._tracking_site_id, self._tracking_site_name = self._resolve_tracking_site(tracking_site)
        self._scene: dict[str, Any] | None = None
        self._status: dict[str, Any] = {}
        self._frame = 0
        self._state = "starting"
        self._client_event = threading.Event()
        self._reset_event = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Return the browser URL, including an OS-assigned port after start."""

        port = self.port
        if self._server is not None:
            port = int(self._server.server_address[1])
        display_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{display_host}:{port}/"

    @property
    def is_running(self) -> bool:
        return self._server is not None

    def start(self) -> "LiveWebViewer":
        """Start the local HTTP server in a background thread."""

        if self._server is not None:
            return self

        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_bytes(_live_html().encode("utf-8"), "text/html; charset=utf-8")
                elif parsed.path == "/scene.json":
                    viewer._note_client()
                    self._send_json(viewer.scene_payload(), compress=True)
                elif parsed.path == "/state.json":
                    viewer._note_client()
                    self._send_json(viewer.state_payload())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/reset":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                if not viewer.allow_reset:
                    self.send_error(HTTPStatus.CONFLICT, "Simulation reset is unavailable")
                    return
                viewer.request_reset()
                self._send_json({"reset_requested": True})

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, value: dict[str, Any], *, compress: bool = False) -> None:
                body = json.dumps(value, separators=(",", ":")).encode("utf-8")
                if compress and "gzip" in self.headers.get("Accept-Encoding", ""):
                    self._send_bytes(
                        gzip.compress(body, compresslevel=5),
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
                try:
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers may cancel a polling request during navigation or
                    # refresh. The simulator and other viewer clients continue.
                    return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._state = "running"
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="simrig-live-web-viewer",
            daemon=True,
        )
        self._server_thread.start()
        print(f"SimRig live viewer: {self.url}")
        return self

    def close(self) -> None:
        """Stop the HTTP server. Safe to call more than once."""

        server = self._server
        thread = self._server_thread
        if server is None:
            return
        with self.lock:
            self._state = "closed"
        server.shutdown()
        server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._server = None
        self._server_thread = None

    def sync(self, **status: Any) -> None:
        """Record a completed simulation step and optional status fields."""

        with self.lock:
            self._frame += 1
            self._status.update(status)

    def update_status(self, **status: Any) -> None:
        """Publish script-specific scalar or JSON-compatible metadata."""

        with self.lock:
            self._status.update(status)

    def mark_complete(self, **status: Any) -> None:
        """Mark the script complete while keeping its last pose available."""

        with self.lock:
            self._state = "complete"
            self._status.update(status)

    def wait_for_client(self, timeout: float | None = None) -> bool:
        """Wait until the browser requests the Three.js scene or live state."""

        return self._client_event.wait(timeout)

    def request_reset(self) -> None:
        """Signal that the script-owned simulation should reset."""

        if self.allow_reset:
            self._reset_event.set()

    def consume_reset_request(self) -> bool:
        """Return and clear the browser's pending reset request."""

        if not self._reset_event.is_set():
            return False
        self._reset_event.clear()
        return True

    def scene_payload(self) -> dict[str, Any]:
        """Return static geometry and the current initial transforms."""

        with self.lock:
            if self._scene is None:
                self._scene = scene_payload(
                    self.mujoco,
                    self.model,
                    self.data,
                    model_name=self.name,
                )
                self._scene.pop("transforms", None)
            return {
                **self._scene,
                "transforms": geom_transforms(self.model, self.data),
                "tracking_position": self._tracking_position(),
                "tracking_body": self._tracking_body_name,
                "tracking_site": self._tracking_site_name,
                "fps_target": self.fps,
                "reset_available": self.allow_reset,
            }

    def state_payload(self) -> dict[str, Any]:
        """Return the latest live transforms and script metadata."""

        with self.lock:
            return {
                "name": self.name,
                "state": self._state,
                "time": float(self.data.time),
                "frame": self._frame,
                "viewer_connected": self._client_event.is_set(),
                "fps_target": self.fps,
                "tracking_body": self._tracking_body_name,
                "tracking_site": self._tracking_site_name,
                "tracking_position": self._tracking_position(),
                "reset_available": self.allow_reset,
                "transforms": geom_transforms(self.model, self.data),
                **self._status,
            }

    def __enter__(self) -> "LiveWebViewer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _resolve_tracking_body(self, body: str | int | None) -> tuple[int | None, str | None]:
        if body is None:
            return None, None
        if isinstance(body, int):
            body_id = body
            if body_id < 0 or body_id >= self.model.nbody:
                raise ValueError(f"Invalid tracking body id: {body_id}")
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            return body_id, name or f"body_{body_id}"
        body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, body
        )
        if body_id < 0:
            raise ValueError(f"Model has no body named {body!r}")
        return int(body_id), body

    def _resolve_tracking_site(self, site: str | int | None) -> tuple[int | None, str | None]:
        if site is None:
            return None, None
        if isinstance(site, int):
            site_id = site
            if site_id < 0 or site_id >= self.model.nsite:
                raise ValueError(f"Invalid tracking site id: {site_id}")
            name = self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_id
            )
            return site_id, name or f"site_{site_id}"
        site_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, site
        )
        if site_id < 0:
            raise ValueError(f"Model has no site named {site!r}")
        return int(site_id), site

    def _tracking_position(self) -> list[float] | None:
        if self._tracking_site_id is not None:
            return np.asarray(
                self.data.site_xpos[self._tracking_site_id], dtype=float
            ).tolist()
        if self._tracking_body_id is None:
            return None
        return np.asarray(
            self.data.xpos[self._tracking_body_id], dtype=float
        ).tolist()

    def _note_client(self) -> None:
        self._client_event.set()


def _live_html() -> str:
    return (
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SimRig Live</title>
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
      <canvas id="three-view" aria-label="Interactive live MuJoCo script"></canvas>
      <div id="loading">Loading live scene…</div>
      <div id="hint">Drag to orbit · scroll to zoom · right-drag to pan</div>
    </div>
  </main>
  <aside>
    <h1>SimRig Live</h1>
    <div id="render-meta">Three.js · connecting to script…</div>
    <button id="reset-simulation" hidden>Reset Simulation</button>
    <button class="secondary" id="reset-camera">Reset Camera</button>
    <button class="secondary" id="clear-trail">Clear Trail</button>
    <button class="secondary" id="hide-trail">Show Trail</button>
    <details class="simrig-debug">
      <summary>Script State</summary>
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
    const renderMetaEl = document.getElementById('render-meta');
    const objects = new Map();
    const meshGeometries = new Map();
    const trailPoints = [];
    let trail = null;
    let trailVisible = false;
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
        color: new THREE.Color(r, g, b), opacity: a, transparent: a < 0.999,
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
        case 0: return new THREE.PlaneGeometry(200, 200);
        case 2: return new THREE.SphereGeometry(x, 32, 20);
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
        case 6: return new THREE.BoxGeometry(2 * x, 2 * y, 2 * z);
        case 7: return meshGeometries.get(geom.mesh_id) || null;
        default: return null;
      }
    }

    function applyTransform(object, transform) {
      if (!object || !transform) return;
      object.position.fromArray(transform.position);
      const m = transform.matrix;
      const rotation = new THREE.Matrix4();
      rotation.set(
        m[0], m[1], m[2], 0, m[3], m[4], m[5], 0,
        m[6], m[7], m[8], 0, 0, 0, 0, 1,
      );
      object.quaternion.setFromRotationMatrix(rotation);
    }

    function updateTransforms(transforms) {
      for (const transform of transforms || []) applyTransform(objects.get(transform.id), transform);
    }

    function updateTrail(rawPosition) {
      if (!Array.isArray(rawPosition)) return;
      const next = new THREE.Vector3().fromArray(rawPosition);
      if (trailPoints.length === 0 || trailPoints[trailPoints.length - 1].distanceTo(next) > 0.001) {
        trailPoints.push(next.clone());
        if (trailPoints.length > 4000) trailPoints.shift();
        if (trail === null) {
          trail = new THREE.Line(
            new THREE.BufferGeometry(),
            new THREE.LineBasicMaterial({color: 0x38bdf8}),
          );
          trail.visible = trailVisible;
          scene.add(trail);
        }
        trail.geometry.dispose();
        trail.geometry = new THREE.BufferGeometry().setFromPoints(trailPoints);
      }
    }

    function setTrailVisible(visible) {
      trailVisible = visible;
      if (trail !== null) trail.visible = visible;
      document.getElementById('hide-trail').textContent = visible ? 'Hide Trail' : 'Show Trail';
    }

    function clearTrail() {
      trailPoints.length = 0;
      if (trail !== null) {
        trail.geometry.dispose();
        scene.remove(trail);
        trail = null;
      }
    }

    function fitCamera() {
      const bounds = new THREE.Box3().setFromObject(modelRoot);
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.25);
      orbit.target.copy(center);
      camera.position.set(center.x + radius * 1.35, center.y - radius * 1.75, center.z + radius * 0.95);
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
      requestAnimationFrame(animate);
    }

    async function loadScene() {
      const res = await fetch('/scene.json', {cache: 'no-store'});
      if (!res.ok) throw new Error(`scene request failed (${res.status})`);
      const payload = await res.json();
      document.getElementById('reset-simulation').hidden = !payload.reset_available;
      targetPollMs = Math.max(16, Math.round(1000 / (payload.fps_target || 30)));
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
        if (geom.type === 0) scene.add(object); else modelRoot.add(object);
        objects.set(geom.id, object);
      }
      const grid = new THREE.GridHelper(40, 80, 0x52647a, 0x263346);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = 0.001;
      grid.material.opacity = 0.42;
      grid.material.transparent = true;
      scene.add(grid);
      fitCamera();
      updateTrail(payload.tracking_position);
      loadingEl.remove();
    }

    function displayStatus(status) {
      const copy = {...status};
      delete copy.transforms;
      delete copy.tracking_position;
      statusEl.textContent = JSON.stringify(copy, null, 2);
      renderMetaEl.textContent = `${status.name} · ${status.time.toFixed(2)} s · frame ${status.frame}`;
    }

    async function refreshState() {
      clearTimeout(stateTimer);
      try {
        const res = await fetch('/state.json', {cache: 'no-store'});
        if (!res.ok) throw new Error(`state request failed (${res.status})`);
        const status = await res.json();
        updateTransforms(status.transforms);
        updateTrail(status.tracking_position);
        displayStatus(status);
      } catch (err) {
        statusEl.textContent = String(err);
      } finally {
        stateTimer = setTimeout(refreshState, targetPollMs);
      }
    }

    async function resetSimulation() {
      const button = document.getElementById('reset-simulation');
      button.disabled = true;
      try {
        const res = await fetch('/reset', {method: 'POST', cache: 'no-store'});
        if (!res.ok) throw new Error(`reset request failed (${res.status})`);
        clearTrail();
      } catch (err) {
        statusEl.textContent = String(err);
      } finally {
        button.disabled = false;
      }
    }

    document.getElementById('reset-simulation').addEventListener('click', resetSimulation);
    document.getElementById('reset-camera').addEventListener('click', fitCamera);
    document.getElementById('clear-trail').addEventListener('click', clearTrail);
    document.getElementById('hide-trail').addEventListener('click', () => {
      setTrailVisible(!trailVisible);
    });
    window.addEventListener('resize', resize);
    resize();
    animate();
    try {
      await loadScene();
      await refreshState();
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
