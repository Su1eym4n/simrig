"""Background MuJoCo frame rendering for browser viewers."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

from simrig.rendering import CameraState, create_mujoco_renderer


def encode_jpeg(image_module: Any, frame: np.ndarray, *, quality: int = 72) -> bytes:
    from io import BytesIO

    image = image_module.fromarray(np.asarray(frame))
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


class MujocoFramePump:
    """Render MuJoCo frames on a dedicated thread and serve cached JPEGs."""

    def __init__(
        self,
        mujoco: Any,
        model: Any,
        mj_data: Any,
        *,
        width: int,
        height: int,
        camera: Any,
        camera_state: CameraState,
        image_module: Any,
        fps: int = 24,
        jpeg_quality: int = 72,
        render_mode: str = "mujoco",
        scene_lock: threading.Lock | None = None,
        before_render: Callable[[], None] | None = None,
        fallback_frame: Callable[[], np.ndarray] | None = None,
        error_frame: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.mj_data = mj_data
        self.width = width
        self.height = height
        self.camera = camera
        self.camera_state = camera_state
        self.image_module = image_module
        self.fps = max(1, int(fps))
        self.jpeg_quality = jpeg_quality
        self.render_mode = render_mode.lower()
        self.scene_lock = scene_lock or threading.Lock()
        self.before_render = before_render
        self.fallback_frame = fallback_frame
        self.error_frame = error_frame

        self.renderer_error: str | None = None
        self._latest_jpeg: bytes = b""
        self._camera_lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="simrig-frame-pump", daemon=True)
        self._thread.start()

    def set_camera_from_query(self, query: dict[str, list[str]]) -> None:
        with self._camera_lock:
            self.camera_state.update_from_query(query)
            self.camera_state.apply(self.camera)

    def get_jpeg(self) -> bytes:
        return self._latest_jpeg

    def stats(self) -> dict[str, Any]:
        return {
            "fps_target": self.fps,
            "renderer_error": self.renderer_error,
            "camera": self.camera_state.to_dict(),
        }

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        renderer = None
        interval = 1.0 / self.fps
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._latest_jpeg = encode_jpeg(self.image_module, placeholder, quality=self.jpeg_quality)

        if self.render_mode == "mujoco":
            try:
                renderer = create_mujoco_renderer(
                    self.mujoco,
                    self.model,
                    height=self.height,
                    width=self.width,
                )
            except Exception as exc:
                self.renderer_error = str(exc)

        while self._running:
            started = time.monotonic()
            try:
                with self.scene_lock:
                    if self.before_render is not None:
                        self.before_render()
                    if self.render_mode == "topdown":
                        if self.fallback_frame is None:
                            raise RuntimeError("topdown mode requires fallback_frame.")
                        frame = self.fallback_frame()
                    elif renderer is None:
                        message = self.renderer_error or "MuJoCo renderer unavailable."
                        if self.error_frame is None:
                            raise RuntimeError(message)
                        frame = self.error_frame(message)
                    else:
                        with self._camera_lock:
                            camera = self.camera_state.apply(self.camera)
                        renderer.update_scene(self.mj_data, camera=camera)
                        frame = renderer.render()
                        self.renderer_error = None
                self._latest_jpeg = encode_jpeg(
                    self.image_module,
                    frame,
                    quality=self.jpeg_quality,
                )
            except Exception as exc:
                self.renderer_error = str(exc)
                if self.error_frame is not None:
                    self._latest_jpeg = encode_jpeg(
                        self.image_module,
                        self.error_frame(str(exc)),
                        quality=self.jpeg_quality,
                    )
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, interval - elapsed))

        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
