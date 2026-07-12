"""MuJoCo offscreen rendering helpers for browser viewers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any


@dataclass
class CameraState:
    azimuth: float = 135.0
    elevation: float = -20.0
    distance: float = 2.4
    interactive: bool = True

    def apply(self, camera: Any) -> Any:
        if not self.interactive or not hasattr(camera, "azimuth"):
            return camera
        camera.azimuth = float(self.azimuth)
        camera.elevation = float(self.elevation)
        camera.distance = float(self.distance)
        return camera

    def update_from_query(self, query: dict[str, list[str]]) -> None:
        for key in ("azimuth", "elevation", "distance"):
            if key in query:
                setattr(self, key, float(query[key][0]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "interactive": self.interactive,
        }


def preferred_gl_backends() -> list[str | None]:
    """Return MuJoCo GL backends to try, in order."""

    configured = os.environ.get("MUJOCO_GL")
    if configured:
        return [configured]
    if sys.platform == "darwin":
        return ["glfw", None]
    if os.environ.get("DISPLAY"):
        return ["egl", "glfw", None]
    return ["osmesa", "egl", None]


def ensure_offscreen_framebuffer(model: Any, *, height: int, width: int) -> None:
    """Grow MuJoCo's offscreen buffer so browser frames can exceed 640x480."""

    vis_global = model.vis.global_
    vis_global.offwidth = max(int(vis_global.offwidth), int(width))
    vis_global.offheight = max(int(vis_global.offheight), int(height))


def create_mujoco_renderer(
    mujoco: Any,
    model: Any,
    *,
    height: int,
    width: int,
) -> Any:
    """Create a MuJoCo offscreen renderer, trying platform-appropriate GL backends."""

    ensure_offscreen_framebuffer(model, height=height, width=width)
    errors: list[str] = []
    for backend in preferred_gl_backends():
        if backend:
            os.environ["MUJOCO_GL"] = backend
        try:
            return mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:
            errors.append(f"{backend or 'default'}: {exc}")
    raise RuntimeError(
        "MuJoCo offscreen rendering failed. Tried "
        + "; ".join(errors)
        + ". On macOS, run from a desktop session and keep MUJOCO_GL=glfw."
    )


def make_tracking_camera(
    mujoco: Any,
    model: Any,
    data: Any,
    camera: str | int | None,
) -> tuple[Any, CameraState]:
    """Build a MuJoCo camera object and mutable browser camera state."""

    if isinstance(camera, int):
        return camera, CameraState(interactive=False)
    if isinstance(camera, str) and camera.isdigit():
        return int(camera), CameraState(interactive=False)
    if camera not in (None, "track", "tracking"):
        return camera, CameraState(interactive=False)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = tracking_body_id(mujoco, model, data)
    cam.distance = 2.4
    cam.azimuth = 135
    cam.elevation = -20
    state = CameraState(
        azimuth=float(cam.azimuth),
        elevation=float(cam.elevation),
        distance=float(cam.distance),
        interactive=True,
    )
    return cam, state


def tracking_body_id(mujoco: Any, model: Any, data: Any | None = None) -> int:
    """Pick a stable body for camera tracking."""

    del data
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[joint_id])
    for name in ("trunk", "base", "torso", "torso_link", "pelvis", "body", "root"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
    return 1 if model.nbody > 1 else 0
