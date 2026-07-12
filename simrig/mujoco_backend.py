"""MuJoCo model discovery and inspection backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simrig.core import BackendInfo, ModelInspectionReport, TrainabilityStatus
from simrig.paths import find_menagerie


def _import_mujoco():
    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed. Install SimRig with the mujoco extra or "
            "install the `mujoco` package."
        ) from exc
    return mujoco


def backend_info() -> BackendInfo:
    try:
        mujoco = _import_mujoco()
    except RuntimeError as exc:
        return BackendInfo(name="mujoco", available=False, detail=str(exc))
    version = getattr(mujoco, "__version__", None)
    return BackendInfo(name="mujoco", available=True, version=version)


def list_models(menagerie: Path | str | None = None) -> list[dict[str, Any]]:
    """List model directories in a MuJoCo Menagerie checkout."""
    root = find_menagerie(menagerie)
    entries: list[dict[str, Any]] = []
    skip = {".git", ".github", "assets", "test", "opensource"}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if directory.name.startswith(".") or directory.name in skip:
            continue
        xmls = sorted(directory.glob("*.xml"))
        if not xmls:
            continue
        scenes = sorted(directory.glob("scene*.xml"))
        mjx_xmls = sorted(directory.glob("*mjx*.xml"))
        entries.append(
            {
                "name": directory.name,
                "path": str(directory),
                "scene_xmls": [str(path.relative_to(root)) for path in scenes],
                "xmls": [str(path.relative_to(root)) for path in xmls],
                "mjx_xmls": [str(path.relative_to(root)) for path in mjx_xmls],
                "has_readme": (directory / "README.md").is_file(),
                "has_license": (directory / "LICENSE").is_file(),
            }
        )
    return entries


def resolve_model_path(model_or_xml: str | Path, menagerie: Path | str | None = None) -> Path:
    """Resolve a model directory name, relative path, or XML path."""
    raw = Path(model_or_xml).expanduser()
    if raw.is_file():
        return raw.resolve()
    if raw.is_dir():
        scene = _preferred_scene(raw)
        if scene is None:
            raise FileNotFoundError(f"No scene*.xml or *.xml found in {raw}")
        return scene.resolve()

    root = find_menagerie(menagerie)
    candidate = root / str(model_or_xml)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_dir():
        scene = _preferred_scene(candidate)
        if scene is not None:
            return scene.resolve()

    # Accept relative XML paths inside Menagerie, e.g. unitree_g1/scene.xml.
    relative_candidate = root / raw
    if relative_candidate.is_file():
        return relative_candidate.resolve()

    raise FileNotFoundError(f"Could not resolve model or XML: {model_or_xml}")


def inspect_model(
    model_or_xml: str | Path,
    *,
    menagerie: Path | str | None = None,
    steps: int = 25,
    noise_scale: float = 0.25,
) -> ModelInspectionReport:
    """Compile and briefly step a MuJoCo model."""
    path = resolve_model_path(model_or_xml, menagerie)
    mujoco = _import_mujoco()
    warnings: list[str] = []
    errors: list[str] = []
    notes: list[str] = []
    compiled = False
    stepped = False
    model = None

    try:
        model = mujoco.MjModel.from_xml_path(str(path))
        compiled = True
    except Exception as exc:  # MuJoCo raises several native exception types.
        errors.append(str(exc))
        return ModelInspectionReport(
            name=path.stem,
            path=str(path),
            backend="mujoco",
            status=TrainabilityStatus.FAILED,
            compiled=False,
            stepped=False,
            errors=errors,
        )

    data = mujoco.MjData(model)
    try:
        for index in range(max(0, steps)):
            _bounded_ctrl_noise(mujoco, model, data, index, noise_scale)
            mujoco.mj_step(model, data)
        stepped = True
    except Exception as exc:
        errors.append(str(exc))

    for warning_index, count in enumerate(data.warning.number):
        if count:
            try:
                name = mujoco.mjtWarning(warning_index).name
            except Exception:
                name = f"warning_{warning_index}"
            warnings.append(f"{name}: count={int(count)}")

    has_mjx_hint = "mjx" in path.name.lower() or any(
        "mjx" in sibling.name.lower() for sibling in path.parent.glob("*.xml")
    )
    has_freejoint = _has_freejoint(mujoco, model)
    if model.nu <= 0:
        notes.append("Model has no actuators; it can be inspected but is not directly controllable.")
    if not has_mjx_hint:
        notes.append("No MJX-named XML variant found near this model.")
    notes.append(
        "Raw MuJoCo model inspection does not prove trainability; a task env still defines observations, rewards, resets, and termination."
    )

    status = TrainabilityStatus.SIMULATABLE if compiled and stepped and not errors else TrainabilityStatus.INSPECTABLE
    return ModelInspectionReport(
        name=path.parent.name if path.parent.name else path.stem,
        path=str(path),
        backend="mujoco",
        status=status,
        compiled=compiled,
        stepped=stepped,
        bodies=int(model.nbody),
        joints=int(model.njnt),
        dofs=int(model.nv),
        actuators=int(model.nu),
        sensors=int(model.nsensor),
        keyframes=int(model.nkey),
        has_freejoint=has_freejoint,
        has_mjx_hint=has_mjx_hint,
        warnings=warnings,
        errors=errors,
        notes=notes,
    )


def _preferred_scene(directory: Path) -> Path | None:
    for pattern in ("scene_mjx.xml", "scene.xml", "scene*.xml", "*.xml"):
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _bounded_ctrl_noise(mujoco: Any, model: Any, data: Any, index: int, noise: float) -> None:
    for actuator_index in range(model.nu):
        ctrlrange = model.actuator_ctrlrange[actuator_index]
        if model.actuator_ctrllimited[actuator_index]:
            center = 0.5 * (ctrlrange[1] + ctrlrange[0])
            radius = 0.5 * (ctrlrange[1] - ctrlrange[0])
        else:
            center = 0.0
            radius = 1.0
        data.ctrl[actuator_index] = center + radius * noise * (
            2 * mujoco.mju_Halton(index + 1, actuator_index + 2) - 1
        )


def _has_freejoint(mujoco: Any, model: Any) -> bool:
    for joint_index in range(model.njnt):
        if model.jnt_type[joint_index] == mujoco.mjtJoint.mjJNT_FREE:
            return True
    return False

