"""Validation for custom env starter modules (static + optional runtime)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from simrig.custom_env import (
    load_custom_env_metadata,
    load_custom_env_static_metadata,
)
from simrig.networks import VISION_CNN_NETWORK
from simrig.scaffold import REQUIRED_CLASS_METHODS, REQUIRED_SECTION_MARKERS


@dataclass(frozen=True)
class EnvValidationResult:
    """Result of custom-env validation.

    ``trainable`` is True only when ``--runtime`` reset/step checks succeed.
    Static-only passes never claim trainability.
    """

    path: str
    passed: bool
    trainable: bool = False
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    network_type: str | None = None
    vision: dict[str, Any] = field(default_factory=dict)


def validate_env(
    path: Path | str,
    *,
    runtime: bool = False,
    vision: bool = False,
) -> EnvValidationResult:
    """Validate a custom env module.

    Static mode checks scaffold structure. Runtime mode also imports the module,
    constructs the env, and runs a short reset/step smoke when JAX is available.
    """
    env_path = Path(path).expanduser()
    missing: list[str] = []
    warnings: list[str] = []
    notes = [
        "Static checklist checks structure only.",
        "Use --runtime before proposing simrig smoke/train on a custom module.",
    ]

    if not env_path.is_file():
        return EnvValidationResult(
            path=str(env_path),
            passed=False,
            missing=[f"file not found: {env_path}"],
            notes=notes,
        )

    source = env_path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    declared_metadata: dict[str, Any] = {}
    network_type: str | None = None
    vision_details: dict[str, Any] = {}
    if vision:
        try:
            declared_metadata = load_custom_env_static_metadata(env_path)
            network_type = str(declared_metadata["network_spec"]["type"])
            vision_details["declared"] = dict(declared_metadata["vision_spec"])
        except Exception as exc:
            missing.append(f"static metadata load failed: {exc}")
    if runtime:
        try:
            metadata = load_custom_env_metadata(env_path)
            runtime_network_type = str(metadata["network_spec"]["type"])
            if vision and declared_metadata:
                missing.extend(_metadata_contract_mismatches(declared_metadata, metadata))
            else:
                network_type = runtime_network_type
        except Exception as exc:
            missing.append(f"metadata load failed: {exc}")
    elif vision:
        metadata = declared_metadata
    if vision and network_type != VISION_CNN_NETWORK:
        missing.append(
            "vision: network_spec must declare type `vision_cnn` when --vision is used"
        )
    if vision and metadata:
        declared = metadata.get("vision_spec", {})
        required_impl = declared.get("requires_impl") if isinstance(declared, Mapping) else None
        configured_impl = metadata.get("default_config", {}).get("impl")
        if not isinstance(declared, Mapping) or not declared.get("pixel_keys"):
            missing.append("vision: vision_spec must declare at least one pixel key")
        if not isinstance(declared, Mapping) or not declared.get("camera_names"):
            missing.append("vision: vision_spec must declare at least one camera name")
        if required_impl and configured_impl != required_impl:
            missing.append(
                f"vision: default_config impl={configured_impl!r} does not satisfy "
                f"requires_impl={required_impl!r}"
            )
        if required_impl == "warp" and not _jax_gpu_available():
            notes.append(
                "Vision runtime requires a JAX-visible CUDA GPU and MuJoCo Warp; "
                "this host can perform metadata checks only."
            )
    if vision and not runtime:
        notes.append("Vision metadata checked; use --runtime --vision to inspect frames.")
    for marker in REQUIRED_SECTION_MARKERS:
        if marker not in source:
            missing.append(f"section marker missing: {marker}")

    try:
        tree = ast.parse(source, filename=str(env_path))
    except SyntaxError as exc:
        return EnvValidationResult(
            path=str(env_path),
            passed=False,
            missing=[f"syntax error: {exc}"],
            notes=notes,
        )

    if "MODEL_PATH" not in source:
        missing.append("symbol missing: MODEL_PATH")
    if "ENV_NAME" not in source:
        missing.append("symbol missing: ENV_NAME")
    if "def default_config" not in source and "def make_env" not in source:
        missing.append("function missing: default_config or make_env")

    custom_env = _find_class(tree, "CustomEnv")
    has_make_env = any(
        isinstance(node, ast.FunctionDef) and node.name == "make_env" for node in tree.body
    )
    if custom_env is None and not has_make_env:
        missing.append("class missing: CustomEnv (or define make_env)")
    elif custom_env is not None:
        method_names = {
            node.name
            for node in custom_env.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in REQUIRED_CLASS_METHODS:
            if name not in method_names:
                missing.append(f"CustomEnv method missing: {name}")
        for prop in ("observation_size", "action_size"):
            if prop not in method_names:
                missing.append(f"CustomEnv property missing: {prop}")

    if "NotImplementedError" in source:
        warnings.append("NotImplementedError still present; implementation is incomplete.")
    if "NOT TRAINABLE YET" in source:
        warnings.append("File still marked NOT TRAINABLE YET.")

    passed = not missing
    trainable = False

    if runtime and passed:
        (
            runtime_missing,
            runtime_warnings,
            runtime_notes,
            trainable,
            runtime_vision,
        ) = _runtime_checks(env_path, require_vision=vision, metadata=metadata)
        missing.extend(runtime_missing)
        warnings.extend(runtime_warnings)
        notes.extend(runtime_notes)
        vision_details.update(runtime_vision)
        passed = not missing
    elif passed:
        notes.append(
            "Checklist structure looks complete. Fill SECTION bodies, then "
            "`simrig validate-env PATH --runtime` and `simrig smoke PATH`."
        )

    return EnvValidationResult(
        path=str(env_path),
        passed=passed,
        trainable=trainable,
        missing=missing,
        warnings=warnings,
        notes=notes,
        network_type=network_type,
        vision=vision_details,
    )


def _metadata_contract_mismatches(
    declared: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[str]:
    """Report disagreement between literal declarations and runtime hooks."""
    mismatches: list[str] = []
    declared_network = declared.get("network_spec", {})
    runtime_network = runtime.get("network_spec", {})
    if declared_network.get("type") != runtime_network.get("type"):
        mismatches.append(
            "vision: NETWORK_SPEC type does not match runtime network_spec()"
        )
    for section_name, constant_name in (
        ("default_config", "DEFAULT_CONFIG"),
        ("vision_spec", "VISION_SPEC"),
    ):
        declared_section = declared.get(section_name, {})
        runtime_section = runtime.get(section_name, {})
        for key, value in declared_section.items():
            if runtime_section.get(key) != value:
                mismatches.append(
                    f"vision: {constant_name}[{key!r}] does not match its runtime hook"
                )
    return mismatches


def _runtime_checks(
    env_path: Path,
    *,
    require_vision: bool,
    metadata: dict[str, Any],
) -> tuple[list[str], list[str], list[str], bool, dict[str, Any]]:
    missing: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    trainable = False
    vision_details: dict[str, Any] = {}

    declared = metadata.get("vision_spec", {})
    if (
        require_vision
        and isinstance(declared, Mapping)
        and declared.get("requires_impl") == "warp"
        and not _jax_gpu_available()
    ):
        missing.append(
            "vision runtime unavailable: a JAX-visible CUDA GPU and MuJoCo Warp "
            "are required"
        )
        return missing, warnings, notes, False, vision_details

    try:
        from simrig.custom_env import load_custom_env

        env = load_custom_env(env_path)
    except Exception as exc:
        missing.append(f"runtime load failed: {exc}")
        return missing, warnings, notes, False, vision_details

    notes.append("Custom env module constructed successfully.")
    action_size = getattr(env, "action_size", None)
    observation_size = getattr(env, "observation_size", None)

    if action_size is None:
        missing.append("runtime: action_size missing")
    obs_warnings = _obs_size_warnings(observation_size)
    warnings.extend(obs_warnings)

    for attr in ("mj_model", "mjx_model"):
        if getattr(env, attr, None) is None:
            warnings.append(f"runtime: {attr} missing (needed for Playground/Brax demos).")

    try:
        import jax  # type: ignore
        import jax.numpy as jp  # type: ignore
    except ImportError:
        warnings.append("runtime: JAX not installed; skipped reset/step smoke.")
        notes.append("Install the playground extra to run reset/step validation.")
        return missing, warnings, notes, False, vision_details

    try:
        state = env.reset(jax.random.PRNGKey(0))
        obs = getattr(state, "obs", None)
        if isinstance(obs, dict):
            if "state" not in obs:
                missing.append("runtime: state.obs missing key `state`")
            if "privileged_state" not in obs:
                warnings.append("runtime: state.obs missing key `privileged_state`")
        elif obs is None:
            missing.append("runtime: state.obs missing")
        else:
            warnings.append(
                "runtime: state.obs is flat; SimRig PPO defaults expect dict keys "
                "`state` and `privileged_state`."
            )

        if action_size is None:
            return missing, warnings, notes, False, vision_details
        next_state = env.step(state, jp.zeros(int(action_size)))
        vision_missing, vision_warnings, vision_details = _vision_runtime_checks(
            env,
            obs,
            getattr(next_state, "obs", None),
            observation_size,
            require_vision=require_vision,
            metadata=metadata,
        )
        missing.extend(vision_missing)
        warnings.extend(vision_warnings)
        state = next_state
        _ = float(state.reward), bool(state.done)
    except Exception as exc:
        missing.append(f"runtime reset/step failed: {exc}")
        return missing, warnings, notes, False, vision_details

    if missing:
        return missing, warnings, notes, False, vision_details

    trainable = True
    notes.append("Runtime reset/step succeeded. Safe to try `simrig smoke` then `simrig train`.")
    return missing, warnings, notes, trainable, vision_details


def _vision_runtime_checks(
    env: Any,
    obs: Any,
    next_obs: Any,
    observation_size: Any,
    *,
    require_vision: bool,
    metadata: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Inspect rendered observations without assuming a task-specific camera."""
    missing: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    if not isinstance(obs, dict):
        if require_vision:
            missing.append("vision: observations must be a dictionary")
        return missing, warnings, details

    pixel_keys = sorted(key for key in obs if str(key).startswith("pixels/"))
    details["pixel_keys"] = pixel_keys
    if require_vision and not pixel_keys:
        missing.append("vision: no observation keys with prefix `pixels/` were found")
        return missing, warnings, details
    if not pixel_keys:
        return missing, warnings, details

    try:
        import numpy as np
    except ImportError:
        warnings.append("vision: NumPy unavailable; skipped pixel value checks")
        return missing, warnings, details

    declared = metadata.get("vision_spec", {})
    declared_keys = declared.get("pixel_keys", []) if isinstance(declared, Mapping) else []
    for key in declared_keys:
        if key not in pixel_keys:
            missing.append(f"vision: declared pixel key missing from observations: {key}")

    frames: dict[str, Any] = {}
    for key in pixel_keys:
        frame = np.asarray(obs[key])
        next_frame = np.asarray(next_obs[key]) if isinstance(next_obs, dict) else None
        shape = tuple(int(x) for x in frame.shape)
        logical_shape = shape[1:] if len(shape) == 4 and shape[0] == 1 else shape
        entry = {
            "shape": list(shape),
            "logical_shape": list(logical_shape),
            "dtype": str(frame.dtype),
            "finite": bool(np.isfinite(frame).all()),
            "minimum": float(np.min(frame)) if frame.size else None,
            "maximum": float(np.max(frame)) if frame.size else None,
            "changed_after_step": (
                bool(not np.array_equal(frame, next_frame))
                if next_frame is not None
                else None
            ),
        }
        frames[key] = entry
        if len(logical_shape) != 3:
            missing.append(f"vision: {key} must have HWC shape, got {shape}")
        if frame.dtype.kind not in "fui":
            missing.append(f"vision: {key} must use a numeric dtype, got {frame.dtype}")
        if not entry["finite"]:
            missing.append(f"vision: {key} contains non-finite values")
        if entry["changed_after_step"] is False:
            warnings.append(f"vision: {key} did not change after one zero-action step")
        if isinstance(observation_size, dict) and key in observation_size:
            expected = observation_size[key]
            if (
                isinstance(expected, (list, tuple))
                and tuple(expected) not in (shape, logical_shape)
            ):
                missing.append(
                    f"vision: observation_size[{key!r}]={tuple(expected)} "
                    f"does not match runtime shape {shape}"
                )
        resolution = declared.get("resolution") if isinstance(declared, Mapping) else None
        if (
            resolution
            and len(logical_shape) >= 2
            and tuple(resolution) != logical_shape[:2]
        ):
            missing.append(
                f"vision: declared resolution {tuple(resolution)} does not match "
                f"{key} shape {logical_shape[:2]}"
            )
        frame_stack = declared.get("frame_stack") if isinstance(declared, Mapping) else None
        channels_per_frame = _channels_per_frame(declared)
        expected_channels = (
            int(frame_stack) * channels_per_frame
            if frame_stack and channels_per_frame is not None
            else None
        )
        if (
            expected_channels is not None
            and len(logical_shape) == 3
            and expected_channels != logical_shape[-1]
        ):
            missing.append(
                f"vision: declared frame_stack={frame_stack} and "
                f"channels_per_frame={channels_per_frame} imply {expected_channels} "
                f"channels, but {key} has {logical_shape[-1]}"
            )
        value_range = declared.get("value_range") if isinstance(declared, Mapping) else None
        if value_range and len(value_range) == 2 and frame.size:
            low, high = float(value_range[0]), float(value_range[1])
            if entry["minimum"] < low - 1e-6 or entry["maximum"] > high + 1e-6:
                missing.append(
                    f"vision: {key} range [{entry['minimum']}, {entry['maximum']}] "
                    f"exceeds declared [{low}, {high}]"
                )
    details["frames"] = frames

    network = metadata.get("network_spec", {})
    factory = network.get("factory", {}) if isinstance(network, dict) else {}
    policy_key = factory.get("policy_obs_key", "")
    value_key = factory.get("value_obs_key", "")
    details["policy_obs_key"] = policy_key
    details["value_obs_key"] = value_key
    if policy_key and policy_key not in obs:
        missing.append(f"vision: policy_obs_key is absent from observations: {policy_key}")
    if value_key and value_key not in obs:
        missing.append(f"vision: value_obs_key is absent from observations: {value_key}")
    if policy_key == "privileged_state":
        missing.append("vision: actor policy_obs_key cannot be `privileged_state`")

    camera_names = declared.get("camera_names", []) if isinstance(declared, Mapping) else []
    if camera_names:
        model = getattr(env, "mj_model", None)
        if model is None:
            missing.append("vision: cannot validate cameras because mj_model is missing")
            return missing, warnings, details
        for name in camera_names:
            try:
                model.camera(name)
            except Exception:
                missing.append(f"vision: declared MuJoCo camera not found: {name}")
    return missing, warnings, details


def _obs_size_warnings(observation_size: Any) -> list[str]:
    warnings: list[str] = []
    if observation_size is None:
        warnings.append("runtime: observation_size missing")
        return warnings
    if isinstance(observation_size, dict):
        if "state" not in observation_size:
            warnings.append("runtime: observation_size missing key `state`")
        if "privileged_state" not in observation_size:
            warnings.append("runtime: observation_size missing key `privileged_state`")
    else:
        warnings.append(
            "runtime: observation_size is flat; SimRig PPO defaults expect "
            "`state` / `privileged_state`."
        )
    return warnings


def _jax_gpu_available() -> bool:
    try:
        import jax  # type: ignore

        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def _channels_per_frame(declared: Mapping[str, Any]) -> int | None:
    explicit = declared.get("channels_per_frame")
    if explicit is not None:
        return int(explicit)
    modalities = [str(value).lower() for value in declared.get("modalities", [])]
    if modalities == ["rgb"]:
        return 3
    if modalities in (["grayscale"], ["depth"]):
        return 1
    return None


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None
