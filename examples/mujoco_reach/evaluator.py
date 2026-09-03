"""Evaluate a Python controller by stepping the included arm in native MuJoCo.

No JAX, learned weights, training reward, or controller-provided success labels
are used. The task definition and the two example controllers live beside this
module. A controller is trusted local Python implementing action(model, data,
target); it must not modify model/data. This example is not a plugin sandbox.
"""

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
from targets import seeded_targets


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "simple_arm.xml"
PREVIEW_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "orbit_arm.xml"
CONTROL_DT = 0.02
SUCCESS_DISTANCE = 0.05
PREVIEW_COMMAND_DELTA = 0.035
PREVIEW_HOLD_TICKS = 8
PREVIEW_FRAME_DELAY_SEC = 0.04
PREVIEW_TARGET_COUNT = 30
EVALUATOR_SPEC = {
    "name": "mujoco-simple-arm-reach",
    "version": "1.0.0",
    "protocol_version": 1,
    "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
}


def _load_controller(path):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("Supply a trusted Python controller, not a policy checkpoint")
    spec = importlib.util.spec_from_file_location("reach_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "action", None)):
        raise ValueError("Controller must define action(model, data, target)")
    return module.action


def evaluate(request):
    """SimRig's evaluator entry point for the original two-joint task."""
    return rollout(request)


def _preview_targets(seed, *, np):
    """Return 30 seeded 3D targets scattered around every side of the base."""
    del np
    return seeded_targets(seed, count=PREVIEW_TARGET_COUNT)


def _normalized_position_targets(qpos, low, high, *, np):
    positions = np.asarray(qpos, dtype=float)
    return 2 * (positions - low) / (high - low) - 1


def rollout(request, *, preview=False, preview_port=0):
    import mujoco
    import numpy as np

    if request["backend"] != "mujoco" or request["environment"] != "simple-arm-reach":
        raise ValueError("This evaluator supports only the included simple-arm task")
    max_steps = request["max_steps"]
    if type(max_steps) is not int or not 1 <= max_steps <= 200:
        raise ValueError("This example requires a horizon of 1..200 control steps")
    controller = _load_controller(request["checkpoint"])
    active_model_path = PREVIEW_MODEL_PATH if preview else MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(active_model_path))
    data = mujoco.MjData(model)
    target = np.asarray(request["parameters"]["target"], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Target must contain three finite world-frame coordinates")
    rng = random.Random(request["seed"])
    # Match demo_reach's reset range for evaluation. The showcase starts centered.
    initial_qpos = (
        np.zeros(model.nq)
        if preview
        else np.asarray([rng.uniform(-0.4, 0.4) for _ in range(model.nq)])
    )
    data.qpos[:] = initial_qpos
    target_id = model.body("target").mocapid[0]
    targets = _preview_targets(request["seed"], np=np) if preview else (target,)
    target_index = 0
    target = targets[target_index]
    data.mocap_pos[target_id] = target
    mujoco.mj_forward(model, data)
    ee_id = model.site("ee_site").id
    n_substeps = round(CONTROL_DT / model.opt.timestep)
    if not np.isclose(n_substeps * model.opt.timestep, CONTROL_DT):
        raise ValueError("Control interval must be an integer number of physics steps")
    low, high = model.actuator_ctrlrange.T
    initial_error = float(np.linalg.norm(data.site_xpos[ee_id] - target))
    errors, events = [], []
    preview_command = _normalized_position_targets(data.qpos[:model.nu], low, high, np=np)
    hold_ticks = 0
    viewer = None
    if preview:
        from simrig import LiveWebViewer

        viewer = LiveWebViewer(
            model,
            data,
            name="MuJoCo reaching",
            port=preview_port,
            tracking_site="ee_site",
            allow_reset=True,
        )
    with viewer if viewer is not None else nullcontext():
        if viewer is not None:
            viewer.wait_for_client(timeout=15)
        tick = 0
        while preview or tick < max_steps:
            if viewer is not None and viewer.consume_reset_request():
                with viewer.lock:
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = initial_qpos
                    target_index = 0
                    target = targets[target_index]
                    data.mocap_pos[target_id] = target
                    mujoco.mj_forward(model, data)
                    preview_command = _normalized_position_targets(
                        data.qpos[:model.nu], low, high, np=np
                    )
                    hold_ticks = 0
                    viewer.sync(
                        target_error_m=float(np.linalg.norm(data.site_xpos[ee_id] - target)),
                        reached=False,
                        target_index=1,
                        target_count=len(targets),
                        phase="reset",
                    )
            with viewer.lock if viewer is not None else nullcontext():
                goal_command = np.asarray(controller(model, data, target.copy()), dtype=float)
                if goal_command.shape != (model.nu,) or not np.isfinite(goal_command).all():
                    raise ValueError("Controller must return one finite action per actuator")
                if np.any(np.abs(goal_command) > 1):
                    raise ValueError("Normalized actions must lie in [-1, 1]")
                if preview:
                    command = np.clip(
                        goal_command,
                        preview_command - PREVIEW_COMMAND_DELTA,
                        preview_command + PREVIEW_COMMAND_DELTA,
                    )
                    preview_command = command
                else:
                    command = goal_command
                data.ctrl[:] = low + (command + 1) * 0.5 * (high - low)
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
                # Refresh derived positions at the final integrated state.
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("MuJoCo produced a non-finite state")
                error = float(np.linalg.norm(data.site_xpos[ee_id] - target))
                if not preview:
                    errors.append(error)
                reached = error < SUCCESS_DISTANCE
                if not preview:
                    events.append({
                        "kind": "signal",
                        "name": "target_reached",
                        "step": tick,
                        "active": reached,
                    })
                if viewer is not None:
                    viewer.sync(
                        target_error_m=error,
                        reached=reached,
                        target_index=target_index + 1,
                        target_count=len(targets),
                        phase="holding target" if reached else "moving to target",
                    )
            if viewer is not None:
                time.sleep(PREVIEW_FRAME_DELAY_SEC)
            tick += 1
            if reached:
                if not preview:
                    break
                hold_ticks += 1
                if hold_ticks >= PREVIEW_HOLD_TICKS:
                    target_index = (target_index + 1) % len(targets)
                    target = targets[target_index]
                    hold_ticks = 0
                    with viewer.lock:
                        data.mocap_pos[target_id] = target
                        mujoco.mj_forward(model, data)
                        viewer.sync(
                            target_error_m=float(np.linalg.norm(data.site_xpos[ee_id] - target)),
                            reached=False,
                            target_index=target_index + 1,
                            target_count=len(targets),
                            phase="restarting route" if target_index == 0 else "next target",
                        )
    return {
        "metrics": {
            "initial_target_error": initial_error,
            "final_target_error": errors[-1],
            "minimum_target_error": min(errors),
            "control_steps": len(errors),
            "physics_steps": len(errors) * n_substeps,
            "simulation_time_sec": float(data.time),
            "mujoco_warning_count": int(sum(data.warning.number)),
        },
        "events": events,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=Path(__file__).parent / "controllers" / "ik.py")
    parser.add_argument("--scenario", choices=("nominal", "boundary"), default="nominal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Browser port; 0 selects an available local port automatically.",
    )
    args = parser.parse_args()
    task = json.loads((Path(__file__).parent / "task.json").read_text())
    scenarios = task["evaluation"]["suites"]["promotion"]["scenarios"]
    scenario = next(item for item in scenarios if item["name"] == args.scenario)
    request = {
        "checkpoint": str(args.controller), "backend": task["environment"]["backend"],
        "environment": task["environment"]["ref"], "parameters": scenario["parameters"],
        "max_steps": task["episode"]["horizon_steps"], "seed": args.seed,
    }
    try:
        result = rollout(request, preview=args.preview, preview_port=args.port)
    except KeyboardInterrupt:
        if args.preview:
            print("\nPreview stopped.")
            return
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
