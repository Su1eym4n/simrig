"""Command-line interface for SimRig."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from simrig._version import __version__
from simrig.core import report_markdown, to_dict
from simrig.huggingface import resolve_policy_checkpoint
from simrig.io import save_json, save_report_pair, slugify
from simrig.lambda_cloud import (
    LambdaSSHConfig,
    check_lambda,
    connect_lambda,
    fetch_lambda,
    prepare_lambda,
    smoke_lambda,
    status_lambda,
    train_lambda,
)
from simrig.mujoco_backend import inspect_model, list_models
from simrig.paths import ensure_project_dirs
from simrig.playground_backend import (
    demo_policy,
    eval_policy,
    inspect_env,
    list_envs,
    smoke_env,
    train_ppo,
)
from simrig.scaffold import new_env
from simrig.validate_env import validate_env


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except Exception as exc:
        print(f"simrig: error: {exc}", file=sys.stderr)
        return 1
    return 0 if result is None else int(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simrig",
        description="Physical AI simulation training starter workflows.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create local SimRig output folders.")
    init.add_argument("--root", type=Path, default=Path("."))
    init.set_defaults(func=_cmd_init)

    list_models_parser = sub.add_parser("list-models", help="List MuJoCo Menagerie models.")
    list_models_parser.add_argument("--menagerie", type=Path)
    list_models_parser.add_argument("--json", action="store_true")
    list_models_parser.set_defaults(func=_cmd_list_models)

    inspect_model_parser = sub.add_parser("inspect-model", help="Inspect a MuJoCo XML/model.")
    inspect_model_parser.add_argument("model_or_xml")
    inspect_model_parser.add_argument("--menagerie", type=Path)
    inspect_model_parser.add_argument("--steps", type=int, default=25)
    inspect_model_parser.add_argument("--json", action="store_true")
    inspect_model_parser.add_argument("--save-report", action="store_true")
    inspect_model_parser.set_defaults(func=_cmd_inspect_model)

    list_envs_parser = sub.add_parser("list-envs", help="List trainable backend envs.")
    list_envs_parser.add_argument("--backend", default="mujoco-playground")
    list_envs_parser.add_argument("--json", action="store_true")
    list_envs_parser.set_defaults(func=_cmd_list_envs)

    inspect_env_parser = sub.add_parser("inspect-env", help="Inspect a Playground or custom env.")
    inspect_env_parser.add_argument(
        "env_name",
        help="Playground env name or path to a custom *.py env module.",
    )
    inspect_env_parser.add_argument("--backend", default="mujoco-playground")
    inspect_env_parser.add_argument("--json", action="store_true")
    inspect_env_parser.add_argument("--save-report", action="store_true")
    inspect_env_parser.set_defaults(func=_cmd_inspect_env)

    smoke_parser = sub.add_parser("smoke", help="Run a short env reset/step smoke test.")
    smoke_parser.add_argument(
        "env_name",
        help="Playground env name or path to a custom *.py env module.",
    )
    smoke_parser.add_argument("--backend", default="mujoco-playground")
    smoke_parser.add_argument("--steps", type=int, default=10)
    smoke_parser.add_argument("--json", action="store_true")
    smoke_parser.set_defaults(func=_cmd_smoke)

    train_parser = sub.add_parser(
        "train",
        help="Train a Playground env or custom *.py module with Brax PPO.",
    )
    train_parser.add_argument(
        "env_name",
        help="Playground env name or path to a custom *.py env module.",
    )
    train_parser.add_argument("--backend", default="mujoco-playground")
    train_parser.add_argument("--preset", choices=("smoke", "local", "cloud"), default="smoke")
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--timesteps", type=int)
    train_parser.add_argument("--num-envs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument(
        "--impl",
        choices=("auto", "jax", "warp"),
        default="auto",
        help="MuJoCo implementation; auto follows the environment default.",
    )
    train_parser.add_argument("--seed", type=int, default=0, help="PPO training seed.")
    train_parser.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the environment's declared domain randomizer when available.",
    )
    train_parser.set_defaults(func=_cmd_train)

    eval_parser = sub.add_parser("eval", help="Headless policy eval.")
    eval_parser.add_argument("checkpoint")
    eval_parser.add_argument(
        "--env",
        dest="env_name",
        required=True,
        help="Playground env name or path to a custom *.py env module.",
    )
    eval_parser.add_argument("--backend", default="mujoco-playground")
    eval_parser.add_argument("--steps", type=int, default=500)
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic environment and policy rollout seed.",
    )
    eval_parser.add_argument(
        "--command",
        type=float,
        nargs="+",
        help="Fix command-like environment state, for example X Y YAW.",
    )
    eval_parser.add_argument("--small-network", action=argparse.BooleanOptionalAction, default=None)
    eval_parser.add_argument("--hf-revision", help="Revision for hf:// policy checkpoints.")
    eval_parser.add_argument("--hf-token", help="Hugging Face token for private policy repos.")
    eval_parser.add_argument(
        "--allow-runtime-mismatch",
        action="store_true",
        help="Allow an explicitly qualitative rollout when recorded runtime versions differ.",
    )
    eval_parser.add_argument("--json", action="store_true")
    eval_parser.set_defaults(func=_cmd_eval)

    demo_parser = sub.add_parser("demo", help="Run a trained policy in a desktop MuJoCo viewer.")
    demo_parser.add_argument("checkpoint")
    demo_parser.add_argument("--env", dest="env_name", required=True)
    demo_parser.add_argument("--backend", default="mujoco-playground")
    demo_parser.add_argument("--steps", type=int, default=5000)
    demo_parser.add_argument("--small-network", action=argparse.BooleanOptionalAction, default=None)
    demo_parser.add_argument("--hf-revision", help="Revision for hf:// policy checkpoints.")
    demo_parser.add_argument("--hf-token", help="Hugging Face token for private policy repos.")
    demo_parser.add_argument("--command", type=float, nargs="+")
    demo_parser.add_argument("--speed", type=float, default=1.0)
    demo_parser.add_argument("--camera-distance", type=float)
    demo_parser.add_argument(
        "--allow-runtime-mismatch",
        action="store_true",
        help="Allow an explicitly qualitative demo when recorded runtime versions differ.",
    )
    demo_parser.add_argument("--json", action="store_true")
    demo_parser.set_defaults(func=_cmd_demo)

    preview_parser = sub.add_parser("preview", help="Serve a trained policy preview in the browser.")
    preview_parser.add_argument("checkpoint")
    preview_parser.add_argument("--env", dest="env_name", required=True)
    preview_parser.add_argument("--backend", default="mujoco-playground")
    preview_parser.add_argument("--host", default="127.0.0.1")
    preview_parser.add_argument("--port", type=int, default=8765)
    preview_parser.add_argument("--width", type=int, default=960)
    preview_parser.add_argument("--height", type=int, default=540)
    preview_parser.add_argument("--frame-skip", type=int, default=1)
    preview_parser.add_argument("--fps", type=int, default=24, help="Browser render loop target FPS.")
    preview_parser.add_argument("--small-network", action=argparse.BooleanOptionalAction, default=None)
    preview_parser.add_argument("--hf-revision", help="Revision for hf:// policy checkpoints.")
    preview_parser.add_argument("--hf-token", help="Hugging Face token for private policy repos.")
    preview_parser.add_argument("--command", type=float, nargs="+")
    preview_parser.add_argument(
        "--camera",
        help=(
            "Named/numbered MuJoCo camera. In Three.js mode this selects the "
            "initial Agent Camera inset."
        ),
    )
    preview_parser.add_argument("--paused", action="store_true", help="Start the browser preview paused.")
    preview_parser.add_argument(
        "--auto-reset",
        action="store_true",
        help="Automatically start a new episode after termination.",
    )
    preview_parser.add_argument(
        "--auto-reset-delay",
        type=float,
        default=1.5,
        help="Seconds to preserve the terminal pose before automatic reset.",
    )
    preview_parser.add_argument(
        "--allow-runtime-mismatch",
        action="store_true",
        help="Allow an explicitly qualitative preview when recorded runtime versions differ.",
    )
    preview_parser.add_argument(
        "--render-mode",
        choices=("threejs", "mujoco", "topdown"),
        default="threejs",
        help=(
            "Browser render mode. threejs renders rollout geometry interactively in WebGL; "
            "mujoco streams offscreen frames; topdown is a schematic debug view."
        ),
    )
    preview_parser.set_defaults(func=_cmd_preview)

    view_model_parser = sub.add_parser(
        "view-model",
        help="Serve a MuJoCo model in the browser with per-joint controls.",
    )
    view_model_parser.add_argument("model_or_xml")
    view_model_parser.add_argument("--menagerie", type=Path)
    view_model_parser.add_argument("--host", default="127.0.0.1")
    view_model_parser.add_argument("--port", type=int, default=8766)
    view_model_parser.add_argument("--width", type=int, default=960)
    view_model_parser.add_argument("--height", type=int, default=540)
    view_model_parser.add_argument("--fps", type=int, default=24, help="Browser render loop target FPS.")
    view_model_parser.add_argument(
        "--camera",
        help=(
            "Named/numbered MuJoCo camera. In Three.js mode this selects the "
            "initial Agent Camera inset."
        ),
    )
    view_model_parser.add_argument(
        "--render-mode",
        choices=("threejs", "mujoco", "topdown"),
        default="threejs",
        help=(
            "Browser render mode. threejs renders geometry interactively in WebGL; "
            "mujoco streams offscreen frames; topdown is a schematic debug view."
        ),
    )
    view_model_parser.set_defaults(func=_cmd_view_model)

    new_env_parser = sub.add_parser("new-env", help="Create an editable custom env starter.")
    new_env_parser.add_argument("name")
    new_env_parser.add_argument("--model", required=True)
    new_env_parser.add_argument("--template", default="mjx")
    new_env_parser.add_argument("--root", type=Path, default=Path("envs"))
    new_env_parser.set_defaults(func=_cmd_new_env)

    validate_env_parser = sub.add_parser(
        "validate-env",
        help="Validate a custom env module (static checklist; optional runtime).",
    )
    validate_env_parser.add_argument("path", type=Path)
    validate_env_parser.add_argument(
        "--runtime",
        action="store_true",
        help="Import the module and run construct/reset/step checks when possible.",
    )
    validate_env_parser.add_argument(
        "--vision",
        action="store_true",
        help="Require and validate vision-network metadata and pixel observations.",
    )
    validate_env_parser.add_argument("--json", action="store_true")
    validate_env_parser.set_defaults(func=_cmd_validate_env)

    cloud_parser = sub.add_parser(
        "cloud",
        help="Connect to and train on an already-provisioned cloud GPU.",
    )
    cloud_providers = cloud_parser.add_subparsers(dest="cloud_provider", required=True)
    lambda_parser = cloud_providers.add_parser(
        "lambda",
        help="Use a Lambda On-Demand Cloud instance over SSH.",
    )
    lambda_actions = lambda_parser.add_subparsers(dest="cloud_action", required=True)

    lambda_connect = lambda_actions.add_parser(
        "connect",
        help="Open an interactive SSH connection to the instance.",
    )
    _add_lambda_connection_args(lambda_connect)
    lambda_connect.add_argument(
        "--tunnel-port",
        type=int,
        help="Forward this localhost port to the same port on the instance.",
    )
    lambda_connect.set_defaults(func=_cmd_lambda_connect)

    lambda_check = lambda_actions.add_parser(
        "check",
        help="Check SSH plus NVIDIA and JAX device visibility.",
    )
    _add_lambda_connection_args(lambda_check)
    lambda_check.set_defaults(func=_cmd_lambda_check)

    lambda_prepare = lambda_actions.add_parser(
        "prepare",
        help="Sync this checkout and prepare a GPU-enabled remote virtualenv.",
    )
    _add_lambda_connection_args(lambda_prepare)
    lambda_prepare.add_argument("--project", type=Path, default=Path("."))
    lambda_prepare.add_argument("--remote-dir")
    lambda_prepare.add_argument(
        "--jax-cuda",
        choices=("preinstalled", "cuda12", "cuda13"),
        default="preinstalled",
        help="Use Lambda's preinstalled JAX or install a pip CUDA wheel.",
    )
    lambda_prepare.add_argument(
        "--python",
        dest="python_command",
        default="python3",
        help="Remote Python 3.11+ executable used to create the training virtualenv.",
    )
    lambda_prepare.set_defaults(func=_cmd_lambda_prepare)

    lambda_smoke = lambda_actions.add_parser(
        "smoke",
        help="Run the reset/step environment smoke gate on the remote GPU.",
    )
    _add_lambda_connection_args(lambda_smoke)
    lambda_smoke.add_argument(
        "env_name",
        help="Playground env name or synced custom *.py env module path.",
    )
    lambda_smoke.add_argument("--remote-dir")
    lambda_smoke.add_argument("--steps", type=int, default=10)
    lambda_smoke.set_defaults(func=_cmd_lambda_smoke)

    lambda_train = lambda_actions.add_parser(
        "train",
        help="Run training on a prepared Lambda instance (smoke preset by default).",
    )
    _add_lambda_connection_args(lambda_train)
    lambda_train.add_argument(
        "env_name",
        help="Playground env name or synced custom *.py env module path.",
    )
    lambda_train.add_argument(
        "--preset",
        choices=("smoke", "local", "cloud"),
        default="smoke",
    )
    lambda_train.add_argument("--remote-dir")
    lambda_train.add_argument(
        "--output",
        help="Remote run directory, relative to --remote-dir unless absolute.",
    )
    lambda_train.add_argument(
        "--detach",
        action="store_true",
        help="Keep training after SSH disconnects and write train.log/train.pid.",
    )
    lambda_train.add_argument("--timesteps", type=int)
    lambda_train.add_argument("--num-envs", type=int)
    lambda_train.add_argument("--batch-size", type=int)
    lambda_train.add_argument(
        "--impl",
        choices=("auto", "jax", "warp"),
        default="auto",
    )
    lambda_train.add_argument("--seed", type=int, default=0)
    lambda_train.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    lambda_train.set_defaults(func=_cmd_lambda_train)

    lambda_status = lambda_actions.add_parser(
        "status",
        help="Show detached process state and recent training log lines.",
    )
    _add_lambda_connection_args(lambda_status)
    lambda_status.add_argument("output", help="Remote output returned by cloud lambda train.")
    lambda_status.add_argument("--remote-dir")
    lambda_status.add_argument("--lines", type=int, default=30)
    lambda_status.set_defaults(func=_cmd_lambda_status)

    lambda_fetch = lambda_actions.add_parser(
        "fetch",
        help="Download a remote run directory into local runs/.",
    )
    _add_lambda_connection_args(lambda_fetch)
    lambda_fetch.add_argument("output", help="Remote output returned by cloud lambda train.")
    lambda_fetch.add_argument("--remote-dir")
    lambda_fetch.add_argument("--local-output", type=Path)
    lambda_fetch.set_defaults(func=_cmd_lambda_fetch)

    return parser


def _cmd_init(args: argparse.Namespace) -> None:
    paths = ensure_project_dirs(args.root)
    for path in paths:
        print(path)


def _cmd_list_models(args: argparse.Namespace) -> None:
    entries = list_models(args.menagerie)
    _print(entries, as_json=args.json)


def _cmd_inspect_model(args: argparse.Namespace) -> None:
    report = inspect_model(args.model_or_xml, menagerie=args.menagerie, steps=args.steps)
    if args.save_report:
        md_path, json_path = save_report_pair(
            report,
            name=report.name,
            title=f"Model Inspection: {report.name}",
        )
        print(f"saved {md_path}")
        print(f"saved {json_path}")
    _print_report(f"Model Inspection: {report.name}", report, as_json=args.json)


def _cmd_list_envs(args: argparse.Namespace) -> None:
    _print(list_envs(args.backend), as_json=args.json)


def _cmd_inspect_env(args: argparse.Namespace) -> None:
    report = inspect_env(args.env_name, backend=args.backend)
    if args.save_report:
        md_path, json_path = save_report_pair(
            report,
            name=report.name,
            title=f"Env Inspection: {report.name}",
        )
        print(f"saved {md_path}")
        print(f"saved {json_path}")
    _print_report(f"Env Inspection: {report.name}", report, as_json=args.json)


def _cmd_smoke(args: argparse.Namespace) -> int:
    result = smoke_env(args.env_name, backend=args.backend, steps=args.steps)
    _print(result, as_json=args.json)
    return 0 if result.passed else 1


def _cmd_train(args: argparse.Namespace) -> None:
    overrides = _training_overrides(args)
    run_config = train_ppo(
        args.env_name,
        backend=args.backend,
        preset_name=args.preset,
        output=args.output,
        overrides=overrides,
        impl=args.impl,
        seed=args.seed,
        domain_randomization=args.domain_randomization,
    )
    print(f"saved run: {run_config.output_dir}")


def _cmd_eval(args: argparse.Namespace) -> None:
    command = tuple(args.command) if args.command is not None else None
    checkpoint = resolve_policy_checkpoint(
        args.checkpoint,
        hf_revision=args.hf_revision,
        hf_token=args.hf_token,
    )
    result = eval_policy(
        checkpoint,
        env_name=args.env_name,
        backend=args.backend,
        steps=args.steps,
        small_network=args.small_network,
        seed=args.seed,
        command=command,
        allow_runtime_mismatch=args.allow_runtime_mismatch,
    )
    save_json(Path("reports") / f"{slugify(args.env_name)}_eval.json", result)
    _print(result, as_json=args.json)


def _cmd_demo(args: argparse.Namespace) -> None:
    command = tuple(args.command) if args.command is not None else None
    checkpoint = resolve_policy_checkpoint(
        args.checkpoint,
        hf_revision=args.hf_revision,
        hf_token=args.hf_token,
    )
    result = demo_policy(
        checkpoint,
        env_name=args.env_name,
        backend=args.backend,
        steps=args.steps,
        small_network=args.small_network,
        command=command,
        speed=args.speed,
        camera_distance=args.camera_distance,
        allow_runtime_mismatch=args.allow_runtime_mismatch,
    )
    _print(result, as_json=args.json)


def _cmd_view_model(args: argparse.Namespace) -> None:
    from simrig.model_view import serve_model_view

    serve_model_view(
        args.model_or_xml,
        menagerie=args.menagerie,
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        render_mode=args.render_mode,
        camera=args.camera,
        fps=args.fps,
    )


def _cmd_preview(args: argparse.Namespace) -> None:
    from simrig.preview import serve_policy_preview

    command = tuple(args.command) if args.command is not None else None
    checkpoint = resolve_policy_checkpoint(
        args.checkpoint,
        hf_revision=args.hf_revision,
        hf_token=args.hf_token,
    )
    serve_policy_preview(
        checkpoint,
        env_name=args.env_name,
        backend=args.backend,
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        frame_skip=args.frame_skip,
        small_network=args.small_network,
        command=command,
        camera=args.camera,
        render_mode=args.render_mode,
        paused=args.paused,
        fps=args.fps,
        auto_reset=args.auto_reset,
        auto_reset_delay=args.auto_reset_delay,
        allow_runtime_mismatch=args.allow_runtime_mismatch,
    )


def _cmd_new_env(args: argparse.Namespace) -> None:
    path = new_env(args.name, args.model, template=args.template, root=args.root)
    print(path)


def _cmd_validate_env(args: argparse.Namespace) -> int:
    result = validate_env(
        args.path,
        runtime=bool(args.runtime),
        vision=bool(args.vision),
    )
    _print(result, as_json=args.json)
    if not args.json:
        status = "passed" if result.passed else "failed"
        print(f"validate-env: {status} (trainable={result.trainable})")
    return 0 if result.passed else 1


def _add_lambda_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("host", help="Public IP address or DNS name from Lambda Cloud.")
    parser.add_argument(
        "--identity",
        "-i",
        type=Path,
        help="Path to the SSH private key selected when the instance was launched.",
    )
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--port", type=int, default=22)


def _lambda_config(args: argparse.Namespace) -> LambdaSSHConfig:
    return LambdaSSHConfig(
        host=args.host,
        user=args.user,
        identity=args.identity,
        port=args.port,
    )


def _cmd_lambda_connect(args: argparse.Namespace) -> int:
    return connect_lambda(_lambda_config(args), tunnel_port=args.tunnel_port)


def _cmd_lambda_check(args: argparse.Namespace) -> int:
    return check_lambda(_lambda_config(args))


def _cmd_lambda_prepare(args: argparse.Namespace) -> None:
    prepare_lambda(
        _lambda_config(args),
        project_dir=args.project,
        remote_dir=args.remote_dir,
        jax_cuda=args.jax_cuda,
        python_command=args.python_command,
    )
    print(f"prepared Lambda project: {args.remote_dir or f'/home/{args.user}/simrig'}")


def _cmd_lambda_train(args: argparse.Namespace) -> int:
    result = train_lambda(
        _lambda_config(args),
        args.env_name,
        preset_name=args.preset,
        remote_dir=args.remote_dir,
        output=args.output,
        detach=args.detach,
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        impl=args.impl,
        seed=args.seed,
        domain_randomization=args.domain_randomization,
    )
    if result.returncode == 0:
        mode = "detached run" if result.detached else "run"
        print(f"Lambda {mode}: {result.output_dir}")
    return result.returncode


def _cmd_lambda_smoke(args: argparse.Namespace) -> int:
    return smoke_lambda(
        _lambda_config(args),
        args.env_name,
        remote_dir=args.remote_dir,
        steps=args.steps,
    )


def _cmd_lambda_status(args: argparse.Namespace) -> int:
    return status_lambda(
        _lambda_config(args),
        args.output,
        remote_dir=args.remote_dir,
        lines=args.lines,
    )


def _cmd_lambda_fetch(args: argparse.Namespace) -> None:
    destination = fetch_lambda(
        _lambda_config(args),
        args.output,
        remote_dir=args.remote_dir,
        local_output=args.local_output,
    )
    print(f"downloaded run: {destination}")
    print(
        "After verifying these artifacts or persistent storage, terminate the "
        "Lambda instance in the cloud console to stop compute charges."
    )


def _training_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for cli_name, key in (
        ("timesteps", "timesteps"),
        ("num_envs", "num_envs"),
        ("batch_size", "batch_size"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            overrides[key] = value
    return overrides


def _print_report(title: str, report: Any, *, as_json: bool) -> None:
    if as_json:
        _print(report, as_json=True)
    else:
        print(report_markdown(title, report), end="")


def _print(value: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(to_dict(value), indent=2, sort_keys=True))
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "name" in item:
                print(item["name"])
            else:
                print(item)
        return
    print(json.dumps(to_dict(value), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
