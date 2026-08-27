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
from simrig.evaluation_suite import (
    EvaluationLimits,
    evaluate_checkpoint_directory,
    run_evaluation_suite,
)
from simrig.gates import (
    adversarial_reward_probes,
    audit_reward_alignment,
    evaluate_gate,
    load_evaluation_records,
)
from simrig.io import save_json, save_report_pair, slugify
from simrig.presets import canonical_preset
from simrig.progress import describe_run
from simrig.ranking import load_suite_reports, rank_checkpoints
from simrig.remote import (
    SSHConfig,
    check_remote,
    connect_remote,
    fetch_remote,
    prepare_remote,
    smoke_remote,
    status_remote,
    train_remote,
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
from simrig.task_contract import (
    COMPATIBILITY_POLICIES,
    SCHEMA_VERSION,
    compare_task_contracts,
    diff_task_contracts,
    load_task_contract,
    save_migrated_task_contract,
    save_frozen_task_contract,
    task_contract_template,
    validate_task_contract,
)
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

    task_parser = sub.add_parser(
        "task",
        help="Create, validate, freeze, and compare task contracts.",
    )
    task_actions = task_parser.add_subparsers(dest="task_action", required=True)
    task_init = task_actions.add_parser("init", help="Create an editable task contract.")
    task_init.add_argument("env_name", help="Environment name or custom *.py module.")
    task_init.add_argument("--name", help="Stable task name; defaults to the environment label.")
    task_init.add_argument("--output", type=Path, default=Path("task.json"))
    task_init.set_defaults(func=_cmd_task_init)

    task_validate = task_actions.add_parser(
        "validate", help="Validate a draft or frozen task contract."
    )
    task_validate.add_argument("path", type=Path)
    task_validate.add_argument("--json", action="store_true")
    task_validate.set_defaults(func=_cmd_task_validate)

    task_freeze = task_actions.add_parser(
        "freeze", help="Validate and freeze a task contract with a content hash."
    )
    task_freeze.add_argument("path", type=Path)
    task_freeze.add_argument("--output", type=Path)
    task_freeze.set_defaults(func=_cmd_task_freeze)

    task_diff = task_actions.add_parser(
        "diff", help="Compare task-contract semantics independent of formatting."
    )
    task_diff.add_argument("left", type=Path)
    task_diff.add_argument("right", type=Path)
    task_diff.add_argument("--json", action="store_true")
    task_diff.set_defaults(func=_cmd_task_diff)

    task_migrate = task_actions.add_parser(
        "migrate", help="Explicitly migrate an older contract to the current draft schema."
    )
    task_migrate.add_argument("path", type=Path)
    task_migrate.add_argument("--output", type=Path, required=True)
    task_migrate.add_argument("--json", action="store_true")
    task_migrate.set_defaults(func=_cmd_task_migrate)

    task_compat = task_actions.add_parser(
        "compatibility", help="Compare contracts under an explicit compatibility policy."
    )
    task_compat.add_argument("left", type=Path)
    task_compat.add_argument("right", type=Path)
    task_compat.add_argument("--policy", choices=sorted(COMPATIBILITY_POLICIES), required=True)
    task_compat.add_argument("--json", action="store_true")
    task_compat.set_defaults(func=_cmd_task_compatibility)

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
    train_parser.add_argument(
        "--preset",
        type=_parse_preset,
        default="smoke",
        help="PPO scale: smoke, local, or large. `cloud` is a hidden alias for large.",
    )
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument(
        "--contract",
        type=Path,
        help="Frozen task contract used to bound and identify this run.",
    )
    train_parser.add_argument(
        "--contract-compatibility",
        choices=("exact", "training_resume"),
        default="exact",
        help="Compatibility policy when resuming under a different frozen contract.",
    )
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
    train_parser.add_argument(
        "--resume",
        help="Run directory or checkpoints/ path to restore Brax/Orbax weights from.",
    )
    train_parser.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="Allow resume when recorded env/network/impl differ from this command.",
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

    gate_parser = sub.add_parser(
        "gate",
        help="Apply a frozen task contract's independent promotion suite to reports.",
    )
    gate_parser.add_argument("reports", type=Path, nargs="+")
    gate_parser.add_argument("--contract", type=Path, required=True)
    gate_parser.add_argument("--suite", default="nominal")
    gate_parser.add_argument("--output", type=Path)
    gate_parser.add_argument("--json", action="store_true")
    gate_parser.set_defaults(func=_cmd_gate)

    reward_audit_parser = sub.add_parser(
        "reward-audit",
        help="Check whether reward agrees with independent success outcomes.",
    )
    reward_audit_parser.add_argument("reports", type=Path, nargs="+")
    reward_audit_parser.add_argument("--reward-metric", default="total_reward")
    reward_audit_parser.add_argument("--success-metric", default="task_success")
    reward_audit_parser.add_argument("--output", type=Path)
    reward_audit_parser.add_argument("--json", action="store_true")
    reward_audit_parser.set_defaults(func=_cmd_reward_audit)

    reward_probe_parser = sub.add_parser(
        "reward-probe",
        help="Find concrete high-reward failures across independent evaluation reports.",
    )
    reward_probe_parser.add_argument("reports", type=Path, nargs="+")
    reward_probe_parser.add_argument("--reward-metric", default="total_reward")
    reward_probe_parser.add_argument("--success-metric", default="task_success")
    reward_probe_parser.add_argument("--output", type=Path)
    reward_probe_parser.add_argument("--json", action="store_true")
    reward_probe_parser.set_defaults(func=_cmd_reward_probe)

    eval_suite_parser = sub.add_parser(
        "eval-suite",
        help="Run a frozen contract's independent scenario-by-seed evaluation matrix.",
    )
    eval_suite_parser.add_argument("checkpoint")
    eval_suite_parser.add_argument("--contract", type=Path, required=True)
    eval_suite_parser.add_argument("--suite", default="nominal")
    eval_suite_parser.add_argument("--evaluator", type=Path)
    eval_suite_parser.add_argument("--output", type=Path)
    _add_eval_limit_args(eval_suite_parser)
    eval_suite_parser.add_argument("--json", action="store_true")
    eval_suite_parser.set_defaults(func=_cmd_eval_suite)

    eval_checkpoints_parser = sub.add_parser(
        "eval-checkpoints",
        help="Boundedly evaluate checkpoints currently available in a run directory.",
    )
    eval_checkpoints_parser.add_argument("run_dir", type=Path)
    eval_checkpoints_parser.add_argument("--contract", type=Path, required=True)
    eval_checkpoints_parser.add_argument("--suite", default="nominal")
    eval_checkpoints_parser.add_argument("--evaluator", type=Path)
    eval_checkpoints_parser.add_argument("--max-checkpoints", type=int, default=1)
    _add_eval_limit_args(eval_checkpoints_parser)
    eval_checkpoints_parser.add_argument("--json", action="store_true")
    eval_checkpoints_parser.set_defaults(func=_cmd_eval_checkpoints)

    rank_parser = sub.add_parser(
        "rank-checkpoints",
        help="Rank comparable checkpoints using independent outcomes, never reward.",
    )
    rank_parser.add_argument("reports", type=Path, nargs="+")
    rank_parser.add_argument("--output", type=Path)
    rank_parser.add_argument("--json", action="store_true")
    rank_parser.set_defaults(func=_cmd_rank_checkpoints)

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

    status_parser = sub.add_parser(
        "status",
        help="Show process state and progress for a local or fetched run directory.",
    )
    status_parser.add_argument("output", type=Path, help="Local run directory.")
    status_parser.add_argument("--lines", type=int, default=30)
    status_parser.set_defaults(func=_cmd_status)

    remote_parser = sub.add_parser(
        "remote",
        help="SSH to an already-running Linux GPU (workstation, lab box, or cloud VM).",
    )
    remote_actions = remote_parser.add_subparsers(dest="remote_action", required=True)
    _register_remote_actions(remote_actions)

    return parser


def _cmd_init(args: argparse.Namespace) -> None:
    paths = ensure_project_dirs(args.root)
    for path in paths:
        print(path)


def _cmd_task_init(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite task contract: {args.output}")
    contract = task_contract_template(args.env_name, name=args.name)
    path = save_json(args.output, contract)
    print(path)
    print("Edit every TODO and set a positive episode horizon before freezing.")


def _cmd_task_validate(args: argparse.Namespace) -> int:
    contract, envelope = load_task_contract(args.path)
    result = validate_task_contract(contract)
    payload = {
        "path": str(args.path),
        "frozen": envelope is not None,
        "passed": result.passed,
        "sha256": result.sha256,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    _print(payload, as_json=args.json)
    return 0 if result.passed else 1


def _cmd_task_freeze(args: argparse.Namespace) -> None:
    path = save_frozen_task_contract(args.path, output=args.output)
    print(path)


def _cmd_task_diff(args: argparse.Namespace) -> int:
    left, _ = load_task_contract(args.left)
    right, _ = load_task_contract(args.right)
    changes = diff_task_contracts(left, right)
    payload = {
        "equal": not changes,
        "changes": changes,
    }
    _print(payload, as_json=args.json)
    return 0 if not changes else 1


def _cmd_task_migrate(args: argparse.Namespace) -> None:
    path, steps = save_migrated_task_contract(args.path, output=args.output)
    _print(
        {"path": str(path), "schema_version": SCHEMA_VERSION, "migration_steps": steps},
        as_json=args.json,
    )


def _cmd_task_compatibility(args: argparse.Namespace) -> int:
    left, _ = load_task_contract(args.left)
    right, _ = load_task_contract(args.right)
    result = compare_task_contracts(left, right, policy=args.policy)
    _print(result, as_json=args.json)
    return 0 if result["compatible"] else 1


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
        resume=args.resume,
        allow_resume_mismatch=args.allow_resume_mismatch,
        task_contract_path=args.contract,
        contract_compatibility=args.contract_compatibility,
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


def _cmd_gate(args: argparse.Namespace) -> int:
    contract, _ = load_task_contract(args.contract, require_frozen=True)
    records = load_evaluation_records(args.reports)
    result = evaluate_gate(contract, suite_name=args.suite, records=records)
    if args.output is not None:
        save_json(args.output, result)
    _print(result, as_json=args.json)
    return 0 if result["passed"] else 1


def _cmd_reward_audit(args: argparse.Namespace) -> int:
    records = load_evaluation_records(args.reports)
    result = audit_reward_alignment(
        records,
        reward_metric=args.reward_metric,
        success_metric=args.success_metric,
    )
    if args.output is not None:
        save_json(args.output, result)
    _print(result, as_json=args.json)
    return 0 if result["passed"] else 1


def _cmd_reward_probe(args: argparse.Namespace) -> int:
    records = load_evaluation_records(args.reports)
    result = adversarial_reward_probes(
        records,
        reward_metric=args.reward_metric,
        success_metric=args.success_metric,
    )
    if args.output is not None:
        save_json(args.output, result)
    _print(result, as_json=args.json)
    return 0 if result["passed"] else 1


def _cmd_eval_suite(args: argparse.Namespace) -> int:
    result = run_evaluation_suite(
        args.checkpoint,
        contract_path=args.contract,
        suite_name=args.suite,
        evaluator_path=args.evaluator,
        limits=_evaluation_limits(args),
    )
    output = args.output or (
        Path("reports")
        / f"{slugify(Path(args.checkpoint).name)}_{slugify(args.suite)}_eval_suite.json"
    )
    save_json(output, result)
    _print(result, as_json=args.json)
    return 0 if result["passed"] else 1


def _cmd_eval_checkpoints(args: argparse.Namespace) -> None:
    result = evaluate_checkpoint_directory(
        args.run_dir,
        contract_path=args.contract,
        suite_name=args.suite,
        evaluator_path=args.evaluator,
        max_checkpoints=args.max_checkpoints,
        limits=_evaluation_limits(args, bounded_defaults=True),
    )
    _print(result, as_json=args.json)


def _cmd_rank_checkpoints(args: argparse.Namespace) -> None:
    result = rank_checkpoints(load_suite_reports(args.reports))
    if args.output is not None:
        save_json(args.output, result)
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


def _cmd_status(args: argparse.Namespace) -> None:
    print(describe_run(args.output, lines=args.lines))


def _add_eval_limit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--max-seeds-per-scenario", type=int)
    parser.add_argument("--max-evaluations", type=int)


def _evaluation_limits(
    args: argparse.Namespace,
    *,
    bounded_defaults: bool = False,
) -> EvaluationLimits:
    return EvaluationLimits(
        max_scenarios=args.max_scenarios or (1 if bounded_defaults else None),
        max_seeds_per_scenario=args.max_seeds_per_scenario
        or (1 if bounded_defaults else None),
        max_evaluations=args.max_evaluations or (1 if bounded_defaults else None),
    )


def _parse_preset(value: str) -> str:
    try:
        return canonical_preset(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _register_remote_actions(actions: argparse._SubParsersAction) -> None:
    connect = actions.add_parser(
        "connect",
        help="Open an interactive SSH session to the GPU host.",
    )
    _add_remote_connection_args(connect)
    connect.add_argument(
        "--tunnel-port",
        type=int,
        help="Forward this localhost port to the same port on the host.",
    )
    connect.set_defaults(func=_cmd_remote_connect)

    check = actions.add_parser(
        "check",
        help="Check SSH plus NVIDIA and JAX device visibility.",
    )
    _add_remote_connection_args(check)
    check.set_defaults(func=_cmd_remote_check)

    prepare = actions.add_parser(
        "prepare",
        help="Sync this checkout and prepare a GPU-enabled remote virtualenv.",
    )
    _add_remote_connection_args(prepare)
    prepare.add_argument("--project", type=Path, default=Path("."))
    prepare.add_argument("--remote-dir")
    prepare.add_argument(
        "--jax-cuda",
        choices=("preinstalled", "cuda12", "cuda13"),
        default="preinstalled",
        help="Use a preinstalled system JAX or install a pip CUDA wheel.",
    )
    prepare.add_argument(
        "--python",
        dest="python_command",
        default="python3",
        help="Remote Python 3.11+ executable used to create the training virtualenv.",
    )
    prepare.set_defaults(func=_cmd_remote_prepare)

    smoke = actions.add_parser(
        "smoke",
        help="Run the reset/step environment smoke gate on the remote GPU.",
    )
    _add_remote_connection_args(smoke)
    smoke.add_argument(
        "env_name",
        help="Playground env name or synced custom *.py env module path.",
    )
    smoke.add_argument("--remote-dir")
    smoke.add_argument("--steps", type=int, default=10)
    smoke.set_defaults(func=_cmd_remote_smoke)

    train = actions.add_parser(
        "train",
        help="Run training on a prepared SSH GPU host (smoke preset by default).",
    )
    _add_remote_connection_args(train)
    train.add_argument(
        "env_name",
        help="Playground env name or synced custom *.py env module path.",
    )
    train.add_argument(
        "--preset",
        type=_parse_preset,
        default="smoke",
        help="PPO scale: smoke, local, or large. `cloud` is a hidden alias for large.",
    )
    train.add_argument("--remote-dir")
    train.add_argument(
        "--output",
        help="Remote run directory, relative to --remote-dir unless absolute.",
    )
    train.add_argument(
        "--contract",
        help="Frozen contract path inside the synced remote project.",
    )
    train.add_argument(
        "--contract-compatibility",
        choices=("exact", "training_resume"),
        default="exact",
    )
    train.add_argument(
        "--detach",
        action="store_true",
        help="Keep training after SSH disconnects and write train.log/train.pid.",
    )
    train.add_argument("--timesteps", type=int)
    train.add_argument("--num-envs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument(
        "--impl",
        choices=("auto", "jax", "warp"),
        default="auto",
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--resume",
        help="Remote run directory or checkpoints/ path to restore Brax/Orbax weights from.",
    )
    train.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="Allow resume when recorded env/network/impl differ from this command.",
    )
    train.set_defaults(func=_cmd_remote_train)

    status = actions.add_parser(
        "status",
        help="Show detached process state, progress.json, and recent training log lines.",
    )
    _add_remote_connection_args(status)
    status.add_argument("output", help="Remote output returned by simrig remote train.")
    status.add_argument("--remote-dir")
    status.add_argument("--lines", type=int, default=30)
    status.set_defaults(func=_cmd_remote_status)

    fetch = actions.add_parser(
        "fetch",
        help="Download a remote run directory into local runs/.",
    )
    _add_remote_connection_args(fetch)
    fetch.add_argument("output", help="Remote output returned by simrig remote train.")
    fetch.add_argument("--remote-dir")
    fetch.add_argument("--local-output", type=Path)
    fetch.set_defaults(func=_cmd_remote_fetch)


def _add_remote_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "host",
        help="IP address or DNS name of an already-running Linux GPU SSH host.",
    )
    parser.add_argument(
        "--identity",
        "-i",
        type=Path,
        help="Path to the SSH private key used to reach the host.",
    )
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--port", type=int, default=22)


def _remote_config(args: argparse.Namespace) -> SSHConfig:
    return SSHConfig(
        host=args.host,
        user=args.user,
        identity=args.identity,
        port=args.port,
    )


def _cmd_remote_connect(args: argparse.Namespace) -> int:
    return connect_remote(_remote_config(args), tunnel_port=args.tunnel_port)


def _cmd_remote_check(args: argparse.Namespace) -> int:
    return check_remote(_remote_config(args))


def _cmd_remote_prepare(args: argparse.Namespace) -> None:
    prepare_remote(
        _remote_config(args),
        project_dir=args.project,
        remote_dir=args.remote_dir,
        jax_cuda=args.jax_cuda,
        python_command=args.python_command,
    )
    print(f"prepared remote project: {args.remote_dir or f'/home/{args.user}/simrig'}")


def _cmd_remote_train(args: argparse.Namespace) -> int:
    result = train_remote(
        _remote_config(args),
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
        resume=args.resume,
        allow_resume_mismatch=args.allow_resume_mismatch,
        task_contract=args.contract,
        contract_compatibility=args.contract_compatibility,
    )
    if result.returncode == 0:
        mode = "detached run" if result.detached else "run"
        print(f"remote {mode}: {result.output_dir}")
    return result.returncode


def _cmd_remote_smoke(args: argparse.Namespace) -> int:
    return smoke_remote(
        _remote_config(args),
        args.env_name,
        remote_dir=args.remote_dir,
        steps=args.steps,
    )


def _cmd_remote_status(args: argparse.Namespace) -> int:
    return status_remote(
        _remote_config(args),
        args.output,
        remote_dir=args.remote_dir,
        lines=args.lines,
    )


def _cmd_remote_fetch(args: argparse.Namespace) -> None:
    destination = fetch_remote(
        _remote_config(args),
        args.output,
        remote_dir=args.remote_dir,
        local_output=args.local_output,
    )
    print(f"downloaded run: {destination}")
    print(
        "After verifying these artifacts or persistent storage, stop the GPU "
        "host if it is a billable VM."
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
