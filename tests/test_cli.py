from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from simrig import __version__
from simrig.cli import build_parser, main


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class CliTests(unittest.TestCase):
    def test_cli_version(self) -> None:
        parser = build_parser()
        stdout = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            parser.parse_args(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), f"simrig {__version__}")

    def test_cli_init_creates_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, stdout, _ = _run_cli(["init", "--root", tmp])

            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "runs").is_dir())
            self.assertTrue((Path(tmp) / "reports").is_dir())
            self.assertIn("runs", stdout)

    def test_task_contract_cli_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            draft = Path(tmp) / "task.json"
            frozen = Path(tmp) / "task.frozen.json"
            code, _, _ = _run_cli(
                ["task", "init", "RobotTask", "--output", str(draft)]
            )
            self.assertEqual(code, 0)

            code, stdout, _ = _run_cli(["task", "validate", str(draft), "--json"])
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(stdout)["passed"])

            payload = json.loads(draft.read_text(encoding="utf-8"))
            payload["behavior"]["objective"] = "Track a command."
            payload["interfaces"]["actions"] = "Normalized actuator commands."
            payload["interfaces"]["observations"] = "Deployable proprioception and command."
            payload["reset"]["training"] = "Nominal state with sampled commands."
            payload["reset"]["native"] = "Same as training; no predecessor."
            payload["episode"]["horizon_steps"] = 100
            payload["outcomes"]["success"] = "Tracking error below threshold."
            payload["outcomes"]["failure"] = "Fall, non-finite state, or timeout."
            draft.write_text(json.dumps(payload), encoding="utf-8")

            code, _, _ = _run_cli(
                ["task", "freeze", str(draft), "--output", str(frozen)]
            )
            self.assertEqual(code, 0)
            self.assertTrue(frozen.is_file())

            code, stdout, _ = _run_cli(["task", "validate", str(frozen), "--json"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stdout)["frozen"])

    def test_task_contract_cli_migration_and_compatibility(self) -> None:
        from simrig.task_contract import task_contract_template

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = task_contract_template("RobotTask")
            old["schema_version"] = 1
            del old["evaluation"]["evaluator"]
            del old["evaluation"]["predicates"]
            del old["outcomes"]["failure_taxonomy"]
            old_path = root / "task-v1.json"
            migrated_path = root / "task-v2.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")

            code, stdout, _ = _run_cli(
                [
                    "task",
                    "migrate",
                    str(old_path),
                    "--output",
                    str(migrated_path),
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["schema_version"], 2)
            self.assertIn("evaluator", json.loads(migrated_path.read_text())["evaluation"])

            revised = json.loads(migrated_path.read_text(encoding="utf-8"))
            revised["compute"]["max_timesteps"] *= 2
            revised_path = root / "task-v2-revised.json"
            revised_path.write_text(json.dumps(revised), encoding="utf-8")
            code, stdout, _ = _run_cli(
                [
                    "task",
                    "compatibility",
                    str(migrated_path),
                    str(revised_path),
                    "--policy",
                    "training_resume",
                    "--json",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stdout)["compatible"])

    def test_gate_cli_returns_nonzero_when_suite_fails(self) -> None:
        from simrig.task_contract import freeze_task_contract, task_contract_template

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = task_contract_template("RobotTask")
            contract["behavior"]["objective"] = "Track a command."
            contract["interfaces"]["actions"] = "Normalized actuator commands."
            contract["interfaces"]["observations"] = "Proprioception and command."
            contract["reset"]["training"] = "Nominal command distribution."
            contract["reset"]["native"] = "Same as training."
            contract["episode"]["horizon_steps"] = 100
            contract["outcomes"]["success"] = "Independent tracking success."
            contract["outcomes"]["failure"] = "Fall or timeout."
            frozen = root / "task.frozen.json"
            report = root / "eval.json"
            frozen.write_text(json.dumps(freeze_task_contract(contract)), encoding="utf-8")
            report.write_text(
                json.dumps({"seed": 0, "task_success": False}), encoding="utf-8"
            )

            code, stdout, _ = _run_cli(
                ["gate", str(report), "--contract", str(frozen), "--json"]
            )

        self.assertEqual(code, 1)
        self.assertFalse(json.loads(stdout)["passed"])

    def test_eval_suite_and_checkpoint_ranking_cli(self) -> None:
        suite_result = {
            "passed": True,
            "checkpoint": {"path": "policy.params", "sha256": "abc"},
        }
        ranking = {"reward_used_for_ranking": False, "checkpoints": []}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite.json"
            with patch("simrig.cli.run_evaluation_suite", return_value=suite_result) as run:
                code, _, _ = _run_cli(
                    [
                        "eval-suite",
                        "policy.params",
                        "--contract",
                        "task.frozen.json",
                        "--suite",
                        "promotion",
                        "--output",
                        str(output),
                        "--max-scenarios",
                        "2",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.kwargs["limits"].max_scenarios, 2)
            self.assertTrue(output.is_file())

            with (
                patch("simrig.cli.load_suite_reports", return_value=[suite_result]),
                patch("simrig.cli.rank_checkpoints", return_value=ranking),
            ):
                code, stdout, _ = _run_cli(
                    ["rank-checkpoints", str(output), "--json"]
                )
            self.assertEqual(code, 0)
            self.assertFalse(json.loads(stdout)["reward_used_for_ranking"])

    def test_cli_list_models_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "bot"
            model.mkdir()
            (root / "README.md").write_text("menagerie\n", encoding="utf-8")
            (model / "scene.xml").write_text("<mujoco model='x'/>\n", encoding="utf-8")

            code, stdout, _ = _run_cli(["list-models", "--menagerie", tmp, "--json"])

            self.assertEqual(code, 0)
            data = json.loads(stdout)
            self.assertEqual(data[0]["name"], "bot")

    def test_cli_unknown_model_returns_error(self) -> None:
        code, _, stderr = _run_cli(["inspect-model", "definitely_missing_model"])

        self.assertEqual(code, 1)
        self.assertIn("error", stderr)

    def test_cli_new_env_creates_editable_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, stdout, _ = _run_cli(
                [
                    "new-env",
                    "reach bot",
                    "--model",
                    "models/bot.xml",
                    "--root",
                    tmp,
                ]
            )

            self.assertEqual(code, 0)
            path = Path(tmp) / "reach_bot.py"
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("NotImplementedError", text)
            self.assertIn("SECTION: rewards", text)
            self.assertIn("NOT TRAINABLE YET", text)
            self.assertIn(str(path), stdout)

    def test_cli_validate_env_passes_fresh_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            create_code, create_stdout, _ = _run_cli(
                ["new-env", "reach", "--model", "models/bot.xml", "--root", tmp]
            )
            self.assertEqual(create_code, 0)
            path = create_stdout.strip().splitlines()[-1]

            code, stdout, _ = _run_cli(["validate-env", path, "--json"])

            self.assertEqual(code, 0)
            data = json.loads(stdout)
            self.assertTrue(data["passed"])
            self.assertFalse(data["trainable"])

    def test_cli_validate_env_fails_for_missing_file(self) -> None:
        code, stdout, _ = _run_cli(["validate-env", "/tmp/simrig_no_such_env.py", "--json"])

        self.assertEqual(code, 1)
        data = json.loads(stdout)
        self.assertFalse(data["passed"])
        self.assertTrue(any("file not found" in item for item in data["missing"]))

    def test_cli_validate_env_runtime_flag_parses(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["validate-env", "envs/reach.py", "--runtime", "--vision"]
        )
        self.assertTrue(args.runtime)
        self.assertTrue(args.vision)
        self.assertEqual(args.path, Path("envs/reach.py"))

    def test_demo_command_parses_policy_env_and_command(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "demo",
                "policy.params",
                "--env",
                "Go1JoystickFlatTerrain",
                "--command",
                "0",
                "0",
                "0",
            ]
        )

        self.assertEqual(args.checkpoint, "policy.params")
        self.assertEqual(args.env_name, "Go1JoystickFlatTerrain")
        self.assertEqual(args.command, [0.0, 0.0, 0.0])

    def test_eval_parses_huggingface_policy_ref(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "eval",
                "hf://simrig/go1-policy/policy.params",
                "--env",
                "Go1JoystickFlatTerrain",
                "--hf-revision",
                "abc123",
            ]
        )

        self.assertEqual(args.checkpoint, "hf://simrig/go1-policy/policy.params")
        self.assertEqual(args.hf_revision, "abc123")

    def test_eval_passes_seed_and_command(self) -> None:
        with (
            patch("simrig.cli.resolve_policy_checkpoint", return_value=Path("policy.params")),
            patch("simrig.cli.eval_policy", return_value={"average_reward": 1.0}) as evaluate,
            patch("simrig.cli.save_json"),
        ):
            code, _, _ = _run_cli(
                [
                    "eval",
                    "policy.params",
                    "--env",
                    "Go1JoystickFlatTerrain",
                    "--seed",
                    "4",
                    "--command",
                    "0.8",
                    "0",
                    "0",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(evaluate.call_args.kwargs["seed"], 4)
        self.assertEqual(evaluate.call_args.kwargs["command"], (0.8, 0.0, 0.0))
        self.assertFalse(evaluate.call_args.kwargs["allow_runtime_mismatch"])

    def test_train_passes_impl_seed_and_randomization_choice(self) -> None:
        with patch("simrig.cli.train_ppo") as train:
            train.return_value.output_dir = "runs/test"
            code, _, _ = _run_cli(
                [
                    "train",
                    "Go1JoystickFlatTerrain",
                    "--impl",
                    "warp",
                    "--seed",
                    "12",
                    "--contract",
                    "task.frozen.json",
                    "--no-domain-randomization",
                    "--resume",
                    "runs/old",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(train.call_args.kwargs["impl"], "warp")
        self.assertEqual(train.call_args.kwargs["seed"], 12)
        self.assertFalse(train.call_args.kwargs["domain_randomization"])
        self.assertEqual(train.call_args.kwargs["resume"], "runs/old")
        self.assertEqual(train.call_args.kwargs["task_contract_path"], Path("task.frozen.json"))

    def test_eval_can_explicitly_allow_runtime_mismatch(self) -> None:
        with (
            patch("simrig.cli.resolve_policy_checkpoint", return_value=Path("policy.params")),
            patch("simrig.cli.eval_policy", return_value={"task_success": None}) as evaluate,
            patch("simrig.cli.save_json"),
        ):
            code, _, _ = _run_cli(
                [
                    "eval",
                    "policy.params",
                    "--env",
                    "Go1JoystickFlatTerrain",
                    "--allow-runtime-mismatch",
                ]
            )

        self.assertEqual(code, 0)
        self.assertTrue(evaluate.call_args.kwargs["allow_runtime_mismatch"])

    def test_preview_command_parses_browser_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "preview",
                "policy.params",
                "--env",
                "Go1JoystickFlatTerrain",
                "--command",
                "0.5",
                "0",
                "0",
                "--port",
                "8765",
                "--render-mode",
                "topdown",
                "--paused",
            ]
        )

        self.assertEqual(args.checkpoint, "policy.params")
        self.assertEqual(args.env_name, "Go1JoystickFlatTerrain")
        self.assertEqual(args.command, [0.5, 0.0, 0.0])
        self.assertEqual(args.port, 8765)
        self.assertEqual(args.render_mode, "topdown")
        self.assertTrue(args.paused)

    def test_view_model_command_parses_model_and_port(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "view-model",
                "unitree_go1/scene.xml",
                "--port",
                "8766",
            ]
        )

        self.assertEqual(args.model_or_xml, "unitree_go1/scene.xml")
        self.assertEqual(args.port, 8766)
        self.assertEqual(args.render_mode, "threejs")

    def test_remote_train_defaults_to_smoke(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "remote",
                "train",
                "203.0.113.12",
                "Go1JoystickFlatTerrain",
                "--identity",
                "id_ed25519",
                "--detach",
                "--contract",
                "task.frozen.json",
            ]
        )

        self.assertEqual(args.host, "203.0.113.12")
        self.assertEqual(args.env_name, "Go1JoystickFlatTerrain")
        self.assertEqual(args.identity, Path("id_ed25519"))
        self.assertEqual(args.preset, "smoke")
        self.assertEqual(args.impl, "auto")
        self.assertEqual(args.seed, 0)
        self.assertTrue(args.domain_randomization)
        self.assertTrue(args.detach)
        self.assertEqual(args.contract, "task.frozen.json")

    def test_remote_prepare_accepts_remote_python(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "remote",
                "prepare",
                "203.0.113.12",
                "--python",
                "/usr/bin/python3.12",
            ]
        )

        self.assertEqual(args.python_command, "/usr/bin/python3.12")

    def test_preset_cloud_is_hidden_alias_for_large(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            ["train", "Go1JoystickFlatTerrain", "--preset", "cloud"]
        )

        self.assertEqual(args.preset, "large")

    def test_cloud_lambda_command_is_removed(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["cloud", "lambda", "train", "203.0.113.12", "Go1JoystickFlatTerrain"]
            )

    def test_preview_accepts_episode_auto_reset_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "preview",
                "runs/example/policy.params",
                "--env",
                "ExampleEnv",
                "--auto-reset",
                "--auto-reset-delay",
                "2.5",
            ]
        )

        self.assertTrue(args.auto_reset)
        self.assertEqual(args.auto_reset_delay, 2.5)


if __name__ == "__main__":
    unittest.main()
