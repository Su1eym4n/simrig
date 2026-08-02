from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from simrig.lambda_cloud import (
    LambdaSSHConfig,
    fetch_lambda,
    prepare_lambda,
    smoke_lambda,
    ssh_command,
    status_lambda,
    train_lambda,
    _check_local_requirements,
)


class LambdaCloudTests(unittest.TestCase):
    def test_ssh_command_uses_identity_user_port_and_tunnel(self) -> None:
        config = LambdaSSHConfig(
            "203.0.113.12",
            user="ubuntu",
            identity=Path("keys/lambda.pem"),
            port=2222,
        )

        command = ssh_command(config, tunnel_port=8765)

        self.assertEqual(command[0:3], ["ssh", "-p", "2222"])
        self.assertIn("-i", command)
        self.assertIn("IdentitiesOnly=yes", command)
        self.assertIn("ExitOnForwardFailure=yes", command)
        self.assertIn("8765:127.0.0.1:8765", command)
        self.assertEqual(command[-1], "ubuntu@203.0.113.12")

    def test_config_rejects_shell_metacharacters_in_host(self) -> None:
        with self.assertRaises(ValueError):
            LambdaSSHConfig("gpu.example; touch /tmp/nope")

    def test_prepare_syncs_checkout_then_installs_and_checks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (project / "simrig").mkdir()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("simrig.lambda_cloud._check_local_requirements"),
                patch("simrig.lambda_cloud.subprocess.run", return_value=completed) as run,
            ):
                prepare_lambda(
                    LambdaSSHConfig("gpu.example"),
                    project_dir=project,
                )

        self.assertEqual(run.call_count, 3)
        rsync = run.call_args_list[1].args[0]
        self.assertEqual(rsync[0], "rsync")
        self.assertIn("--exclude=runs/", rsync)
        self.assertEqual(rsync[-1], "ubuntu@gpu.example:/home/ubuntu/simrig/")
        setup = run.call_args_list[2].args[0][-1]
        self.assertIn("--system-site-packages", setup)
        self.assertIn("--clear", setup)
        self.assertIn("Python 3.11 or newer", setup)
        self.assertIn(".[playground]", setup)
        self.assertIn("nvidia-smi -L", setup)
        self.assertIn("d.platform", setup)
        self.assertIn("JAX cannot see a GPU", setup)

    def test_prepare_can_install_a_pip_cuda_jax_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (project / "simrig").mkdir()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("simrig.lambda_cloud._check_local_requirements"),
                patch("simrig.lambda_cloud.subprocess.run", return_value=completed) as run,
            ):
                prepare_lambda(
                    LambdaSSHConfig("gpu.example"),
                    project_dir=project,
                    jax_cuda="cuda12",
                )

        setup = run.call_args_list[2].args[0][-1]
        self.assertNotIn("--system-site-packages", setup)
        self.assertIn("jax[cuda12]", setup)

    def test_prepare_accepts_explicit_remote_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (project / "simrig").mkdir()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("simrig.lambda_cloud._check_local_requirements"),
                patch("simrig.lambda_cloud.subprocess.run", return_value=completed) as run,
            ):
                prepare_lambda(
                    LambdaSSHConfig("gpu.example"),
                    project_dir=project,
                    python_command="/usr/bin/python3.12",
                )

        setup = run.call_args_list[2].args[0][-1]
        self.assertIn("/usr/bin/python3.12 -c", setup)
        self.assertIn("/usr/bin/python3.12 -m venv", setup)

    def test_train_detached_defaults_to_smoke_and_records_pid(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.lambda_cloud._check_local_requirements"),
            patch("simrig.lambda_cloud.timestamp", return_value="20260102-030405"),
            patch("simrig.lambda_cloud.subprocess.run", return_value=completed) as run,
        ):
            result = train_lambda(
                LambdaSSHConfig("gpu.example"),
                "Go1JoystickFlatTerrain",
                detach=True,
            )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.detached)
        self.assertEqual(
            result.output_dir,
            "/home/ubuntu/simrig/runs/20260102-030405-Go1JoystickFlatTerrain-smoke",
        )
        remote = run.call_args.args[0][-1]
        self.assertIn("--preset smoke", remote)
        self.assertIn("nohup", remote)
        self.assertIn("train.log", remote)
        self.assertIn("train.pid", remote)
        self.assertIn("{ nohup", remote)
        syntax = subprocess.run(["sh", "-n", "-c", remote], check=False)
        self.assertEqual(syntax.returncode, 0)

    def test_remote_smoke_checks_gpu_then_environment(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.lambda_cloud._check_local_requirements"),
            patch("simrig.lambda_cloud.subprocess.run", return_value=completed) as run,
        ):
            code = smoke_lambda(
                LambdaSSHConfig("gpu.example"),
                "envs/reach.py",
                steps=7,
            )

        self.assertEqual(code, 0)
        remote = run.call_args.args[0][-1]
        self.assertIn("nvidia-smi -L", remote)
        self.assertIn("smoke envs/reach.py --steps 7", remote)

    def test_remote_output_cannot_escape_project(self) -> None:
        with (
            patch("simrig.lambda_cloud._check_local_requirements"),
            self.assertRaises(ValueError),
        ):
            train_lambda(
                LambdaSSHConfig("gpu.example"),
                "Go1JoystickFlatTerrain",
                output="/tmp/outside-project",
            )

    def test_fetch_downloads_run_to_local_runs_directory(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "downloaded"
            with (
                patch("simrig.lambda_cloud._check_local_requirements"),
                patch("simrig.lambda_cloud.subprocess.run", return_value=completed) as run,
            ):
                destination = fetch_lambda(
                    LambdaSSHConfig("gpu.example"),
                    "runs/cloud-run",
                    local_output=local,
                )

            self.assertEqual(destination, local.resolve())
            self.assertTrue(destination.is_dir())
            command = run.call_args.args[0]
            self.assertEqual(command[0], "rsync")
            self.assertIn("--progress", command)
            self.assertNotIn("--info=progress2", command)
            self.assertIn(
                "ubuntu@gpu.example:/home/ubuntu/simrig/runs/cloud-run/",
                command,
            )

    def test_status_requires_policy_metrics_and_checkpoint(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.lambda_cloud._check_local_requirements"),
            patch("simrig.lambda_cloud.subprocess.run", return_value=completed) as run,
        ):
            status_lambda(LambdaSSHConfig("gpu.example"), "runs/cloud-run")

        remote = run.call_args.args[0][-1]
        self.assertIn("policy.params", remote)
        self.assertIn("final_metrics.json", remote)
        self.assertIn("checkpoints", remote)
        self.assertIn("artifacts=complete", remote)

    def test_private_key_permissions_must_not_be_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "lambda.pem"
            key.write_text("not-a-key", encoding="utf-8")
            key.chmod(0o644)

            with self.assertRaises(PermissionError):
                _check_local_requirements(
                    LambdaSSHConfig("gpu.example", identity=key),
                    commands=(),
                )

    def test_private_key_must_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "lambda.pem"
            key.write_text("not-a-key", encoding="utf-8")
            key.chmod(0o600)
            invalid = subprocess.CompletedProcess(
                [],
                255,
                stderr="not a key file",
            )
            with (
                patch("simrig.lambda_cloud.shutil.which", return_value="/usr/bin/ssh-keygen"),
                patch("simrig.lambda_cloud.subprocess.run", return_value=invalid),
                self.assertRaises(ValueError),
            ):
                _check_local_requirements(
                    LambdaSSHConfig("gpu.example", identity=key),
                    commands=(),
                )


if __name__ == "__main__":
    unittest.main()
