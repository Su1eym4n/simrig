from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from simrig.remote import (
    SSHConfig,
    fetch_remote,
    prepare_remote,
    smoke_remote,
    ssh_command,
    status_remote,
    train_remote,
    _check_local_requirements,
)


class RemoteSSHTests(unittest.TestCase):
    def test_ssh_command_uses_identity_user_port_and_tunnel(self) -> None:
        config = SSHConfig(
            "203.0.113.12",
            user="ubuntu",
            identity=Path("keys/id_ed25519"),
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
            SSHConfig("gpu.example; touch /tmp/nope")

    def test_prepare_syncs_checkout_then_installs_and_checks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "pyproject.toml").write_text(
                '[project]\ndependencies = ["jax==0.10.2"]\n',
                encoding="utf-8",
            )
            (project / "simrig").mkdir()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("simrig.remote._check_local_requirements"),
                patch("simrig.remote.subprocess.run", return_value=completed) as run,
            ):
                prepare_remote(
                    SSHConfig("gpu.example"),
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
            (project / "pyproject.toml").write_text(
                '[project]\ndependencies = []\n'
                '[project.optional-dependencies]\nplayground = ["jax==0.10.2"]\n',
                encoding="utf-8",
            )
            (project / "simrig").mkdir()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("simrig.remote._check_local_requirements"),
                patch("simrig.remote.subprocess.run", return_value=completed) as run,
            ):
                prepare_remote(
                    SSHConfig("gpu.example"),
                    project_dir=project,
                    jax_cuda="cuda12",
                )

        setup = run.call_args_list[2].args[0][-1]
        self.assertNotIn("--system-site-packages", setup)
        self.assertIn("jax[cuda12]==0.10.2", setup)
        self.assertLess(setup.index(".[playground]"), setup.index("jax[cuda12]==0.10.2"))

    def test_prepare_accepts_explicit_remote_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            (project / "simrig").mkdir()
            completed = subprocess.CompletedProcess([], 0)
            with (
                patch("simrig.remote._check_local_requirements"),
                patch("simrig.remote.subprocess.run", return_value=completed) as run,
            ):
                prepare_remote(
                    SSHConfig("gpu.example"),
                    project_dir=project,
                    python_command="/usr/bin/python3.12",
                )

        setup = run.call_args_list[2].args[0][-1]
        self.assertIn("/usr/bin/python3.12 -c", setup)
        self.assertIn("/usr/bin/python3.12 -m venv", setup)

    def test_train_detached_defaults_to_smoke_and_records_pid(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.remote._check_local_requirements"),
            patch("simrig.remote.timestamp", return_value="20260102-030405"),
            patch("simrig.remote.subprocess.run", return_value=completed) as run,
        ):
            result = train_remote(
                SSHConfig("gpu.example"),
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
        self.assertIn("--impl auto", remote)
        self.assertIn("--seed 0", remote)
        self.assertIn("nohup", remote)
        self.assertIn("train.log", remote)
        self.assertIn("train.pid", remote)
        self.assertIn("{ nohup", remote)
        syntax = subprocess.run(["sh", "-n", "-c", remote], check=False)
        self.assertEqual(syntax.returncode, 0)

    def test_train_forwards_impl_seed_resume_and_randomization_choice(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.remote._check_local_requirements"),
            patch("simrig.remote.subprocess.run", return_value=completed) as run,
        ):
            train_remote(
                SSHConfig("gpu.example"),
                "Go1JoystickFlatTerrain",
                impl="warp",
                seed=11,
                domain_randomization=False,
                resume="runs/old/checkpoints",
                task_contract="task.frozen.json",
            )

        remote = run.call_args.args[0][-1]
        self.assertIn("--impl warp", remote)
        self.assertIn("--seed 11", remote)
        self.assertIn("--no-domain-randomization", remote)
        self.assertIn("--resume runs/old/checkpoints", remote)
        self.assertIn("--contract task.frozen.json", remote)

    def test_train_canonicalizes_cloud_preset_to_large(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.remote._check_local_requirements"),
            patch("simrig.remote.timestamp", return_value="20260102-030405"),
            patch("simrig.remote.subprocess.run", return_value=completed) as run,
        ):
            result = train_remote(
                SSHConfig("gpu.example"),
                "Go1JoystickFlatTerrain",
                preset_name="cloud",
            )

        self.assertEqual(
            result.output_dir,
            "/home/ubuntu/simrig/runs/20260102-030405-Go1JoystickFlatTerrain-large",
        )
        remote = run.call_args.args[0][-1]
        self.assertIn("--preset large", remote)
        self.assertNotIn("--preset cloud", remote)

    def test_remote_smoke_checks_gpu_then_environment(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.remote._check_local_requirements"),
            patch("simrig.remote.subprocess.run", return_value=completed) as run,
        ):
            code = smoke_remote(
                SSHConfig("gpu.example"),
                "envs/reach.py",
                steps=7,
            )

        self.assertEqual(code, 0)
        remote = run.call_args.args[0][-1]
        self.assertIn("nvidia-smi -L", remote)
        self.assertIn("smoke envs/reach.py --steps 7", remote)

    def test_remote_output_cannot_escape_project(self) -> None:
        with (
            patch("simrig.remote._check_local_requirements"),
            self.assertRaises(ValueError),
        ):
            train_remote(
                SSHConfig("gpu.example"),
                "Go1JoystickFlatTerrain",
                output="/tmp/outside-project",
            )

    def test_fetch_downloads_run_to_local_runs_directory(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "downloaded"
            with (
                patch("simrig.remote._check_local_requirements"),
                patch("simrig.remote.subprocess.run", return_value=completed) as run,
            ):
                destination = fetch_remote(
                    SSHConfig("gpu.example"),
                    "runs/large-run",
                    local_output=local,
                )

            self.assertEqual(destination, local.resolve())
            self.assertTrue(destination.is_dir())
            command = run.call_args.args[0]
            self.assertEqual(command[0], "rsync")
            self.assertIn("--progress", command)
            self.assertNotIn("--info=progress2", command)
            self.assertIn(
                "ubuntu@gpu.example:/home/ubuntu/simrig/runs/large-run/",
                command,
            )

    def test_status_reads_progress_json_and_artifacts(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            patch("simrig.remote._check_local_requirements"),
            patch("simrig.remote.subprocess.run", return_value=completed) as run,
        ):
            status_remote(SSHConfig("gpu.example"), "runs/large-run")

        remote = run.call_args.args[0][-1]
        self.assertIn("policy.params", remote)
        self.assertIn("final_metrics.json", remote)
        self.assertIn("checkpoints", remote)
        self.assertIn("progress.json", remote)
        self.assertIn("run_manifest.json", remote)
        self.assertIn("artifacts=complete", remote)

    def test_private_key_permissions_must_not_be_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "id_ed25519"
            key.write_text("not-a-key", encoding="utf-8")
            key.chmod(0o644)

            with self.assertRaises(PermissionError):
                _check_local_requirements(
                    SSHConfig("gpu.example", identity=key),
                    commands=(),
                )

    def test_private_key_must_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "id_ed25519"
            key.write_text("not-a-key", encoding="utf-8")
            key.chmod(0o600)
            invalid = subprocess.CompletedProcess(
                [],
                255,
                stderr="not a key file",
            )
            with (
                patch("simrig.remote.shutil.which", return_value="/usr/bin/ssh-keygen"),
                patch("simrig.remote.subprocess.run", return_value=invalid),
                self.assertRaises(ValueError),
            ):
                _check_local_requirements(
                    SSHConfig("gpu.example", identity=key),
                    commands=(),
                )


if __name__ == "__main__":
    unittest.main()
