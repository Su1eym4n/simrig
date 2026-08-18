"""SSH-based training helpers for Lambda On-Demand Cloud instances.

SimRig deliberately does not provision or terminate billable instances.  These
helpers operate on an instance the user has already launched and make the
project sync, GPU verification, training, monitoring, and artifact download
steps reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import tomllib
from typing import Sequence

from simrig.io import slugify, timestamp


_HOST_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def _pinned_jax_version(project: Path) -> str:
    """Return the exact JAX version required by the synced SimRig checkout."""
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    project_metadata = metadata.get("project", {})
    dependencies = list(project_metadata.get("dependencies", []))
    optional_dependencies = project_metadata.get("optional-dependencies", {})
    for group in optional_dependencies.values():
        dependencies.extend(group)
    for dependency in dependencies:
        match = re.fullmatch(r"jax\s*==\s*([A-Za-z0-9._+-]+)", dependency)
        if match:
            return match.group(1)
    raise ValueError("SimRig pyproject.toml must pin JAX with an exact jax==VERSION dependency")


@dataclass(frozen=True)
class LambdaSSHConfig:
    """Connection details for a Lambda Cloud instance."""

    host: str
    user: str = "ubuntu"
    identity: Path | None = None
    port: int = 22

    def __post_init__(self) -> None:
        if not self.host or not _HOST_RE.fullmatch(self.host):
            raise ValueError("Lambda host must be an IP address or DNS name without whitespace.")
        if not self.user or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", self.user):
            raise ValueError("SSH user contains unsupported characters.")
        if not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535.")

    @property
    def target(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.user}@{host}"


@dataclass(frozen=True)
class LambdaTrainResult:
    """Location and process result for a remote training launch."""

    output_dir: str
    detached: bool
    returncode: int


def ssh_command(
    config: LambdaSSHConfig,
    remote_command: str | None = None,
    *,
    batch: bool = False,
    tunnel_port: int | None = None,
) -> list[str]:
    """Build an argv-safe SSH command."""
    command = ["ssh", "-p", str(config.port)]
    if config.identity is not None:
        command.extend(
            [
                "-i",
                str(config.identity.expanduser().resolve()),
                "-o",
                "IdentitiesOnly=yes",
            ]
        )
    command.extend(["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"])
    if batch:
        command.extend(["-o", "BatchMode=yes"])
    if tunnel_port is not None:
        if not 1 <= tunnel_port <= 65535:
            raise ValueError("Tunnel port must be between 1 and 65535.")
        command.extend(
            [
                "-o",
                "ExitOnForwardFailure=yes",
                "-L",
                f"{tunnel_port}:127.0.0.1:{tunnel_port}",
            ]
        )
    command.append(config.target)
    if remote_command is not None:
        command.append(remote_command)
    return command


def connect_lambda(config: LambdaSSHConfig, *, tunnel_port: int | None = None) -> int:
    """Open an interactive SSH session, optionally forwarding one localhost port."""
    _check_local_requirements(config, commands=("ssh",))
    return subprocess.run(ssh_command(config, tunnel_port=tunnel_port), check=False).returncode


def check_lambda(config: LambdaSSHConfig) -> int:
    """Verify SSH, NVIDIA visibility, and JAX GPU visibility when JAX is installed."""
    _check_local_requirements(config, commands=("ssh",))
    python_check = """\
import importlib.util
spec = importlib.util.find_spec("jax")
if spec is None:
    print("jax: not installed (prepare will install SimRig)")
else:
    import jax
    devices = jax.devices()
    print("jax devices:", devices)
    if not any(device.platform == "gpu" for device in devices):
        raise SystemExit("JAX is installed but cannot see a GPU")
"""
    remote = _shell_chain(
        [
            ["nvidia-smi", "-L"],
            ["python3", "-c", python_check],
        ]
    )
    return subprocess.run(ssh_command(config, remote, batch=True), check=False).returncode


def prepare_lambda(
    config: LambdaSSHConfig,
    *,
    project_dir: Path | str = ".",
    remote_dir: str | None = None,
    jax_cuda: str = "preinstalled",
    python_command: str = "python3",
) -> None:
    """Sync a SimRig source checkout and install it in a remote virtualenv."""
    _check_local_requirements(config, commands=("ssh", "rsync"))
    project = Path(project_dir).expanduser().resolve()
    if not (project / "pyproject.toml").is_file() or not (project / "simrig").is_dir():
        raise ValueError(f"Not a SimRig source checkout: {project}")
    if jax_cuda not in {"preinstalled", "cuda12", "cuda13"}:
        raise ValueError("JAX CUDA mode must be one of: preinstalled, cuda12, cuda13")
    if not re.fullmatch(r"/?[A-Za-z0-9._/-]+", python_command) or ".." in PurePosixPath(
        python_command
    ).parts:
        raise ValueError("Remote Python command contains unsupported characters.")
    remote_root = _resolve_remote_dir(config, remote_dir)

    mkdir = _shell_command(["mkdir", "-p", remote_root])
    subprocess.run(ssh_command(config, mkdir, batch=True), check=True)

    transport = ssh_command(config, batch=True)
    # rsync supplies the destination host itself, so omit the final SSH target.
    transport.pop()
    sync = [
        "rsync",
        "-az",
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=__pycache__/",
        "--exclude=.pytest_cache/",
        "--exclude=runs/",
        "--exclude=reports/",
        "--exclude=*.pyc",
        "-e",
        shlex.join(transport),
        f"{project}{os.sep}",
        f"{config.target}:{remote_root}/",
    ]
    subprocess.run(sync, check=True)

    gpu_check = (
        "import jax; "
        "devices=jax.devices(); "
        "print('jax devices:', devices); "
        "assert any(d.platform == 'gpu' for d in devices), "
        "'JAX cannot see a GPU; inspect the Lambda image and JAX installation'"
    )
    python_check = (
        "import sys; print('python:', sys.version.split()[0]); "
        "assert sys.version_info >= (3, 11), "
        "'SimRig Playground requires Python 3.11 or newer; pass --python PATH'"
    )
    venv = [python_command, "-m", "venv", "--clear"]
    if jax_cuda == "preinstalled":
        venv.append("--system-site-packages")
    venv.append(".venv")
    setup_commands = [
        ["cd", remote_root],
        [python_command, "-c", python_check],
        venv,
        [".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"],
        [".venv/bin/python", "-m", "pip", "install", "-e", ".[playground]"],
    ]
    if jax_cuda != "preinstalled":
        jax_version = _pinned_jax_version(project)
        setup_commands.append(
            [
                ".venv/bin/python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                f"jax[{jax_cuda}]=={jax_version}",
            ]
        )
    setup_commands.extend(
        [
            ["nvidia-smi", "-L"],
            [".venv/bin/python", "-c", gpu_check],
        ]
    )
    setup = _shell_chain(setup_commands)
    subprocess.run(ssh_command(config, setup, batch=True), check=True)


def train_lambda(
    config: LambdaSSHConfig,
    env_name: str,
    *,
    preset_name: str = "smoke",
    remote_dir: str | None = None,
    output: str | None = None,
    detach: bool = False,
    timesteps: int | None = None,
    num_envs: int | None = None,
    batch_size: int | None = None,
    impl: str = "auto",
    seed: int = 0,
    domain_randomization: bool = True,
) -> LambdaTrainResult:
    """Run SimRig training on a prepared Lambda instance."""
    _check_local_requirements(config, commands=("ssh",))
    if preset_name not in {"smoke", "local", "cloud"}:
        raise ValueError("Preset must be one of: smoke, local, cloud")
    if impl not in {"auto", "jax", "warp"}:
        raise ValueError("Implementation must be one of: auto, jax, warp")
    if seed < 0:
        raise ValueError("Training seed must be non-negative.")
    remote_root = _resolve_remote_dir(config, remote_dir)
    relative_output = output or (
        f"runs/{timestamp()}-{slugify(env_name)}-{slugify(preset_name)}"
    )
    remote_output = _resolve_remote_output(remote_root, relative_output)

    train = [
        f"{remote_root}/.venv/bin/simrig",
        "train",
        env_name,
        "--preset",
        preset_name,
        "--output",
        remote_output,
        "--impl",
        impl,
        "--seed",
        str(seed),
    ]
    if not domain_randomization:
        train.append("--no-domain-randomization")
    for flag, value in (
        ("--timesteps", timesteps),
        ("--num-envs", num_envs),
        ("--batch-size", batch_size),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"{flag} must be positive")
            train.extend([flag, str(value)])

    prefix = _shell_chain(
        [
            ["cd", remote_root],
            ["test", "-x", ".venv/bin/simrig"],
            ["nvidia-smi", "-L"],
            ["mkdir", "-p", remote_output],
        ]
    )
    if detach:
        log_path = f"{remote_output}/train.log"
        pid_path = f"{remote_output}/train.pid"
        remote = (
            f"{prefix} && {{ nohup {_shell_command(train)} > {shlex.quote(log_path)} 2>&1 "
            f"< /dev/null & pid=$!; printf '%s\\n' \"$pid\" > {shlex.quote(pid_path)}; "
            f"printf 'started pid=%s output=%s\\n' \"$pid\" {shlex.quote(remote_output)}; }}"
        )
    else:
        remote = f"{prefix} && {_shell_command(train)}"

    returncode = subprocess.run(
        ssh_command(config, remote, batch=True),
        check=False,
    ).returncode
    return LambdaTrainResult(
        output_dir=remote_output,
        detached=detach,
        returncode=returncode,
    )


def smoke_lambda(
    config: LambdaSSHConfig,
    env_name: str,
    *,
    remote_dir: str | None = None,
    steps: int = 10,
) -> int:
    """Run the environment reset/step smoke gate on the remote GPU instance."""
    _check_local_requirements(config, commands=("ssh",))
    if steps <= 0:
        raise ValueError("Smoke steps must be positive.")
    remote_root = _resolve_remote_dir(config, remote_dir)
    remote = _shell_chain(
        [
            ["cd", remote_root],
            ["test", "-x", ".venv/bin/simrig"],
            ["nvidia-smi", "-L"],
            [".venv/bin/simrig", "smoke", env_name, "--steps", str(steps)],
        ]
    )
    return subprocess.run(ssh_command(config, remote, batch=True), check=False).returncode


def status_lambda(
    config: LambdaSSHConfig,
    output: str,
    *,
    remote_dir: str | None = None,
    lines: int = 30,
) -> int:
    """Report a detached run's process state and recent log output."""
    _check_local_requirements(config, commands=("ssh",))
    if lines <= 0:
        raise ValueError("Log line count must be positive.")
    remote_root = _resolve_remote_dir(config, remote_dir)
    remote_output = _resolve_remote_output(remote_root, output)
    pid_path = shlex.quote(f"{remote_output}/train.pid")
    log_path = shlex.quote(f"{remote_output}/train.log")
    policy_path = shlex.quote(f"{remote_output}/policy.params")
    metrics_path = shlex.quote(f"{remote_output}/final_metrics.json")
    checkpoint_path = shlex.quote(f"{remote_output}/checkpoints")
    remote = (
        f"if test -f {pid_path}; then pid=$(cat {pid_path}); else pid=unknown; fi; "
        f"if test -f {policy_path} && test -f {metrics_path} "
        f"&& test -n \"$(find {checkpoint_path} -type f -print -quit 2>/dev/null)\"; "
        "then artifacts=complete; else artifacts=incomplete; fi; "
        "if test \"$pid\" != unknown && kill -0 \"$pid\" 2>/dev/null; "
        "then state=running; elif test \"$artifacts\" = complete; "
        "then state=completed; else state=stopped; fi; "
        "printf 'status=%s pid=%s artifacts=%s\\n' \"$state\" \"$pid\" \"$artifacts\"; "
        f"if test -f {log_path}; then tail -n {lines} {log_path}; fi"
    )
    return subprocess.run(ssh_command(config, remote, batch=True), check=False).returncode


def fetch_lambda(
    config: LambdaSSHConfig,
    output: str,
    *,
    remote_dir: str | None = None,
    local_output: Path | str | None = None,
) -> Path:
    """Download one remote run directory with rsync."""
    _check_local_requirements(config, commands=("ssh", "rsync"))
    remote_root = _resolve_remote_dir(config, remote_dir)
    remote_output = _resolve_remote_output(remote_root, output)
    destination = (
        Path(local_output).expanduser()
        if local_output is not None
        else Path("runs") / PurePosixPath(remote_output).name
    )
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    transport = ssh_command(config, batch=True)
    transport.pop()
    command = [
        "rsync",
        "-az",
        "--progress",
        "-e",
        shlex.join(transport),
        f"{config.target}:{remote_output}/",
        f"{destination}{os.sep}",
    ]
    subprocess.run(command, check=True)
    return destination


def _check_local_requirements(config: LambdaSSHConfig, *, commands: Sequence[str]) -> None:
    for command in commands:
        if shutil.which(command) is None:
            raise RuntimeError(f"Required local command is not installed: {command}")
    if config.identity is not None:
        identity = config.identity.expanduser().resolve()
        if not identity.is_file():
            raise FileNotFoundError(f"SSH private key not found: {identity}")
        if identity.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                f"SSH private key permissions are too open: {identity}. "
                f"Run: chmod 600 {shlex.quote(str(identity))}"
            )
        ssh_keygen = shutil.which("ssh-keygen")
        if ssh_keygen is None:
            raise RuntimeError("Required local command is not installed: ssh-keygen")
        validation = subprocess.run(
            [ssh_keygen, "-lf", str(identity)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if validation.returncode != 0:
            detail = validation.stderr.strip() or "ssh-keygen could not parse the file"
            raise ValueError(f"Invalid SSH identity file {identity}: {detail}")


def _resolve_remote_dir(config: LambdaSSHConfig, remote_dir: str | None) -> str:
    value = remote_dir or f"/home/{config.user}/simrig"
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Remote directory must be an absolute path without '..'.")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", str(path)):
        raise ValueError("Remote directory contains unsupported characters.")
    return str(path)


def _resolve_remote_output(remote_dir: str, output: str) -> str:
    path = PurePosixPath(output)
    if not re.fullmatch(r"/?[A-Za-z0-9._/-]+", str(path)):
        raise ValueError("Remote output contains unsupported characters.")
    if path.is_absolute():
        remote = path
    else:
        if ".." in path.parts:
            raise ValueError("Remote output must not contain '..'.")
        remote = PurePosixPath(remote_dir) / path
    root = PurePosixPath(remote_dir)
    if remote == root or remote.parts[: len(root.parts)] != root.parts:
        raise ValueError("Remote output must be inside the remote project directory.")
    return str(remote)


def _shell_command(parts: Sequence[str]) -> str:
    return shlex.join([str(part) for part in parts])


def _shell_chain(commands: Sequence[Sequence[str]]) -> str:
    return " && ".join(_shell_command(command) for command in commands)
