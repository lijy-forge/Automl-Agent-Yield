"""Constrained subprocess runner used by YieldMind tool execution."""

from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BASENAMES = {"python", "python3"}
ALLOWED_DOCKER_IMAGES = {"python:3.11-slim", "yieldmind-sandbox:local"}


class SandboxCommand(BaseModel):
    argv: list[str] = Field(..., min_length=1, description="Argument vector. Shell strings are not accepted.")
    cwd: str = Field(default=".", description="Working directory relative to the repository root.")
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("argv")
    @classmethod
    def reject_shell_tokens(cls, value: list[str]) -> list[str]:
        joined = " ".join(value)
        banned = [";", "&&", "||", "|", "`", "$(", ">", "<"]
        if any(token in joined for token in banned):
            raise ValueError("SandboxCommand.argv must not contain shell control operators.")
        return value


class SandboxResult(BaseModel):
    ok: bool
    argv: list[str]
    cwd: str
    returncode: int | None
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str
    mode: str = "isolated_subprocess"


class DockerSandboxCommand(BaseModel):
    argv: list[str] = Field(
        default_factory=lambda: ["python", "-c", "print('yieldmind docker sandbox ok')"],
        min_length=1,
        description="Python argument vector executed inside the container.",
    )
    cwd: str = Field(default=".", description="Read-only working directory relative to the repository root.")
    output_dir: str = Field(
        default="agent_workspace/yieldmind/docker_runs/default",
        description="Writable output directory relative to the repository root.",
    )
    image: str = "python:3.11-slim"
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)
    memory: str = Field(default="512m", pattern=r"^[1-9][0-9]*[mMgG]$")
    cpus: float = Field(default=1.0, gt=0.0, le=8.0)
    pids_limit: int = Field(default=128, ge=16, le=1024)
    allow_docker: bool = False

    @field_validator("argv")
    @classmethod
    def validate_python_argv(cls, value: list[str]) -> list[str]:
        SandboxCommand.reject_shell_tokens(value)
        if Path(value[0]).name not in PYTHON_BASENAMES:
            raise ValueError("Docker sandbox only accepts python/python3 entrypoints.")
        return value

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if value not in ALLOWED_DOCKER_IMAGES:
            allowed = ", ".join(sorted(ALLOWED_DOCKER_IMAGES))
            raise ValueError(f"Docker image is not allowlisted. Allowed images: {allowed}")
        return value


class DockerSandboxResult(BaseModel):
    ok: bool
    mode: str = "docker_no_network"
    authorized: bool
    docker_cli_available: bool
    daemon_available: bool
    image_available: bool
    image: str
    container_name: str = ""
    command: list[str] = Field(default_factory=list)
    returncode: int | None = None
    timed_out: bool = False
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    isolation: dict[str, Any] = Field(default_factory=dict)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_python_executable(raw: str) -> bool:
    path = Path(raw)
    if path.name in PYTHON_BASENAMES:
        return True
    if path.resolve() == Path(sys.executable).resolve():
        return True
    return path.name == "python" and path.exists()


def _minimal_env(extra: dict[str, str]) -> dict[str, str]:
    allow_exact = {"PATH", "PYTHONPATH", "MPLCONFIGDIR", "LC_ALL", "LANG"}
    allow_prefix = ("YIELD_", "AMLA_", "YIELDMIND_")
    env: dict[str, str] = {}
    for key in allow_exact:
        if key in os.environ:
            env[key] = os.environ[key]
    for key, value in os.environ.items():
        if key.startswith(allow_prefix):
            env[key] = value
    for key, value in extra.items():
        if key in allow_exact or key.startswith(allow_prefix):
            env[key] = str(value)
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    return env


def _resolve_repo_path(workspace_root: Path, relative_path: str, *, label: str) -> Path:
    path = (workspace_root / relative_path).resolve()
    if not _is_under(path, workspace_root):
        raise ValueError(f"{label} must stay inside repository root: {path}")
    return path


def docker_isolation_policy(command: DockerSandboxCommand) -> dict[str, Any]:
    return {
        "network": "none",
        "root_filesystem": "read_only",
        "repository_mount": "read_only",
        "output_mount": "read_write",
        "capabilities": "drop_all",
        "no_new_privileges": True,
        "pull_policy": "never",
        "memory": command.memory.lower(),
        "cpus": command.cpus,
        "pids_limit": command.pids_limit,
        "host_environment_forwarded": False,
    }


def build_docker_argv(
    command: DockerSandboxCommand,
    *,
    workspace_root: str | Path,
    docker_executable: str,
    container_name: str,
) -> tuple[list[str], Path]:
    root = Path(workspace_root).resolve()
    cwd = _resolve_repo_path(root, command.cwd, label="Docker sandbox cwd")
    output_dir = _resolve_repo_path(root, command.output_dir, label="Docker sandbox output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    relative_cwd = cwd.relative_to(root)
    container_cwd = Path("/workspace") / relative_cwd
    entrypoint = Path(command.argv[0]).name
    argv = [
        docker_executable,
        "run",
        "--rm",
        "--init",
        "--name",
        container_name,
        "--pull=never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        str(command.pids_limit),
        "--memory",
        command.memory.lower(),
        "--cpus",
        str(command.cpus),
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,src={root},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        "--workdir",
        container_cwd.as_posix(),
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint",
        entrypoint,
    ]
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        argv.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    argv.extend([command.image, *command.argv[1:]])
    return argv, output_dir


class SubprocessSandbox:
    """Run whitelisted Python subprocesses without invoking a shell."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or PROJECT_ROOT).resolve()

    def run(self, command: SandboxCommand) -> SandboxResult:
        cwd = (self.workspace_root / command.cwd).resolve()
        if not _is_under(cwd, self.workspace_root):
            raise ValueError(f"Sandbox cwd must stay inside repository root: {cwd}")
        if not _is_python_executable(command.argv[0]):
            raise ValueError(
                "Sandbox currently allows Python executables only. "
                "Use a Python entrypoint script instead of arbitrary shell commands."
            )

        started = time.time()
        proc = subprocess.Popen(
            command.argv,
            cwd=str(cwd),
            env=_minimal_env(command.env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=command.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                stdout, stderr = proc.communicate()
        duration = time.time() - started
        return SandboxResult(
            ok=(proc.returncode == 0 and not timed_out),
            argv=command.argv,
            cwd=str(cwd),
            returncode=proc.returncode,
            timed_out=timed_out,
            duration_seconds=round(duration, 4),
            stdout=(stdout or "")[-20000:],
            stderr=(stderr or "")[-20000:],
        )


class DockerSandbox:
    """Run explicitly authorized Python commands in a locked-down local container."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or PROJECT_ROOT).resolve()

    def _base_result(self, request: DockerSandboxCommand, **updates: Any) -> DockerSandboxResult:
        payload: dict[str, Any] = {
            "ok": False,
            "authorized": request.allow_docker,
            "docker_cli_available": False,
            "daemon_available": False,
            "image_available": False,
            "image": request.image,
            "isolation": docker_isolation_policy(request),
        }
        payload.update(updates)
        return DockerSandboxResult(**payload)

    def run(self, command: DockerSandboxCommand) -> DockerSandboxResult:
        if not command.allow_docker:
            return self._base_result(
                command,
                error="Docker execution requires allow_docker=true; no container was started.",
            )

        docker_executable = shutil.which("docker")
        if not docker_executable:
            return self._base_result(command, error="Docker CLI is not installed or not on PATH.")

        daemon_check = subprocess.run(
            [docker_executable, "info"],
            capture_output=True,
            text=True,
            timeout=min(command.timeout_seconds, 10.0),
            check=False,
        )
        if daemon_check.returncode != 0:
            return self._base_result(
                command,
                docker_cli_available=True,
                error="Docker daemon is unavailable.",
                stderr=(daemon_check.stderr or daemon_check.stdout)[-20000:],
            )

        image_check = subprocess.run(
            [docker_executable, "image", "inspect", command.image],
            capture_output=True,
            text=True,
            timeout=min(command.timeout_seconds, 10.0),
            check=False,
        )
        if image_check.returncode != 0:
            return self._base_result(
                command,
                docker_cli_available=True,
                daemon_available=True,
                error=(
                    f"Docker image is not available locally: {command.image}. "
                    "Pull/build it explicitly; sandbox execution never pulls images."
                ),
                stderr=(image_check.stderr or image_check.stdout)[-20000:],
            )

        container_name = f"yieldmind-sandbox-{uuid.uuid4().hex[:12]}"
        argv, _ = build_docker_argv(
            command,
            workspace_root=self.workspace_root,
            docker_executable=docker_executable,
            container_name=container_name,
        )
        started = time.time()
        proc = subprocess.Popen(
            argv,
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=command.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(
                [docker_executable, "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            stdout, stderr = proc.communicate(timeout=5)
        duration = time.time() - started
        return self._base_result(
            command,
            ok=proc.returncode == 0 and not timed_out,
            docker_cli_available=True,
            daemon_available=True,
            image_available=True,
            container_name=container_name,
            command=argv,
            returncode=proc.returncode,
            timed_out=timed_out,
            duration_seconds=round(duration, 4),
            stdout=(stdout or "")[-20000:],
            stderr=(stderr or "")[-20000:],
            error="Docker sandbox timed out." if timed_out else ("" if proc.returncode == 0 else "Docker sandbox command failed."),
        )


def result_to_dict(result: SandboxResult) -> dict[str, Any]:
    return result.model_dump()
