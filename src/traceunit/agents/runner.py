from __future__ import annotations

import os
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from traceunit.config import AgentConfig
from traceunit.io import expand_placeholders


@dataclass(frozen=True)
class AgentRunResult:
    role: str
    returncode: int
    duration_s: float
    stdout_path: str
    stderr_path: str
    final_message_path: str
    timed_out: bool = False


class WorkspaceAgent(Protocol):
    def preflight(self) -> None: ...

    def run(
        self, *, role: str, prompt: str, workspace: Path, log_dir: Path
    ) -> AgentRunResult: ...


_DEFAULT_IMAGES = {
    "codex": "node:20-slim",
    "claude": "docker-claude:latest",
}


def _resolve_image(config: AgentConfig) -> str:
    provider = config.provider.strip().lower()
    image = config.container_image or _DEFAULT_IMAGES.get(provider, "")
    if not image:
        raise RuntimeError(f"no container image configured for provider {provider!r}")
    return image


class CommandWorkspaceAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def preflight(self) -> None:
        if not self.config.enabled or self.config.isolation != "docker":
            return
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("agent isolation=docker requires Docker")
        provider = self.config.provider.strip().lower()
        image = _resolve_image(self.config)
        try:
            inspected = subprocess.run(
                [docker, "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"cannot inspect role-agent image {image!r}") from exc
        if inspected.returncode != 0:
            raise RuntimeError(f"role-agent image is not cached: {image!r}")
        if provider == "codex":
            if not shutil.which("codex"):
                raise RuntimeError(
                    "containerized Codex needs a host Codex installation"
                )
            if not (Path.home() / ".codex/auth.json").is_file():
                raise RuntimeError("containerized Codex requires ~/.codex/auth.json")

    def _argv(
        self,
        *,
        workspace: Path,
        prompt_path: Path,
        final_message_path: Path,
        containerized: bool = False,
    ) -> list[str]:
        values = {
            "workspace": str(workspace.resolve()),
            "prompt_file": str(prompt_path.resolve()),
            "output_file": str(final_message_path.resolve()),
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
        }
        if self.config.command:
            return [expand_placeholders(item, values) for item in self.config.command]

        provider = self.config.provider.strip().lower()
        if provider == "codex":
            argv = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "--cd",
                str(workspace.resolve()),
                "--output-last-message",
                str(final_message_path.resolve()),
            ]
            if containerized:
                argv.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                argv.extend(["--sandbox", "workspace-write"])
            if self.config.model:
                argv.extend(["--model", self.config.model])
            if self.config.reasoning_effort:
                argv.extend(
                    [
                        "--config",
                        f'model_reasoning_effort="{self.config.reasoning_effort}"',
                    ]
                )
            argv.append("-")
            return argv
        if provider == "claude":
            argv = [
                "claude",
                "--print",
                "--permission-mode",
                "acceptEdits",
                "--no-session-persistence",
                "--output-format",
                "text",
            ]
            if self.config.model:
                argv.extend(["--model", self.config.model])
            if self.config.reasoning_effort:
                argv.extend(["--effort", self.config.reasoning_effort])
            return argv
        raise ValueError(f"unsupported agent provider: {self.config.provider!r}")

    def run(
        self, *, role: str, prompt: str, workspace: Path, log_dir: Path
    ) -> AgentRunResult:
        if not self.config.enabled:
            raise RuntimeError(f"agent role {role} is disabled")
        workspace.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = log_dir / "prompt.md"
        stdout_path = log_dir / "stdout.txt"
        stderr_path = log_dir / "stderr.txt"
        final_message_path = log_dir / "final_message.txt"
        containerized = self.config.isolation == "docker"
        effective_prompt = (
            prompt.replace(str(workspace.resolve()), "/workspace")
            if containerized
            else prompt
        )
        prompt_path.write_text(effective_prompt, encoding="utf-8")
        if containerized:
            inner_argv = self._argv(
                workspace=Path("/workspace"),
                prompt_path=Path("/logs/prompt.md"),
                final_message_path=Path("/logs/final_message.txt"),
                containerized=True,
            )
            agent_home = Path(tempfile.mkdtemp(prefix=".agent-home-", dir=str(log_dir)))
            _stage_agent_credentials(self.config.provider, agent_home)
            argv, env, cidfile = self._docker_argv(
                inner_argv=inner_argv,
                workspace=workspace,
                log_dir=log_dir,
                agent_home=agent_home,
            )
        else:
            if self.config.isolation == "external" and not self.config.command:
                raise RuntimeError(
                    "agent isolation=external requires an explicit command"
                )
            argv = self._argv(
                workspace=workspace,
                prompt_path=prompt_path,
                final_message_path=final_message_path,
            )
            env = os.environ.copy()
            env.update(self.config.environment)
            cidfile = None
            agent_home = None
        started = time.monotonic()
        timed_out = False
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            proc = subprocess.Popen(
                argv,
                cwd=workspace,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
                env=env,
            )
            try:
                proc.communicate(effective_prompt, timeout=self.config.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                if cidfile is not None:
                    _force_remove_container(cidfile, env)
                _terminate_process_group(proc)
                proc.communicate()
        if cidfile is not None:
            cidfile.unlink(missing_ok=True)
        if agent_home is not None:
            shutil.rmtree(agent_home, ignore_errors=True)
        if not final_message_path.exists():
            final_message_path.write_text("", encoding="utf-8")
        return AgentRunResult(
            role=role,
            returncode=int(proc.returncode or 0),
            duration_s=time.monotonic() - started,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            final_message_path=str(final_message_path),
            timed_out=timed_out,
        )

    def _docker_argv(
        self,
        *,
        inner_argv: list[str],
        workspace: Path,
        log_dir: Path,
        agent_home: Path,
    ) -> tuple[list[str], dict[str, str], Path]:
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("agent isolation=docker requires Docker")
        provider = self.config.provider.strip().lower()
        image = _resolve_image(self.config)
        cidfile = log_dir / "container.cid"
        cidfile.unlink(missing_ok=True)
        outer_env = {
            key: os.environ[key]
            for key in ("DOCKER_HOST", "LANG", "LC_ALL", "PATH")
            if key in os.environ
        }
        outer_env.setdefault("PATH", "/usr/bin:/bin")
        command = [
            docker,
            "run",
            "--rm",
            "--init",
            "--interactive",
            "--network",
            "bridge",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--memory",
            "6g",
            "--cpus",
            "4",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cidfile",
            str(cidfile.resolve()),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1g",
            "--mount",
            f"type=bind,src={workspace.resolve()},dst=/workspace",
            "--mount",
            f"type=bind,src={log_dir.resolve()},dst=/logs",
            "--mount",
            f"type=bind,src={agent_home.resolve()},dst=/home/agent",
            "--mount",
            "type=bind,src=/etc/ssl/certs,dst=/etc/ssl/certs,readonly",
            "--env",
            "HOME=/home/agent",
            "--env",
            "LANG=C.UTF-8",
            "--env",
            "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
            "--env",
            "NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt",
            "--workdir",
            "/workspace",
        ]
        for key, value in self.config.environment.items():
            outer_env[key] = value
            command.extend(["--env", key])
        if provider == "codex":
            executable = shutil.which("codex")
            if not executable:
                raise RuntimeError(
                    "containerized Codex needs a host Codex installation"
                )
            module_root = Path(executable).resolve().parent.parent
            command.extend(
                [
                    "--mount",
                    f"type=bind,src={module_root},dst=/opt/codex,readonly",
                    "--entrypoint",
                    "node",
                    image,
                    "/opt/codex/bin/codex.js",
                    *inner_argv[1:],
                ]
            )
        else:
            command.extend(
                [
                    "--entrypoint",
                    inner_argv[0],
                    image,
                    *inner_argv[1:],
                ]
            )
        return command, outer_env, cidfile


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=5)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _stage_agent_credentials(provider: str, destination: Path) -> None:
    provider = provider.strip().lower()
    home = Path.home()
    if provider == "codex":
        target = destination / ".codex"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("auth.json", "config.toml"):
            source = home / ".codex" / name
            if source.is_file():
                shutil.copy2(source, target / name)
        if not (target / "auth.json").is_file():
            raise RuntimeError("containerized Codex requires ~/.codex/auth.json")
        return
    if provider == "claude":
        source_dir = home / ".claude"
        if source_dir.is_dir():
            shutil.copytree(
                source_dir,
                destination / ".claude",
                ignore=shutil.ignore_patterns(
                    "debug",
                    "history.jsonl",
                    "projects",
                    "session-env",
                    "shell-snapshots",
                ),
            )
        settings = home / ".claude.json"
        if settings.is_file():
            shutil.copy2(settings, destination / ".claude.json")


def _force_remove_container(cidfile: Path, env: dict[str, str]) -> None:
    try:
        container_id = cidfile.read_text(encoding="utf-8").strip()
    except OSError:
        return
    docker = shutil.which("docker")
    if not docker or not container_id:
        return
    try:
        subprocess.run(
            [docker, "rm", "--force", container_id],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def build_agent(config: AgentConfig) -> WorkspaceAgent:
    return CommandWorkspaceAgent(config)
