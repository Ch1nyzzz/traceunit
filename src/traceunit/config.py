from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "codex"
    model: str = ""
    reasoning_effort: str = "high"
    timeout_s: int = 1800
    command: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    isolation: str = "docker"
    container_image: str = ""


@dataclass(frozen=True)
class AgentsConfig:
    experimentalist: AgentConfig = field(default_factory=AgentConfig)
    optimizer: AgentConfig = field(default_factory=AgentConfig)
    auditor: AgentConfig = field(default_factory=lambda: AgentConfig(enabled=False))


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    worldcalib_root: Path = Path("../WorldCalib")
    env_file: Path | None = None
    seed_source_path: Path | None = None
    diagnostic_data_path: Path | None = None
    canary_data_path: Path | None = None
    audit_data_path: Path | None = None
    split_manifest_path: Path | None = None
    diagnostic_split: str = "train"
    canary_split: str = "test"
    audit_split: str = "test"
    diagnostic_limit: int = 0
    canary_limit: int = 0
    audit_limit: int = 0
    dry_run: bool = False
    force: bool = False
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    concurrency: int = 1
    repeats: int = 1
    timeout_s: int = 900
    max_interactions: int = 100
    agent_command: str = ""
    evaluator_command: str = ""
    unit_python: Path | None = None
    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    dataset_split: str = "test"
    split_seed: int = 1729
    diagnostic_fraction: float = 0.6
    canary_fraction: float = 0.2


@dataclass(frozen=True)
class DecisionConfig:
    min_admission_score: float = 0.6
    min_public_gain: float = 0.5
    min_hidden_gain: float = 0.5
    min_bridge_gain: float = 0.5
    max_regression_loss: float = 0.0
    min_diagnostic_delta: float = 0.0
    min_canary_delta: float = 0.0
    min_audit_delta: float = 0.0
    noninferiority_margin: float = 0.0
    rejected_probe_rate: float = 0.1
    require_audit_for_promotion: bool = False


@dataclass(frozen=True)
class LoopConfig:
    run_dir: Path
    run_id: str = ""
    iterations: int = 5
    seed: int = 0
    resume: bool = True
    max_failure_traces: int = 8
    max_candidates_per_packet: int = 2
    max_trace_chars_per_artifact: int = 100_000
    retain_agent_logs: bool = True
    posthoc_audit: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    loop: LoopConfig
    benchmark: BenchmarkConfig
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


def _agent(value: Mapping[str, Any] | None, default: AgentConfig) -> AgentConfig:
    raw = dict(value or {})
    return AgentConfig(
        provider=str(raw.get("provider", default.provider)),
        model=str(raw.get("model", default.model)),
        reasoning_effort=str(raw.get("reasoning_effort", default.reasoning_effort)),
        timeout_s=int(raw.get("timeout_s", default.timeout_s)),
        command=tuple(str(x) for x in raw.get("command") or default.command),
        environment={
            str(k): str(v) for k, v in dict(raw.get("environment") or {}).items()
        },
        enabled=bool(raw.get("enabled", default.enabled)),
        isolation=str(raw.get("isolation", default.isolation)).lower(),
        container_image=str(raw.get("container_image", default.container_image)),
    )


def _path(base: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> ProjectConfig:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    base = path.parent
    loop_raw = dict(raw.get("loop") or {})
    bench_raw = dict(raw.get("benchmark") or {})
    if "name" not in bench_raw:
        raise ValueError("benchmark.name is required")
    if "run_dir" not in loop_raw:
        raise ValueError("loop.run_dir is required")

    loop = LoopConfig(
        run_dir=_path(base, loop_raw["run_dir"]) or base / "runs/default",
        run_id=str(loop_raw.get("run_id") or ""),
        iterations=max(0, int(loop_raw.get("iterations", 5))),
        seed=int(loop_raw.get("seed", 0)),
        resume=bool(loop_raw.get("resume", True)),
        max_failure_traces=max(1, int(loop_raw.get("max_failure_traces", 8))),
        max_candidates_per_packet=max(
            1, int(loop_raw.get("max_candidates_per_packet", 2))
        ),
        max_trace_chars_per_artifact=max(
            1000, int(loop_raw.get("max_trace_chars_per_artifact", 100_000))
        ),
        retain_agent_logs=bool(loop_raw.get("retain_agent_logs", True)),
        posthoc_audit=bool(loop_raw.get("posthoc_audit", True)),
    )
    worldcalib_root = (
        _path(base, bench_raw.get("worldcalib_root", "../WorldCalib"))
        or (base / "../WorldCalib").resolve()
    )
    env_file = _path(base, bench_raw.get("env_file")) or worldcalib_root / ".env"
    benchmark = BenchmarkConfig(
        name=str(bench_raw["name"]).lower(),
        worldcalib_root=worldcalib_root,
        env_file=env_file,
        seed_source_path=_path(base, bench_raw.get("seed_source_path")),
        diagnostic_data_path=_path(base, bench_raw.get("diagnostic_data_path")),
        canary_data_path=_path(base, bench_raw.get("canary_data_path")),
        audit_data_path=_path(base, bench_raw.get("audit_data_path")),
        split_manifest_path=_path(base, bench_raw.get("split_manifest_path")),
        diagnostic_split=str(bench_raw.get("diagnostic_split", "train")),
        canary_split=str(bench_raw.get("canary_split", "test")),
        audit_split=str(bench_raw.get("audit_split", "test")),
        diagnostic_limit=int(bench_raw.get("diagnostic_limit", 0)),
        canary_limit=int(bench_raw.get("canary_limit", 0)),
        audit_limit=int(bench_raw.get("audit_limit", 0)),
        dry_run=bool(bench_raw.get("dry_run", False)),
        force=bool(bench_raw.get("force", False)),
        model=str(bench_raw.get("model", "deepseek-v4-flash")),
        base_url=str(bench_raw.get("base_url", "https://api.deepseek.com/v1")),
        api_key_env=str(bench_raw.get("api_key_env", "DEEPSEEK_API_KEY")),
        concurrency=max(1, int(bench_raw.get("concurrency", 1))),
        repeats=max(1, int(bench_raw.get("repeats", 1))),
        timeout_s=max(1, int(bench_raw.get("timeout_s", 900))),
        max_interactions=max(1, int(bench_raw.get("max_interactions", 100))),
        agent_command=str(bench_raw.get("agent_command") or ""),
        evaluator_command=str(bench_raw.get("evaluator_command") or ""),
        unit_python=_path(base, bench_raw.get("unit_python")),
        dataset_name=str(
            bench_raw.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
        ),
        dataset_split=str(bench_raw.get("dataset_split", "test")),
        split_seed=int(bench_raw.get("split_seed", 1729)),
        diagnostic_fraction=float(bench_raw.get("diagnostic_fraction", 0.6)),
        canary_fraction=float(bench_raw.get("canary_fraction", 0.2)),
    )
    if benchmark.name not in {"swebench_verified", "appworld"}:
        raise ValueError("benchmark.name must be swebench_verified or appworld")
    if benchmark.api_key_env not in os.environ and env_file.is_file():
        value = dotenv_values(env_file).get(benchmark.api_key_env)
        if value:
            os.environ[benchmark.api_key_env] = str(value)

    agents_raw = dict(raw.get("agents") or {})
    default_agent = AgentConfig()
    agents = AgentsConfig(
        experimentalist=_agent(agents_raw.get("experimentalist"), default_agent),
        optimizer=_agent(agents_raw.get("optimizer"), default_agent),
        auditor=_agent(agents_raw.get("auditor"), AgentConfig(enabled=False)),
    )
    for role, agent in (
        ("experimentalist", agents.experimentalist),
        ("optimizer", agents.optimizer),
        ("auditor", agents.auditor),
    ):
        if agent.isolation not in {"docker", "none", "external"}:
            raise ValueError(
                f"agents.{role}.isolation must be docker, external, or none"
            )
    decision_raw = dict(raw.get("decision") or {})
    defaults = DecisionConfig()
    decision = DecisionConfig(
        **{
            name: decision_raw.get(name, getattr(defaults, name))
            for name in defaults.__dataclass_fields__
        }
    )
    return ProjectConfig(
        loop=loop, benchmark=benchmark, agents=agents, decision=decision
    )
