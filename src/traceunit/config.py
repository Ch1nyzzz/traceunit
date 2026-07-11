from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
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
    test_author: AgentConfig = field(default_factory=AgentConfig)
    search: AgentConfig = field(default_factory=AgentConfig)
    regression_author: AgentConfig = field(
        default_factory=lambda: AgentConfig(enabled=False)
    )
    ut_critic: AgentConfig = field(default_factory=lambda: AgentConfig(enabled=False))


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    worldcalib_root: Path = Path("../WorldCalib")
    env_file: Path | None = None
    baseline_source_path: Path | None = None
    search_data_path: Path | None = None
    final_data_path: Path | None = None
    split_manifest_path: Path | None = None
    search_split: str = "train"
    heldout_split: str = "test"
    search_limit: int = 0
    final_limit: int = 0
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
    benchmark_seed: int = 1729
    search_fraction: float = 0.6


@dataclass(frozen=True)
class DecisionConfig:
    max_regression_loss: float = 0.0
    min_search_delta: float = 0.0
    noninferiority_margin: float = 0.0


class ExperimentCondition(StrEnum):
    SCORE_ONLY = "c0_score_only"
    RAW_TRACEUNIT = "c1_raw_traceunit"
    ARCHIVE = "c2_archive"
    FULL = "c3_full"


@dataclass(frozen=True)
class ConditionCapabilities:
    generated_packets: bool
    unit_gate: bool
    partial_archive: bool
    online_ut_memory: bool


@dataclass(frozen=True)
class ProtocolConfig:
    condition: ExperimentCondition = ExperimentCondition.FULL


@dataclass(frozen=True)
class MemoryConfig:
    """Bound the small, online UT-design memory exposed to later authors."""

    max_world_model_lessons: int = 64


@dataclass(frozen=True)
class LoopConfig:
    run_dir: Path
    run_id: str = ""
    iterations: int = 5
    resume: bool = True
    max_failure_traces: int = 8
    max_attempts_per_packet: int = 4
    max_trace_chars_per_artifact: int = 100_000
    retain_agent_logs: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    loop: LoopConfig
    benchmark: BenchmarkConfig
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @property
    def capabilities(self) -> ConditionCapabilities:
        condition = self.protocol.condition
        return ConditionCapabilities(
            generated_packets=condition is not ExperimentCondition.SCORE_ONLY,
            unit_gate=condition is not ExperimentCondition.SCORE_ONLY,
            partial_archive=condition
            in {ExperimentCondition.ARCHIVE, ExperimentCondition.FULL},
            online_ut_memory=condition is ExperimentCondition.FULL,
        )


def _path(base: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown {section} configuration keys: {joined}")


def _agent(value: Mapping[str, Any] | None, default: AgentConfig) -> AgentConfig:
    raw = dict(value or {})
    _reject_unknown(raw, set(AgentConfig.__dataclass_fields__), "agent")
    return AgentConfig(
        provider=str(raw.get("provider", default.provider)),
        model=str(raw.get("model", default.model)),
        reasoning_effort=str(raw.get("reasoning_effort", default.reasoning_effort)),
        timeout_s=max(1, int(raw.get("timeout_s", default.timeout_s))),
        command=tuple(str(item) for item in raw.get("command") or default.command),
        environment={
            str(key): str(item)
            for key, item in dict(raw.get("environment") or {}).items()
        },
        enabled=bool(raw.get("enabled", default.enabled)),
        isolation=str(raw.get("isolation", default.isolation)).lower(),
        container_image=str(raw.get("container_image", default.container_image)),
    )


def _section(raw: Mapping[str, Any], cls: type[Any], name: str) -> dict[str, Any]:
    values = dict(raw or {})
    _reject_unknown(values, set(cls.__dataclass_fields__), name)
    return values


def load_config(path: Path) -> ProjectConfig:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    _reject_unknown(
        raw,
        {
            "loop",
            "benchmark",
            "protocol",
            "agents",
            "decision",
            "memory",
        },
        "root",
    )
    base = path.parent

    loop_raw = _section(dict(raw.get("loop") or {}), LoopConfig, "loop")
    benchmark_raw = _section(
        dict(raw.get("benchmark") or {}), BenchmarkConfig, "benchmark"
    )
    if "run_dir" not in loop_raw:
        raise ValueError("loop.run_dir is required")
    if "name" not in benchmark_raw:
        raise ValueError("benchmark.name is required")

    loop = LoopConfig(
        run_dir=_path(base, loop_raw["run_dir"]) or base / "runs/default",
        run_id=str(loop_raw.get("run_id") or ""),
        iterations=max(0, int(loop_raw.get("iterations", 5))),
        resume=bool(loop_raw.get("resume", True)),
        max_failure_traces=max(1, int(loop_raw.get("max_failure_traces", 8))),
        max_attempts_per_packet=max(1, int(loop_raw.get("max_attempts_per_packet", 4))),
        max_trace_chars_per_artifact=max(
            1_000, int(loop_raw.get("max_trace_chars_per_artifact", 100_000))
        ),
        retain_agent_logs=bool(loop_raw.get("retain_agent_logs", True)),
    )

    worldcalib_root = (
        _path(base, benchmark_raw.get("worldcalib_root", "../WorldCalib"))
        or (base / "../WorldCalib").resolve()
    )
    env_file = _path(base, benchmark_raw.get("env_file")) or worldcalib_root / ".env"
    benchmark = BenchmarkConfig(
        name=str(benchmark_raw["name"]).lower(),
        worldcalib_root=worldcalib_root,
        env_file=env_file,
        baseline_source_path=_path(base, benchmark_raw.get("baseline_source_path")),
        search_data_path=_path(base, benchmark_raw.get("search_data_path")),
        final_data_path=_path(base, benchmark_raw.get("final_data_path")),
        split_manifest_path=_path(base, benchmark_raw.get("split_manifest_path")),
        search_split=str(benchmark_raw.get("search_split", "train")),
        heldout_split=str(benchmark_raw.get("heldout_split", "test")),
        search_limit=max(0, int(benchmark_raw.get("search_limit", 0))),
        final_limit=max(0, int(benchmark_raw.get("final_limit", 0))),
        dry_run=bool(benchmark_raw.get("dry_run", False)),
        force=bool(benchmark_raw.get("force", False)),
        model=str(benchmark_raw.get("model", "deepseek-v4-flash")),
        base_url=str(benchmark_raw.get("base_url", "https://api.deepseek.com/v1")),
        api_key_env=str(benchmark_raw.get("api_key_env", "DEEPSEEK_API_KEY")),
        concurrency=max(1, int(benchmark_raw.get("concurrency", 1))),
        repeats=max(1, int(benchmark_raw.get("repeats", 1))),
        timeout_s=max(1, int(benchmark_raw.get("timeout_s", 900))),
        max_interactions=max(1, int(benchmark_raw.get("max_interactions", 100))),
        agent_command=str(benchmark_raw.get("agent_command") or ""),
        evaluator_command=str(benchmark_raw.get("evaluator_command") or ""),
        unit_python=_path(base, benchmark_raw.get("unit_python")),
        dataset_name=str(
            benchmark_raw.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
        ),
        dataset_split=str(benchmark_raw.get("dataset_split", "test")),
        benchmark_seed=int(benchmark_raw.get("benchmark_seed", 1729)),
        search_fraction=float(benchmark_raw.get("search_fraction", 0.6)),
    )
    if benchmark.name not in {"swebench_verified", "appworld"}:
        raise ValueError("benchmark.name must be swebench_verified or appworld")
    if not 0 < benchmark.search_fraction < 1:
        raise ValueError("benchmark.search_fraction must be between 0 and 1")
    if benchmark.api_key_env not in os.environ and env_file.is_file():
        secret = dotenv_values(env_file).get(benchmark.api_key_env)
        if secret:
            os.environ[benchmark.api_key_env] = str(secret)

    agents_raw = dict(raw.get("agents") or {})
    _reject_unknown(
        agents_raw,
        {"test_author", "search", "regression_author", "ut_critic"},
        "agents",
    )
    default_agent = AgentConfig()
    agents = AgentsConfig(
        test_author=_agent(agents_raw.get("test_author"), default_agent),
        search=_agent(agents_raw.get("search"), default_agent),
        regression_author=_agent(
            agents_raw.get("regression_author"), AgentConfig(enabled=False)
        ),
        ut_critic=_agent(agents_raw.get("ut_critic"), AgentConfig(enabled=False)),
    )
    for role, agent in (
        ("test_author", agents.test_author),
        ("search", agents.search),
        ("regression_author", agents.regression_author),
        ("ut_critic", agents.ut_critic),
    ):
        if agent.isolation not in {"docker", "none", "external"}:
            raise ValueError(
                f"agents.{role}.isolation must be docker, external, or none"
            )

    decision_values = _section(
        dict(raw.get("decision") or {}), DecisionConfig, "decision"
    )
    memory_values = _section(dict(raw.get("memory") or {}), MemoryConfig, "memory")
    protocol_values = _section(
        dict(raw.get("protocol") or {}), ProtocolConfig, "protocol"
    )
    try:
        protocol = ProtocolConfig(
            condition=ExperimentCondition(
                str(protocol_values.get("condition", ExperimentCondition.FULL.value))
            )
        )
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ExperimentCondition)
        raise ValueError(f"protocol.condition must be one of: {allowed}") from exc
    decision = DecisionConfig(**decision_values)
    memory = MemoryConfig(**memory_values)
    for name, value in decision.__dict__.items():
        if float(value) < 0:
            raise ValueError(f"decision.{name} must be nonnegative")
    if memory.max_world_model_lessons < 1:
        raise ValueError("memory.max_world_model_lessons must be positive")
    return ProjectConfig(
        loop=loop,
        benchmark=benchmark,
        protocol=protocol,
        agents=agents,
        decision=decision,
        memory=memory,
    )
