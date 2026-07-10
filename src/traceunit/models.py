from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Decision(StrEnum):
    REJECT = "reject"
    PARTIAL_ARCHIVE = "partial_archive"
    ESCALATE = "escalate"
    PROMOTE = "promote"
    TEST_CHALLENGE = "test_challenge"


class TestTier(StrEnum):
    PUBLIC = "public"
    HIDDEN = "hidden"
    BRIDGE = "bridge"
    REGRESSION = "regression"
    ADMISSION = "admission"


class TestStatus(StrEnum):
    PROPOSED = "proposed"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    CHALLENGED = "challenged"
    RETIRED = "retired"


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    kind: str
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceEvent":
        return cls(
            event_id=str(value.get("event_id") or value.get("id") or ""),
            kind=str(value.get("kind") or "event"),
            input=value.get("input"),
            output=value.get("output"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TraceRun:
    trace_id: str
    benchmark: str
    split: str
    candidate_id: str
    task_id: str
    score: float
    passed: bool
    status: str = "ok"
    input_summary: str = ""
    output_summary: str = ""
    events: tuple[TraceEvent, ...] = ()
    artifact_paths: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceRun":
        return cls(
            trace_id=str(value["trace_id"]),
            benchmark=str(value["benchmark"]),
            split=str(value["split"]),
            candidate_id=str(value["candidate_id"]),
            task_id=str(value["task_id"]),
            score=float(value.get("score") or 0.0),
            passed=bool(value.get("passed")),
            status=str(value.get("status") or "ok"),
            input_summary=str(value.get("input_summary") or ""),
            output_summary=str(value.get("output_summary") or ""),
            events=tuple(
                TraceEvent.from_dict(item)
                for item in value.get("events") or []
                if isinstance(item, Mapping)
            ),
            artifact_paths=tuple(
                str(item) for item in value.get("artifact_paths") or []
            ),
            metrics=dict(value.get("metrics") or {}),
        )


@dataclass(frozen=True)
class FailureHypothesis:
    hypothesis_id: str
    mechanism: str
    target_boundary: str
    claim: str
    evidence_trace_ids: tuple[str, ...]
    alternatives: tuple[str, ...] = ()
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureHypothesis":
        return cls(
            hypothesis_id=str(value["hypothesis_id"]),
            mechanism=str(value.get("mechanism") or "unknown"),
            target_boundary=str(value.get("target_boundary") or "behavior"),
            claim=str(value.get("claim") or ""),
            evidence_trace_ids=tuple(
                str(x) for x in value.get("evidence_trace_ids") or []
            ),
            alternatives=tuple(str(x) for x in value.get("alternatives") or []),
            confidence=float(value.get("confidence", 0.5)),
        )


@dataclass(frozen=True)
class TestCaseSpec:
    case_id: str
    family_id: str
    tier: TestTier
    path: str
    driver: str = "python"
    arguments: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 60
    expected_incumbent_pass: bool = False
    expected_candidate_pass: bool = True
    description: str = ""
    admission_role: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TestCaseSpec":
        return cls(
            case_id=str(value["case_id"]),
            family_id=str(value.get("family_id") or value["case_id"]),
            tier=TestTier(str(value["tier"])),
            path=str(value["path"]),
            driver=str(value.get("driver") or "python"),
            arguments=tuple(str(x) for x in value.get("arguments") or []),
            environment={
                str(k): str(v) for k, v in dict(value.get("environment") or {}).items()
            },
            timeout_s=max(1, int(value.get("timeout_s") or 60)),
            expected_incumbent_pass=bool(value.get("expected_incumbent_pass", False)),
            expected_candidate_pass=bool(value.get("expected_candidate_pass", True)),
            description=str(value.get("description") or ""),
            admission_role=str(value.get("admission_role") or ""),
        )


@dataclass(frozen=True)
class TestPacket:
    packet_id: str
    version: int
    source_trace_ids: tuple[str, ...]
    hypotheses: tuple[FailureHypothesis, ...]
    target_hypothesis_id: str
    public_contract: str
    hidden_variant_strategy: str
    cases: tuple[TestCaseSpec, ...]
    status: TestStatus = TestStatus.PROPOSED
    admission_score: float = 0.0
    content_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        for case in value["cases"]:
            case["tier"] = str(case["tier"])
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TestPacket":
        return cls(
            packet_id=str(value["packet_id"]),
            version=int(value.get("version") or 1),
            source_trace_ids=tuple(str(x) for x in value.get("source_trace_ids") or []),
            hypotheses=tuple(
                FailureHypothesis.from_dict(item)
                for item in value.get("hypotheses") or []
                if isinstance(item, Mapping)
            ),
            target_hypothesis_id=str(value["target_hypothesis_id"]),
            public_contract=str(value.get("public_contract") or ""),
            hidden_variant_strategy=str(value.get("hidden_variant_strategy") or ""),
            cases=tuple(
                TestCaseSpec.from_dict(item)
                for item in value.get("cases") or []
                if isinstance(item, Mapping)
            ),
            status=TestStatus(str(value.get("status") or TestStatus.PROPOSED)),
            admission_score=float(value.get("admission_score") or 0.0),
            content_sha256=str(value.get("content_sha256") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TestExecution:
    case_id: str
    family_id: str
    tier: TestTier
    subject: str
    passed: bool
    returncode: int | None
    duration_s: float
    stdout_path: str
    stderr_path: str
    timed_out: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tier"] = self.tier.value
        return value


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    score: float
    passed: bool
    trace_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskOutcome":
        return cls(
            task_id=str(value["task_id"]),
            score=float(value.get("score") or 0.0),
            passed=bool(value.get("passed")),
            trace_id=str(value.get("trace_id") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class BenchmarkEvaluation:
    evaluation_id: str
    benchmark: str
    candidate_id: str
    split: str
    score: float
    passrate: float
    cost: float
    outcomes: tuple[TaskOutcome, ...]
    trace_path: str
    result_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkEvaluation":
        return cls(
            evaluation_id=str(value["evaluation_id"]),
            benchmark=str(value["benchmark"]),
            candidate_id=str(value["candidate_id"]),
            split=str(value["split"]),
            score=float(value.get("score") or 0.0),
            passrate=float(value.get("passrate") or 0.0),
            cost=float(value.get("cost") or 0.0),
            outcomes=tuple(
                TaskOutcome.from_dict(item)
                for item in value.get("outcomes") or []
                if isinstance(item, Mapping)
            ),
            trace_path=str(value.get("trace_path") or ""),
            result_path=str(value.get("result_path") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CandidateProposal:
    candidate_id: str
    parent_id: str
    hypothesis_id: str
    mechanism_claim: str
    predicted_effect: str
    regression_risks: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateProposal":
        return cls(
            candidate_id=str(value["candidate_id"]),
            parent_id=str(value["parent_id"]),
            hypothesis_id=str(value["hypothesis_id"]),
            mechanism_claim=str(value.get("mechanism_claim") or ""),
            predicted_effect=str(value.get("predicted_effect") or ""),
            regression_risks=tuple(str(x) for x in value.get("regression_risks") or []),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    iteration: int
    candidate_id: str
    packet_id: str
    public_gain: float
    hidden_gain: float
    bridge_gain: float
    regression_loss: float
    admission_score: float
    diagnostic_delta: float | None = None
    canary_delta: float | None = None
    audit_delta: float | None = None
    total_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        def optional_float(name: str) -> float | None:
            raw = value.get(name)
            return None if raw is None else float(raw)

        return cls(
            iteration=int(value["iteration"]),
            candidate_id=str(value["candidate_id"]),
            packet_id=str(value["packet_id"]),
            public_gain=float(value.get("public_gain") or 0.0),
            hidden_gain=float(value.get("hidden_gain") or 0.0),
            bridge_gain=float(value.get("bridge_gain") or 0.0),
            regression_loss=float(value.get("regression_loss") or 0.0),
            admission_score=float(value.get("admission_score") or 0.0),
            diagnostic_delta=optional_float("diagnostic_delta"),
            canary_delta=optional_float("canary_delta"),
            audit_delta=optional_float("audit_delta"),
            total_cost=float(value.get("total_cost") or 0.0),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class DecisionRecord:
    iteration: int
    candidate_id: str
    decision: Decision
    reason: str
    confidence: float
    evidence: EvidenceRecord

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionRecord":
        return cls(
            iteration=int(value["iteration"]),
            candidate_id=str(value["candidate_id"]),
            decision=Decision(str(value["decision"])),
            reason=str(value.get("reason") or ""),
            confidence=float(value.get("confidence") or 0.0),
            evidence=EvidenceRecord.from_dict(dict(value.get("evidence") or {})),
        )


@dataclass
class RunState:
    run_id: str
    benchmark: str
    status: str
    next_iteration: int
    incumbent_id: str
    incumbent_source: str
    incumbent_diagnostic_score: float
    incumbent_canary_score: float | None = None
    promoted_ids: list[str] = field(default_factory=list)
    partial_archive_ids: list[str] = field(default_factory=list)
    challenged_packet_ids: list[str] = field(default_factory=list)
    active_packet_id: str = ""
    active_packet_path: str = ""
    active_packet_uses: int = 0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunState":
        return cls(
            run_id=str(value["run_id"]),
            benchmark=str(value["benchmark"]),
            status=str(value.get("status") or "running"),
            next_iteration=int(value.get("next_iteration") or 1),
            incumbent_id=str(value["incumbent_id"]),
            incumbent_source=str(value["incumbent_source"]),
            incumbent_diagnostic_score=float(
                value.get("incumbent_diagnostic_score") or 0.0
            ),
            incumbent_canary_score=(
                None
                if value.get("incumbent_canary_score") is None
                else float(value["incumbent_canary_score"])
            ),
            promoted_ids=[str(x) for x in value.get("promoted_ids") or []],
            partial_archive_ids=[
                str(x) for x in value.get("partial_archive_ids") or []
            ],
            challenged_packet_ids=[
                str(x) for x in value.get("challenged_packet_ids") or []
            ],
            active_packet_id=str(value.get("active_packet_id") or ""),
            active_packet_path=str(value.get("active_packet_path") or ""),
            active_packet_uses=int(value.get("active_packet_uses") or 0),
            total_cost=float(value.get("total_cost") or 0.0),
        )
