from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Decision(StrEnum):
    REJECT = "reject"
    PARTIAL_ELIGIBLE = "partial_eligible"
    ARCHIVE = "archive"
    QUARANTINE = "quarantine"
    EVALUATE_SEARCH = "evaluate_search"
    PROMOTE = "promote"
    CHALLENGE_PACKET = "challenge_packet"


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


class PoolRole(StrEnum):
    SEARCH = "search"
    CALIBRATION = "calibration"
    FINAL = "final"


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
class PoolSliceRef:
    slice_id: str
    role: PoolRole
    manifest_path: str
    manifest_sha256: str
    cluster_ids: tuple[str, ...]
    ordinal: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["cluster_ids"] = list(self.cluster_ids)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoolSliceRef":
        return cls(
            slice_id=str(value["slice_id"]),
            role=PoolRole(str(value["role"])),
            manifest_path=str(value["manifest_path"]),
            manifest_sha256=str(value["manifest_sha256"]),
            cluster_ids=tuple(str(item) for item in value.get("cluster_ids") or []),
            ordinal=int(value.get("ordinal") or 0),
        )


@dataclass(frozen=True)
class BenchmarkPlan:
    benchmark: str
    search: PoolSliceRef
    calibration: tuple[PoolSliceRef, ...]
    final: PoolSliceRef
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "search": self.search.to_dict(),
            "calibration": [item.to_dict() for item in self.calibration],
            "final": self.final.to_dict(),
            "plan_sha256": self.plan_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkPlan":
        return cls(
            benchmark=str(value["benchmark"]),
            search=PoolSliceRef.from_dict(dict(value["search"])),
            calibration=tuple(
                PoolSliceRef.from_dict(dict(item))
                for item in value.get("calibration") or []
            ),
            final=PoolSliceRef.from_dict(dict(value["final"])),
            plan_sha256=str(value["plan_sha256"]),
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
    plan_id: str = ""
    selected_archive_ids: tuple[str, ...] = ()
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
            plan_id=str(value.get("plan_id") or ""),
            selected_archive_ids=tuple(
                str(item) for item in value.get("selected_archive_ids") or []
            ),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ScoreOnlyProposal:
    candidate_id: str
    parent_id: str
    mechanism_claim: str
    predicted_effect: str
    regression_risks: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreOnlyProposal":
        return cls(
            candidate_id=str(value["candidate_id"]),
            parent_id=str(value["parent_id"]),
            mechanism_claim=str(value.get("mechanism_claim") or ""),
            predicted_effect=str(value.get("predicted_effect") or ""),
            regression_risks=tuple(
                str(item) for item in value.get("regression_risks") or []
            ),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ScoreOnlyEvidence:
    iteration: int
    candidate_id: str
    parent_id: str
    search_delta: float | None
    total_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreOnlyDecisionRecord:
    iteration: int
    candidate_id: str
    decision: Decision
    reason: str
    confidence: float
    evidence: ScoreOnlyEvidence

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.value
        return value


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
    search_delta: float | None = None
    archive_replay_passed: bool = True
    preservation_passed: bool = True
    total_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            iteration=int(value["iteration"]),
            candidate_id=str(value["candidate_id"]),
            packet_id=str(value["packet_id"]),
            public_gain=float(value.get("public_gain") or 0.0),
            hidden_gain=float(value.get("hidden_gain") or 0.0),
            bridge_gain=float(value.get("bridge_gain") or 0.0),
            regression_loss=float(value.get("regression_loss") or 0.0),
            admission_score=float(value.get("admission_score") or 0.0),
            search_delta=(
                None
                if value.get("search_delta") is None
                else float(value["search_delta"])
            ),
            archive_replay_passed=bool(value.get("archive_replay_passed", True)),
            preservation_passed=bool(value.get("preservation_passed", True)),
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
    incumbent_search_score: float
    condition: str = "c3_full"
    capabilities: dict[str, bool] = field(default_factory=dict)
    promoted_ids: list[str] = field(default_factory=list)
    archive_ids: list[str] = field(default_factory=list)
    partial_eligible_ids: list[str] = field(default_factory=list)
    quarantined_ids: list[str] = field(default_factory=list)
    challenged_packet_ids: list[str] = field(default_factory=list)
    preserved_packet_refs: list[dict[str, str]] = field(default_factory=list)
    active_packet_id: str = ""
    active_packet_path: str = ""
    active_packet_uses: int = 0
    calibration_epoch: int = 0
    next_calibration_shard: int = 0
    pending_calibration_ids: list[str] = field(default_factory=list)
    committed_iterations: list[int] = field(default_factory=list)
    applied_calibration_checkpoint_ids: list[str] = field(default_factory=list)
    search_cost: float = 0.0
    calibration_cost: float = 0.0
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
            incumbent_search_score=float(value.get("incumbent_search_score") or 0.0),
            condition=str(value.get("condition") or "c3_full"),
            capabilities={
                str(key): bool(item)
                for key, item in dict(value.get("capabilities") or {}).items()
            },
            promoted_ids=[str(x) for x in value.get("promoted_ids") or []],
            archive_ids=[str(x) for x in value.get("archive_ids") or []],
            partial_eligible_ids=[
                str(x) for x in value.get("partial_eligible_ids") or []
            ],
            quarantined_ids=[str(x) for x in value.get("quarantined_ids") or []],
            challenged_packet_ids=[
                str(x) for x in value.get("challenged_packet_ids") or []
            ],
            preserved_packet_refs=[
                {str(key): str(item) for key, item in dict(ref).items()}
                for ref in value.get("preserved_packet_refs") or []
            ],
            active_packet_id=str(value.get("active_packet_id") or ""),
            active_packet_path=str(value.get("active_packet_path") or ""),
            active_packet_uses=int(value.get("active_packet_uses") or 0),
            calibration_epoch=int(value.get("calibration_epoch") or 0),
            next_calibration_shard=int(value.get("next_calibration_shard") or 0),
            pending_calibration_ids=[
                str(x) for x in value.get("pending_calibration_ids") or []
            ],
            committed_iterations=[
                int(x) for x in value.get("committed_iterations") or []
            ],
            applied_calibration_checkpoint_ids=[
                str(x) for x in value.get("applied_calibration_checkpoint_ids") or []
            ],
            search_cost=float(value.get("search_cost") or 0.0),
            calibration_cost=float(value.get("calibration_cost") or 0.0),
            total_cost=float(value.get("total_cost") or 0.0),
        )
