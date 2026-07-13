from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Decision(StrEnum):
    REJECT = "reject"
    ARCHIVE = "archive"
    PROMOTE = "promote"


class TestTier(StrEnum):
    PUBLIC = "public"
    HIDDEN = "hidden"
    BRIDGE = "bridge"
    REGRESSION = "regression"
    ADMISSION = "admission"


class UnitFamily(StrEnum):
    """Coarse trace-diagnosis direction, never a transfer ranking."""

    INSTRUCTION = "instruction"
    CONTEXT = "context"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    STATE = "state"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    TERMINATION = "termination"
    OTHER = "other"
    UNCERTAIN = "uncertain"


class EvidenceRole(StrEnum):
    TARGET_REPRODUCER = "target_reproducer"
    STRUCTURAL_SIBLING = "structural_sibling"
    DOWNSTREAM_BRIDGE = "downstream_bridge"
    POSITIVE_WITNESS = "positive_witness"
    PRESERVATION_CONTROL = "preservation_control"
    OFF_TARGET_CONTROL = "off_target_control"


class InterventionKind(StrEnum):
    LOCAL_REPAIR = "local_repair"
    CAPABILITY_AUGMENTATION = "capability_augmentation"
    ORCHESTRATION_CHANGE = "orchestration_change"


class TestExecutionMode(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL_BACKED_PROBE = "model_backed_probe"


class TestStatus(StrEnum):
    PROPOSED = "proposed"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    RETIRED = "retired"


class PoolRole(StrEnum):
    SEARCH = "search"
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
    family: UnitFamily
    intervention_kind: InterventionKind
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
            family=UnitFamily(str(value.get("family") or UnitFamily.UNCERTAIN)),
            intervention_kind=InterventionKind(
                str(value.get("intervention_kind") or InterventionKind.LOCAL_REPAIR)
            ),
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
    tier: TestTier
    evidence_role: EvidenceRole
    path: str
    driver: str = "python"
    execution_mode: TestExecutionMode = TestExecutionMode.DETERMINISTIC
    arguments: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 60
    expected_incumbent_pass: bool = False
    expected_candidate_pass: bool = True
    description: str = ""
    max_model_calls: int = 0
    max_tokens: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TestCaseSpec":
        return cls(
            case_id=str(value["case_id"]),
            tier=TestTier(str(value["tier"])),
            evidence_role=EvidenceRole(str(value["evidence_role"])),
            path=str(value["path"]),
            driver=str(value.get("driver") or "python"),
            execution_mode=TestExecutionMode(
                str(value.get("execution_mode") or TestExecutionMode.DETERMINISTIC)
            ),
            arguments=tuple(str(x) for x in value.get("arguments") or []),
            environment={
                str(k): str(v) for k, v in dict(value.get("environment") or {}).items()
            },
            timeout_s=max(1, int(value.get("timeout_s") or 60)),
            expected_incumbent_pass=bool(value.get("expected_incumbent_pass", False)),
            expected_candidate_pass=bool(value.get("expected_candidate_pass", True)),
            description=str(value.get("description") or ""),
            max_model_calls=max(0, int(value.get("max_model_calls") or 0)),
            max_tokens=max(0, int(value.get("max_tokens") or 0)),
        )


@dataclass(frozen=True)
class TestPacket:
    packet_id: str
    version: int
    hypotheses: tuple[FailureHypothesis, ...]
    target_hypothesis_id: str
    primary_family: UnitFamily | None
    public_contract: str
    hidden_variant_strategy: str
    cases: tuple[TestCaseSpec, ...]
    status: TestStatus = TestStatus.PROPOSED
    admission_passed: bool = False
    content_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["primary_family"] = (
            self.primary_family.value if self.primary_family is not None else None
        )
        for hypothesis in value["hypotheses"]:
            hypothesis["family"] = str(hypothesis["family"])
            hypothesis["intervention_kind"] = str(hypothesis["intervention_kind"])
        for case in value["cases"]:
            case["tier"] = str(case["tier"])
            case["evidence_role"] = str(case["evidence_role"])
            case["execution_mode"] = str(case["execution_mode"])
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TestPacket":
        return cls(
            packet_id=str(value["packet_id"]),
            version=int(value.get("version") or 1),
            hypotheses=tuple(
                FailureHypothesis.from_dict(item)
                for item in value.get("hypotheses") or []
                if isinstance(item, Mapping)
            ),
            target_hypothesis_id=str(value["target_hypothesis_id"]),
            primary_family=(
                None
                if value.get("primary_family") in (None, "")
                else UnitFamily(str(value["primary_family"]))
            ),
            public_contract=str(value.get("public_contract") or ""),
            hidden_variant_strategy=str(value.get("hidden_variant_strategy") or ""),
            cases=tuple(
                TestCaseSpec.from_dict(item)
                for item in value.get("cases") or []
                if isinstance(item, Mapping)
            ),
            status=TestStatus(str(value.get("status") or TestStatus.PROPOSED)),
            admission_passed=bool(value.get("admission_passed", False)),
            content_sha256=str(value.get("content_sha256") or ""),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TestExecution:
    case_id: str
    tier: TestTier
    evidence_role: EvidenceRole
    execution_mode: TestExecutionMode
    subject: str
    passed: bool
    returncode: int | None
    duration_s: float
    stdout_path: str
    stderr_path: str
    timed_out: bool = False
    error: str = ""
    model_calls: int = 0
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tier"] = self.tier.value
        value["evidence_role"] = self.evidence_role.value
        value["execution_mode"] = self.execution_mode.value
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
        )


@dataclass(frozen=True)
class BenchmarkPlan:
    benchmark: str
    search: PoolSliceRef
    final: PoolSliceRef
    plan_sha256: str
    ontology: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "search": self.search.to_dict(),
            "final": self.final.to_dict(),
            "plan_sha256": self.plan_sha256,
            "ontology": dict(self.ontology),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkPlan":
        return cls(
            benchmark=str(value["benchmark"]),
            search=PoolSliceRef.from_dict(dict(value["search"])),
            final=PoolSliceRef.from_dict(dict(value["final"])),
            plan_sha256=str(value["plan_sha256"]),
            ontology={
                str(key): str(item)
                for key, item in dict(value.get("ontology") or {}).items()
            },
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
    mechanism_claim: str
    predicted_effect: str
    hypothesis_id: str = ""
    intervention_kind: InterventionKind = InterventionKind.LOCAL_REPAIR
    regression_risks: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateProposal":
        return cls(
            candidate_id=str(value["candidate_id"]),
            parent_id=str(value["parent_id"]),
            hypothesis_id=str(value.get("hypothesis_id") or ""),
            intervention_kind=InterventionKind(
                str(value.get("intervention_kind") or InterventionKind.LOCAL_REPAIR)
            ),
            mechanism_claim=str(value.get("mechanism_claim") or ""),
            predicted_effect=str(value.get("predicted_effect") or ""),
            regression_risks=tuple(str(x) for x in value.get("regression_risks") or []),
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
    """The unit half is the capability battery: did the diagnosed capability's
    pass rate improve, and did every other capability stay intact."""

    iteration: int
    candidate_id: str
    target_capability: str
    target_improved: bool
    collateral_ok: bool
    target_delta: float = 0.0
    collateral_delta: float = 0.0
    primary_family: UnitFamily | None = None
    intervention_kind: InterventionKind = InterventionKind.LOCAL_REPAIR
    search_delta: float | None = None
    total_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["primary_family"] = (
            self.primary_family.value if self.primary_family is not None else None
        )
        value["intervention_kind"] = self.intervention_kind.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            iteration=int(value["iteration"]),
            candidate_id=str(value["candidate_id"]),
            target_capability=str(value.get("target_capability") or ""),
            target_improved=bool(value.get("target_improved", False)),
            collateral_ok=bool(value.get("collateral_ok", False)),
            target_delta=float(value.get("target_delta") or 0.0),
            collateral_delta=float(value.get("collateral_delta") or 0.0),
            primary_family=(
                None
                if value.get("primary_family") in (None, "")
                else UnitFamily(str(value["primary_family"]))
            ),
            intervention_kind=InterventionKind(
                str(value.get("intervention_kind") or InterventionKind.LOCAL_REPAIR)
            ),
            search_delta=(
                None
                if value.get("search_delta") is None
                else float(value["search_delta"])
            ),
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
    # Archived candidates are records, not protocol capabilities: an edit whose
    # contract passed while search stayed flat, or whose search improved while
    # its contract failed. Later agents read the records and may re-litigate
    # them through the normal propose -> unit -> search path.
    archived_ids: list[str] = field(default_factory=list)
    archive_refs: list[dict[str, str]] = field(default_factory=list)
    committed_iterations: list[int] = field(default_factory=list)
    search_cost: float = 0.0
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
            archived_ids=[str(x) for x in value.get("archived_ids") or []],
            archive_refs=[
                {str(key): str(item) for key, item in dict(ref).items()}
                for ref in value.get("archive_refs") or []
            ],
            search_cost=float(value.get("search_cost") or 0.0),
            committed_iterations=[
                int(x) for x in value.get("committed_iterations") or []
            ],
            total_cost=float(value.get("total_cost") or 0.0),
        )
