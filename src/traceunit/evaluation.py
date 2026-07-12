from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from traceunit.agents.prompts import regression_author_prompt
from traceunit.agents.runner import WorkspaceAgent
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.decision import DecisionPolicy
from traceunit.io import copy_source, read_json, write_json
from traceunit.models import (
    BenchmarkEvaluation,
    BenchmarkPlan,
    CandidateProposal,
    Decision,
    DecisionRecord,
    EvidenceRecord,
    PoolSliceRef,
    RunState,
    TestExecution,
    TestPacket,
    TestTier,
)
from traceunit.paired import paired_task_differences
from traceunit.replay import FrozenPacketRef, PacketReplayer, PacketReplayResult
from traceunit.store import RunStore
from traceunit.tests_runtime import (
    InvalidTestPacket,
    admission_contract,
    candidate_contract,
    load_test_packet,
    paired_test_metrics,
    run_test_cases,
)


@dataclass(frozen=True)
class UnitEvidence:
    """One complete unit verdict for a candidate source against a frozen packet.

    Produced by ``UnitEvidenceRunner`` both inside the proposer's inner retry
    loop (as concrete feedback) and as the authoritative unit half of the
    final decision evidence.
    """

    violations: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    contract_passed: bool = False
    contract_reasons: tuple[str, ...] = ()
    bridge_contract_passed: bool = False
    bridge_contract_reasons: tuple[str, ...] = ()
    has_bridge: bool = False
    preservation: tuple[PacketReplayResult, ...] = ()
    incumbent_results: tuple[TestExecution, ...] = ()
    candidate_results: tuple[TestExecution, ...] = ()
    attempts: int = 1
    unit_seconds: float = 0.0
    probe_calls: int = 0
    probe_tokens: int = 0

    @property
    def preservation_passed(self) -> bool:
        return all(item.contract_passed for item in self.preservation)

    def unit_ok(self, max_regression_loss: float) -> bool:
        return (
            not self.violations
            and self.contract_passed
            and self.preservation_passed
            and self.metrics.get("regression_loss", 0.0) <= max_regression_loss
        )


class UnitEvidenceRunner:
    """Run the frozen packet plus all preserved contracts against one source."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark

    def run(
        self,
        *,
        packet: TestPacket,
        packet_path: Path,
        incumbent_source: Path,
        candidate_source: Path,
        preserved_refs: Iterable[dict[str, str]],
        diff_text: str,
        output_dir: Path,
        incumbent_results: tuple[TestExecution, ...] | None = None,
    ) -> UnitEvidence:
        violations = mechanical_violations(
            benchmark=self.benchmark,
            candidate_source=candidate_source,
            diff_text=diff_text,
            out_dir=output_dir / "smoke",
        )
        if violations:
            return UnitEvidence(violations=tuple(violations))
        if incumbent_results is None:
            incumbent_results = run_test_cases(
                packet=packet,
                bundle=packet_path,
                source=incumbent_source,
                subject="incumbent",
                output_dir=output_dir / "incumbent",
                python=self.config.benchmark.unit_python,
                probe_runner=self.benchmark.run_agent_probe,
            )
        candidate_results = run_test_cases(
            packet=packet,
            bundle=packet_path,
            source=candidate_source,
            subject="candidate",
            output_dir=output_dir / "candidate",
            python=self.config.benchmark.unit_python,
            probe_runner=self.benchmark.run_agent_probe,
        )
        metrics = paired_test_metrics(packet, incumbent_results, candidate_results)
        contract_passed, contract_reasons = candidate_contract(
            packet, candidate_results
        )
        has_bridge = any(case.tier is TestTier.BRIDGE for case in packet.cases)
        bridge_contract_passed, bridge_contract_reasons = candidate_contract(
            packet,
            candidate_results,
            tiers=frozenset({TestTier.BRIDGE}),
        )
        if not has_bridge:
            bridge_contract_passed = False
        preservation = PacketReplayer(
            packet_root=self.store.packet_store_root,
            python=self.config.benchmark.unit_python,
            probe_runner=self.benchmark.run_agent_probe,
        ).replay(
            refs=(FrozenPacketRef.from_dict(item) for item in preserved_refs),
            candidate_source=candidate_source,
            output_dir=output_dir / "preservation",
        )
        executions = [*incumbent_results, *candidate_results]
        return UnitEvidence(
            violations=(),
            metrics=metrics,
            contract_passed=contract_passed,
            contract_reasons=tuple(contract_reasons),
            bridge_contract_passed=bridge_contract_passed,
            bridge_contract_reasons=tuple(bridge_contract_reasons),
            has_bridge=has_bridge,
            preservation=preservation,
            incumbent_results=tuple(incumbent_results),
            candidate_results=tuple(candidate_results),
            unit_seconds=sum(item.duration_s for item in executions),
            probe_calls=sum(item.model_calls for item in executions),
            probe_tokens=sum(item.tokens for item in executions),
        )


class CandidateEvaluator:
    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        benchmark_plan: BenchmarkPlan,
        policy: DecisionPolicy,
        regression_author: WorkspaceAgent | None,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.plan = benchmark_plan
        self.policy = policy
        self.regression_author = regression_author

    def evaluate_candidate(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        proposal: CandidateProposal,
        packet: TestPacket,
        packet_path: Path,
        candidate_source: Path,
        diff_text: str,
        unit: UnitEvidence,
    ) -> tuple[EvidenceRecord, DecisionRecord]:
        metadata = {
            "parent_id": state.incumbent_id,
            "parent_source": state.incumbent_source,
            "candidate_source": str(candidate_source.resolve()),
            "has_bridge": unit.has_bridge,
            "violations": list(unit.violations),
            "unit_attempts": unit.attempts,
            "costs": {
                "unit_test_wall_seconds": unit.unit_seconds,
                "model_probe_calls": unit.probe_calls,
                "model_probe_tokens": unit.probe_tokens,
            },
        }
        if unit.violations:
            evidence = EvidenceRecord(
                iteration=iteration,
                candidate_id=proposal.candidate_id,
                packet_id=packet.packet_id,
                public_gain=0.0,
                hidden_gain=0.0,
                bridge_gain=0.0,
                regression_loss=1.0,
                contract_passed=False,
                bridge_contract_passed=False,
                primary_family=packet.primary_family,
                intervention_kind=proposal.intervention_kind,
                preservation_passed=False,
                metadata=metadata,
            )
            return evidence, DecisionRecord(
                iteration=iteration,
                candidate_id=proposal.candidate_id,
                decision=Decision.REJECT,
                reason="; ".join(unit.violations),
                confidence=1.0,
                evidence=evidence,
            )

        regression_loss = max(
            unit.metrics["regression_loss"],
            self._regression_loss(
                iteration_dir=iteration_dir,
                state=state,
                proposal=proposal,
                candidate_source=candidate_source,
                diff_text=diff_text,
            ),
        )
        metadata.update(
            {
                "candidate_contract_passed": unit.contract_passed,
                "candidate_contract_reasons": list(unit.contract_reasons),
                "bridge_contract_passed": unit.bridge_contract_passed,
                "bridge_contract_reasons": list(unit.bridge_contract_reasons),
                "preservation_replay": [
                    item.to_dict() for item in unit.preservation
                ],
                "incumbent_test_results": [
                    result.to_dict() for result in unit.incumbent_results
                ],
                "candidate_test_results": [
                    result.to_dict() for result in unit.candidate_results
                ],
            }
        )
        evidence = EvidenceRecord(
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            packet_id=packet.packet_id,
            public_gain=unit.metrics["public_gain"],
            hidden_gain=unit.metrics["hidden_gain"],
            bridge_gain=unit.metrics["bridge_gain"],
            regression_loss=regression_loss,
            contract_passed=unit.contract_passed,
            bridge_contract_passed=unit.bridge_contract_passed,
            primary_family=packet.primary_family,
            intervention_kind=proposal.intervention_kind,
            preservation_passed=unit.preservation_passed,
            metadata=metadata,
        )
        candidate_eval = self.evaluate_pool(
            source=candidate_source,
            candidate_id=proposal.candidate_id,
            pool=self.plan.search,
        )
        differences = self._search_differences(
            parent_id=state.incumbent_id,
            candidate=candidate_eval,
        )
        search_delta = sum(differences) / len(differences) if differences else 0.0
        metadata = dict(evidence.metadata)
        metadata["search"] = {
            "candidate_score": candidate_eval.score,
            "candidate_passrate": candidate_eval.passrate,
            "paired_task_count": len(differences),
        }
        evidence = replace(
            evidence,
            search_delta=search_delta,
            total_cost=candidate_eval.cost,
            metadata=metadata,
        )
        return evidence, self.policy.decide(evidence)

    def evaluate_pool(
        self,
        *,
        source: Path,
        candidate_id: str,
        pool: PoolSliceRef,
        cache_tag: str = "",
    ) -> BenchmarkEvaluation:
        storage_id = candidate_id if not cache_tag else f"{candidate_id}__{cache_tag}"
        return self.benchmark.evaluate(
            source=source,
            candidate_id=storage_id,
            pool=pool,
            out_dir=self.store.evaluation_dir(storage_id, pool.slice_id),
        )

    def _search_differences(
        self, *, parent_id: str, candidate: BenchmarkEvaluation
    ) -> list[float]:
        path = self.store.evaluation_dir(parent_id, self.plan.search.slice_id)
        parent = BenchmarkEvaluation.from_dict(read_json(path / "evaluation.json"))
        return paired_task_differences(parent, candidate)

    def _regression_loss(
        self,
        *,
        iteration_dir: Path,
        state: RunState,
        proposal: CandidateProposal,
        candidate_source: Path,
        diff_text: str,
    ) -> float:
        if self.regression_author is None:
            return 0.0
        workspace = iteration_dir / "regression_author" / "workspace"
        incumbent = workspace / "incumbent_source"
        candidate = workspace / "candidate_source"
        copy_source(Path(state.incumbent_source), incumbent)
        copy_source(candidate_source, candidate)
        diff_path = workspace / "candidate.diff"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(diff_text, encoding="utf-8")
        proposal_path = workspace / "proposal.json"
        write_json(proposal_path, proposal.to_dict())
        output = workspace / "output"
        run = self.regression_author.run(
            role="regression_author",
            prompt=regression_author_prompt(
                benchmark_context=self.benchmark.context(),
                incumbent_source=incumbent,
                candidate_source=candidate,
                diff_path=diff_path,
                proposal_path=proposal_path,
                output_dir=output,
            ),
            workspace=workspace,
            log_dir=iteration_dir / "regression_author" / "agent",
        )
        if run.returncode != 0 or run.timed_out:
            return 1.0
        try:
            packet = load_test_packet(output)
        except InvalidTestPacket:
            return 1.0
        incumbent_results = run_test_cases(
            packet=packet,
            bundle=output,
            source=Path(state.incumbent_source),
            subject="incumbent",
            output_dir=iteration_dir / "regression_author" / "incumbent",
            python=self.config.benchmark.unit_python,
        )
        admitted, _ = admission_contract(packet, incumbent_results)
        if not admitted:
            return 1.0
        candidate_results = run_test_cases(
            packet=packet,
            bundle=output,
            source=candidate_source,
            subject="candidate",
            output_dir=iteration_dir / "regression_author" / "candidate",
            python=self.config.benchmark.unit_python,
        )
        passed, _ = candidate_contract(packet, candidate_results)
        return 0.0 if passed else 1.0


def _external_symlink_violations(source: Path) -> list[str]:
    root = source.resolve()
    violations: list[str] = []
    for path in source.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            target = path.resolve(strict=False)
        except OSError:
            violations.append(
                f"unresolvable source symlink: {path.relative_to(source)}"
            )
            continue
        if target != root and root not in target.parents:
            violations.append(
                f"source symlink escapes candidate snapshot: "
                f"{path.relative_to(source)} -> {target}"
            )
    return violations


def mechanical_violations(
    *,
    benchmark: BenchmarkAdapter,
    candidate_source: Path,
    diff_text: str,
    out_dir: Path,
) -> list[str]:
    smoke_ok, smoke_message = benchmark.smoke_test(candidate_source, out_dir)
    violations = [
        *benchmark.policy_violations(candidate_source, diff_text),
        *_external_symlink_violations(candidate_source),
    ]
    if not diff_text.strip():
        violations.append("candidate source is identical to the incumbent")
    if not smoke_ok:
        violations.append(f"candidate smoke check failed: {smoke_message[-1000:]}")
    return violations
