from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from traceunit.agents.prompts import regression_author_prompt
from traceunit.agents.runner import WorkspaceAgent
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.decision import DecisionPolicy
from traceunit.io import copy_source, read_json, write_json
from traceunit.models import (
    AttributionScope,
    BenchmarkEvaluation,
    BenchmarkPlan,
    CandidateProposal,
    Decision,
    DecisionRecord,
    EvidenceRecord,
    PoolSliceRef,
    RunState,
    TestPacket,
    TestTier,
    UnitFamily,
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
        packet_reused: bool,
    ) -> tuple[EvidenceRecord, DecisionRecord]:
        violations = mechanical_violations(
            benchmark=self.benchmark,
            candidate_source=candidate_source,
            diff_text=diff_text,
            out_dir=iteration_dir / "smoke",
        )
        metadata = {
            "parent_id": state.incumbent_id,
            "parent_source": state.incumbent_source,
            "candidate_source": str(candidate_source.resolve()),
            "packet_reused": packet_reused,
            "has_bridge": any(case.tier is TestTier.BRIDGE for case in packet.cases),
            "violations": violations,
            "costs": {
                "unit_test_wall_seconds": 0.0,
                "model_probe_calls": 0,
                "model_probe_tokens": 0,
                "natural_task_tokens": 0.0,
            },
        }
        if violations:
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
                reason="; ".join(violations),
                confidence=1.0,
                evidence=evidence,
            )

        pair_dir = iteration_dir / "paired_tests"
        incumbent_results = run_test_cases(
            packet=packet,
            bundle=packet_path,
            source=Path(state.incumbent_source),
            subject="incumbent",
            output_dir=pair_dir / "incumbent",
            python=self.config.benchmark.unit_python,
            probe_runner=self.benchmark.run_agent_probe,
        )
        candidate_results = run_test_cases(
            packet=packet,
            bundle=packet_path,
            source=candidate_source,
            subject="candidate",
            output_dir=pair_dir / "candidate",
            python=self.config.benchmark.unit_python,
            probe_runner=self.benchmark.run_agent_probe,
        )
        metrics = paired_test_metrics(packet, incumbent_results, candidate_results)
        contract_passed, contract_reasons = candidate_contract(
            packet, candidate_results
        )
        has_bridge = bool(metadata["has_bridge"])
        bridge_contract_passed, bridge_contract_reasons = candidate_contract(
            packet,
            candidate_results,
            tiers=frozenset({TestTier.BRIDGE}),
        )
        if not has_bridge:
            bridge_contract_passed = False
        regression_loss = max(
            metrics["regression_loss"],
            self._regression_loss(
                iteration_dir=iteration_dir,
                state=state,
                proposal=proposal,
                candidate_source=candidate_source,
                diff_text=diff_text,
            ),
        )
        preservation = self._replay(
            refs=(
                FrozenPacketRef.from_dict(item) for item in state.preserved_packet_refs
            ),
            candidate_source=candidate_source,
            output_dir=iteration_dir / "preservation_replay",
        )
        latent = self._replay(
            refs=(
                FrozenPacketRef.from_dict(item) for item in state.latent_packet_refs
            ),
            candidate_source=candidate_source,
            output_dir=iteration_dir / "latent_replay",
        )
        realized = tuple(
            result.content_sha256 for result in latent if result.contract_passed
        )
        component_families = tuple(
            sorted(
                {
                    UnitFamily(result.primary_family)
                    for result in latent
                    if result.contract_passed and result.primary_family
                },
                key=lambda item: item.value,
            )
        )
        unit_seconds = sum(
            result.duration_s for result in [*incumbent_results, *candidate_results]
        )
        probe_calls = sum(
            result.model_calls for result in [*incumbent_results, *candidate_results]
        )
        probe_tokens = sum(
            result.tokens for result in [*incumbent_results, *candidate_results]
        )
        metadata.update(
            {
                "candidate_contract_passed": contract_passed,
                "candidate_contract_reasons": contract_reasons,
                "bridge_contract_passed": bridge_contract_passed,
                "bridge_contract_reasons": bridge_contract_reasons,
                "preservation_replay": [item.to_dict() for item in preservation],
                "latent_replay": [item.to_dict() for item in latent],
                "incumbent_test_results": [
                    result.to_dict() for result in incumbent_results
                ],
                "candidate_test_results": [
                    result.to_dict() for result in candidate_results
                ],
                "costs": {
                    "unit_test_wall_seconds": unit_seconds,
                    "model_probe_calls": probe_calls,
                    "model_probe_tokens": probe_tokens,
                    "natural_task_tokens": 0.0,
                },
            }
        )
        evidence = EvidenceRecord(
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            packet_id=packet.packet_id,
            public_gain=metrics["public_gain"],
            hidden_gain=metrics["hidden_gain"],
            bridge_gain=metrics["bridge_gain"],
            regression_loss=regression_loss,
            contract_passed=contract_passed,
            bridge_contract_passed=bridge_contract_passed,
            primary_family=packet.primary_family,
            intervention_kind=proposal.intervention_kind,
            attribution_scope=(
                AttributionScope.COMPOSITION if realized else AttributionScope.ATOMIC
            ),
            component_families=component_families,
            realized_latent=realized,
            preservation_passed=all(item.contract_passed for item in preservation),
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
        costs = dict(metadata["costs"])
        costs["natural_task_tokens"] = candidate_eval.cost
        metadata["costs"] = costs
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
        decision = self.policy.decide(evidence)
        if (
            decision.decision is Decision.ARCHIVE
            and not self.config.capabilities.partial_archive
        ):
            decision = DecisionRecord(
                iteration=iteration,
                candidate_id=proposal.candidate_id,
                decision=Decision.PARTIAL_ELIGIBLE,
                reason=(
                    "the edit is partial-archive eligible, but component persistence "
                    "is disabled for this condition"
                ),
                confidence=decision.confidence,
                evidence=evidence,
            )
        return evidence, decision

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

    def _replay(
        self,
        *,
        refs: Iterable[FrozenPacketRef],
        candidate_source: Path,
        output_dir: Path,
    ) -> tuple[PacketReplayResult, ...]:
        return PacketReplayer(
            packet_root=self.store.packet_store_root,
            python=self.config.benchmark.unit_python,
            probe_runner=self.benchmark.run_agent_probe,
        ).replay(
            refs=refs,
            candidate_source=candidate_source,
            output_dir=output_dir,
        )

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
