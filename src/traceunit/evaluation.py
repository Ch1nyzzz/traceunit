from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from traceunit.battery import (
    Battery,
    BatteryResult,
    BatteryRunner,
    battery_deltas,
)
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.decision import DecisionPolicy
from traceunit.io import read_json
from traceunit.models import (
    BenchmarkEvaluation,
    BenchmarkPlan,
    CandidateProposal,
    Decision,
    DecisionRecord,
    EvidenceRecord,
    PoolSliceRef,
    RunState,
    UnitFamily,
)
from traceunit.paired import paired_task_differences
from traceunit.store import RunStore


@dataclass(frozen=True)
class UnitEvidence:
    """One complete battery verdict for a candidate source.

    Produced by ``UnitEvidenceRunner`` both inside the proposer's inner retry
    loop (as concrete feedback) and as the authoritative unit half of the
    final decision evidence.
    """

    violations: tuple[str, ...]
    target_capability: str = ""
    results: tuple[BatteryResult, ...] = ()
    deltas: dict[str, dict] = field(default_factory=dict)
    target_improved: bool = False
    collateral_ok: bool = False
    target_delta: float = 0.0
    collateral_delta: float = 0.0
    attempts: int = 1
    unit_seconds: float = 0.0
    probe_calls: int = 0
    probe_tokens: int = 0

    def unit_ok(self) -> bool:
        return not self.violations and self.target_improved and self.collateral_ok

    def instance_results(self) -> dict[str, bool]:
        return {item.instance_id: item.passed for item in self.results}


class UnitEvidenceRunner:
    """Run the full capability battery against one candidate source."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        battery: Battery,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.battery = battery
        self.runner = BatteryRunner(
            battery=battery,
            python=config.benchmark.unit_python,
            probe_runner=benchmark.run_agent_probe,
        )

    def run(
        self,
        *,
        target_capability: str,
        candidate_source: Path,
        diff_text: str,
        output_dir: Path,
    ) -> UnitEvidence:
        violations = mechanical_violations(
            benchmark=self.benchmark,
            candidate_source=candidate_source,
            diff_text=diff_text,
            out_dir=output_dir / "smoke",
        )
        if violations:
            return UnitEvidence(
                violations=tuple(violations), target_capability=target_capability
            )
        results = self.runner.run(
            source=candidate_source,
            subject="candidate",
            output_dir=output_dir / "battery",
        )
        deltas = battery_deltas(
            instances=self.battery.load(),
            reference=self.battery.load_reference(),
            results=results,
        )
        target = deltas.get(target_capability) or {}
        target_improved = int(target.get("candidate_passed") or 0) > int(
            target.get("incumbent_passed") or 0
        )
        collateral = [
            float(delta.get("delta") or 0.0)
            for capability, delta in deltas.items()
            if capability != target_capability
        ]
        collateral_delta = min(collateral, default=0.0)
        collateral_ok = (
            collateral_delta >= -self.config.decision.max_battery_regression
        )
        return UnitEvidence(
            violations=(),
            target_capability=target_capability,
            results=results,
            deltas={key: dict(value) for key, value in deltas.items()},
            target_improved=target_improved,
            collateral_ok=collateral_ok,
            target_delta=float(target.get("delta") or 0.0),
            collateral_delta=collateral_delta,
            unit_seconds=sum(item.duration_s for item in results),
            probe_calls=sum(item.model_calls for item in results),
            probe_tokens=sum(item.tokens for item in results),
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
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.plan = benchmark_plan
        self.policy = policy

    def evaluate_candidate(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        proposal: CandidateProposal,
        target_capability: str,
        target_family: UnitFamily,
        candidate_source: Path,
        diff_text: str,
        unit: UnitEvidence,
    ) -> tuple[EvidenceRecord, DecisionRecord]:
        metadata = {
            "parent_id": state.incumbent_id,
            "parent_source": state.incumbent_source,
            "candidate_source": str(candidate_source.resolve()),
            "violations": list(unit.violations),
            "unit_attempts": unit.attempts,
            "battery_deltas": unit.deltas,
            "battery_instance_results": unit.instance_results(),
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
                target_capability=target_capability,
                target_improved=False,
                collateral_ok=False,
                primary_family=target_family,
                intervention_kind=proposal.intervention_kind,
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

        evidence = EvidenceRecord(
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            target_capability=target_capability,
            target_improved=unit.target_improved,
            collateral_ok=unit.collateral_ok,
            target_delta=unit.target_delta,
            collateral_delta=unit.collateral_delta,
            primary_family=target_family,
            intervention_kind=proposal.intervention_kind,
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
