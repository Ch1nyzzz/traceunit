"""Out-of-distribution transfer evaluation (SEAGym method A).

A source-domain run (e.g. HLE Math+Physics) optimizes an editable harness. This
module takes the harness that run produced -- both the clean baseline and the
optimized (terminal incumbent) harness -- and scores each on a frozen OOD pool
built from a different domain (e.g. HLE CS/AI+Engineering). The paired delta is
the transfer measurement: how much the source-domain optimization helped on a
domain it was never trained on. This module never runs the optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.io import write_json
from traceunit.models import BenchmarkPlan, PoolRole
from traceunit.paired import paired_task_differences, paired_uncertainty
from traceunit.store import RunStore


class OODEvaluationRunner:
    """Score a source run's baseline and optimized harness on an OOD pool."""

    def __init__(
        self,
        *,
        ood_store: RunStore,
        benchmark: BenchmarkAdapter,
        ood_plan: BenchmarkPlan,
        source_run_dir: Path,
    ) -> None:
        self.ood_store = ood_store
        self.benchmark = benchmark
        self.ood_plan = ood_plan
        self.source_run_dir = source_run_dir

    def run(self) -> dict[str, Any]:
        if self.ood_plan.final.role is not PoolRole.FINAL:
            raise RuntimeError("OOD plan does not contain a final pool")
        source_store = RunStore(self.source_run_dir)
        state = source_store.load_state()
        if state is None:
            raise RuntimeError(f"no completed source run under {self.source_run_dir}")
        if state.status not in {"completed", "converged"}:
            raise RuntimeError(
                "source-domain search must be complete before OOD evaluation; "
                f"status is {state.status!r}"
            )

        baseline_source = source_store.root / "candidates" / "baseline" / "source"
        subjects = (
            ("baseline", "baseline", baseline_source),
            ("terminal", state.incumbent_id, Path(state.incumbent_source)),
        )
        evaluations = {}
        for role, candidate_id, source in subjects:
            source = source.resolve()
            if not source.is_dir():
                raise RuntimeError(f"{role} harness source is missing: {source}")
            evaluations[role] = self.benchmark.evaluate(
                source=source,
                candidate_id=f"ood__{role}__{candidate_id}",
                pool=self.ood_plan.final,
                out_dir=self.ood_store.sealed_root / "ood" / "evaluations" / role,
            )

        baseline = evaluations["baseline"]
        terminal = evaluations["terminal"]
        differences = paired_task_differences(baseline, terminal)
        report = {
            "source_run": str(source_store.root),
            "terminal_candidate_id": state.incumbent_id,
            "baseline_score": baseline.score,
            "terminal_score": terminal.score,
            "paired_delta": sum(differences) / len(differences) if differences else 0.0,
            "paired_uncertainty": paired_uncertainty(differences),
            "matched_tasks": len(differences),
            "cost": baseline.cost + terminal.cost,
        }
        write_json(self.ood_store.sealed_root / "ood" / "report.json", report)
        return report
