from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.io import read_json, sha256_file, sha256_tree, write_json
from traceunit.models import BenchmarkPlan, PoolRole, RunState
from traceunit.paired import paired_task_differences, paired_uncertainty
from traceunit.store import RunStore


@dataclass(frozen=True)
class FinalSubject:
    role: str
    candidate_id: str
    source_path: str
    source_sha256: str


@dataclass(frozen=True)
class FinalEvaluationPlan:
    evaluation_id: str
    search_state_sha256: str
    benchmark_plan_sha256: str
    final_pool: dict[str, Any]
    subjects: tuple[FinalSubject, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "search_state_sha256": self.search_state_sha256,
            "benchmark_plan_sha256": self.benchmark_plan_sha256,
            "final_pool": self.final_pool,
            "subjects": [asdict(item) for item in self.subjects],
        }


class FinalEvaluationRunner:
    """One-way sealed evaluation; this module never imports the calibrator."""

    def __init__(
        self,
        *,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        benchmark_plan: BenchmarkPlan,
    ) -> None:
        self.store = store
        self.benchmark = benchmark
        self.benchmark_plan = benchmark_plan

    def seal(self, state: RunState) -> FinalEvaluationPlan:
        if state.status not in {"completed", "converged"}:
            raise RuntimeError(
                "search must be complete before sealing final evaluation"
            )
        if self.benchmark_plan.final.role is not PoolRole.FINAL:
            raise RuntimeError("benchmark plan does not contain a final pool")
        baseline_source = self.store.root / "candidates" / "baseline" / "source"
        subjects = (
            FinalSubject(
                role="baseline",
                candidate_id="baseline",
                source_path=str(baseline_source.resolve()),
                source_sha256=sha256_tree(baseline_source),
            ),
            FinalSubject(
                role="terminal",
                candidate_id=state.incumbent_id,
                source_path=str(Path(state.incumbent_source).resolve()),
                source_sha256=sha256_tree(Path(state.incumbent_source)),
            ),
        )
        identity = {
            "search_state_sha256": sha256_file(self.store.state_path),
            "benchmark_plan_sha256": self.benchmark_plan.plan_sha256,
            "final_pool": self.benchmark_plan.final.to_dict(),
            "subjects": [asdict(item) for item in subjects],
        }
        evaluation_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        plan = FinalEvaluationPlan(
            evaluation_id=evaluation_id,
            search_state_sha256=identity["search_state_sha256"],
            benchmark_plan_sha256=self.benchmark_plan.plan_sha256,
            final_pool=self.benchmark_plan.final.to_dict(),
            subjects=subjects,
        )
        plan_path = self.store.sealed_root / "final" / "plan.json"
        if plan_path.exists():
            existing = read_json(plan_path)
            if existing != plan.to_dict():
                raise RuntimeError(
                    "a different final evaluation plan is already sealed"
                )
        else:
            write_json(plan_path, plan.to_dict())
        return plan

    def run(self, plan: FinalEvaluationPlan) -> dict[str, Any]:
        evaluations = {}
        for subject in plan.subjects:
            source = Path(subject.source_path)
            if sha256_tree(source) != subject.source_sha256:
                raise RuntimeError(
                    f"final subject source changed after sealing: {subject.candidate_id}"
                )
            evaluations[subject.role] = self.benchmark.evaluate(
                source=source,
                candidate_id=f"final__{subject.candidate_id}",
                pool=self.benchmark_plan.final,
                out_dir=self.store.sealed_root / "final" / "evaluations" / subject.role,
            )
        baseline = evaluations["baseline"]
        terminal = evaluations["terminal"]
        differences = paired_task_differences(baseline, terminal)
        report = {
            "evaluation_id": plan.evaluation_id,
            "baseline_score": baseline.score,
            "terminal_candidate_id": next(
                item.candidate_id for item in plan.subjects if item.role == "terminal"
            ),
            "terminal_score": terminal.score,
            "paired_delta": sum(differences) / len(differences) if differences else 0.0,
            "paired_uncertainty": paired_uncertainty(differences),
            "matched_tasks": len(differences),
            "cost": baseline.cost + terminal.cost,
        }
        write_json(self.store.sealed_root / "final" / "report.json", report)
        return report
