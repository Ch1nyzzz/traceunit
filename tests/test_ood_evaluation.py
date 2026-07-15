from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from traceunit.io import write_json
from traceunit.models import (
    BenchmarkEvaluation,
    PoolRole,
    RunState,
    TaskOutcome,
)
from traceunit.ood_evaluation import OODEvaluationRunner
from traceunit.store import RunStore


class _FakeAdapter:
    """Records evaluate() calls and returns a fixed baseline/terminal contrast."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def evaluate(
        self, *, source: Path, candidate_id: str, pool: object, out_dir: Path
    ) -> BenchmarkEvaluation:
        self.calls.append((candidate_id, source))
        # Terminal harness flips task t1; t2 already passes for both.
        scores = (
            {"t1": 0.0, "t2": 1.0}
            if "baseline" in candidate_id
            else {"t1": 1.0, "t2": 1.0}
        )
        outcomes = tuple(
            TaskOutcome(task_id=key, score=value, passed=bool(value), trace_id=key)
            for key, value in scores.items()
        )
        mean = sum(scores.values()) / len(scores)
        return BenchmarkEvaluation(
            evaluation_id=candidate_id,
            benchmark="hle",
            candidate_id=candidate_id,
            split="final",
            score=mean,
            passrate=mean,
            cost=1.0,
            outcomes=outcomes,
            trace_path="traces.jsonl",
            result_path="result.json",
        )


def _ood_plan() -> SimpleNamespace:
    return SimpleNamespace(final=SimpleNamespace(role=PoolRole.FINAL))


def _write_source_run(tmp_path: Path, *, status: str) -> Path:
    source_run = tmp_path / "source_run"
    (source_run / "candidates" / "baseline" / "source").mkdir(parents=True)
    incumbent_source = source_run / "candidates" / "iter003_candidate" / "source"
    incumbent_source.mkdir(parents=True)
    state = RunState(
        run_id="hle_source_v1",
        benchmark="hle",
        status=status,
        next_iteration=4,
        incumbent_id="iter003_candidate",
        incumbent_source=str(incumbent_source),
        incumbent_search_score=0.5,
    )
    write_json(source_run / "run_state.json", state.to_dict())
    return source_run


def test_ood_runner_scores_both_harnesses_and_reports_transfer_delta(
    tmp_path: Path,
) -> None:
    source_run = _write_source_run(tmp_path, status="completed")
    adapter = _FakeAdapter()
    runner = OODEvaluationRunner(
        ood_store=RunStore(tmp_path / "ood_run"),
        benchmark=adapter,
        ood_plan=_ood_plan(),
        source_run_dir=source_run,
    )

    report = runner.run()

    # Baseline is scored before terminal, each on the OOD pool.
    assert [candidate_id for candidate_id, _ in adapter.calls] == [
        "ood__baseline__baseline",
        "ood__terminal__iter003_candidate",
    ]
    assert report["baseline_score"] == 0.5
    assert report["terminal_score"] == 1.0
    # Paired delta over t1 (0->1) and t2 (1->1) is 0.5.
    assert report["paired_delta"] == 0.5
    assert report["matched_tasks"] == 2
    assert report["terminal_candidate_id"] == "iter003_candidate"
    assert report["cost"] == 2.0
    written = tmp_path / "ood_run" / "sealed" / "ood" / "report.json"
    assert written.is_file()


def test_ood_runner_rejects_incomplete_source_run(tmp_path: Path) -> None:
    source_run = _write_source_run(tmp_path, status="running")
    runner = OODEvaluationRunner(
        ood_store=RunStore(tmp_path / "ood_run"),
        benchmark=_FakeAdapter(),
        ood_plan=_ood_plan(),
        source_run_dir=source_run,
    )
    with pytest.raises(RuntimeError, match="must be complete"):
        runner.run()


def test_ood_runner_rejects_missing_source_run(tmp_path: Path) -> None:
    runner = OODEvaluationRunner(
        ood_store=RunStore(tmp_path / "ood_run"),
        benchmark=_FakeAdapter(),
        ood_plan=_ood_plan(),
        source_run_dir=tmp_path / "does_not_exist",
    )
    with pytest.raises(RuntimeError, match="no completed source run"):
        runner.run()
