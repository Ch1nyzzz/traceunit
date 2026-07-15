"""Normalize native (non-WorldCalib) evaluation envelopes into TraceRun JSONL.

Harbor-hosted benchmarks (Terminal-Bench 2.0) and the native HLE runner both
emit the same portable result envelope -- a ``candidate`` summary plus a list of
per-task records. This module converts that envelope into a ``BenchmarkEvaluation``
and a ``traces.jsonl`` file, mirroring what ``common.normalize_worldcalib_result``
does for the WorldCalib adapters but without importing any WorldCalib substrate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from traceunit.io import write_json
from traceunit.models import BenchmarkEvaluation, TaskOutcome, TraceEvent, TraceRun

_SUMMARY_CHAR_LIMIT = 20_000

# Per-task keys copied verbatim into a trace's ``metrics``. Everything here is
# safe telemetry about the run itself -- never gold answers or verifier internals.
_SAFE_TASK_METRIC_KEYS = (
    "reward",
    "status",
    "trial_name",
    "source",
    "prompt_tokens",
    "completion_tokens",
    "cache_tokens",
    "total_tokens",
    "cost_usd",
    "duration_s",
    "returncode",
    "error",
)


def normalize_trial_results(
    *,
    result_path: Path,
    benchmark: str,
    split: str,
    candidate_id: str,
    out_dir: Path,
    extra_events: Mapping[str, list[TraceEvent]] | None = None,
) -> BenchmarkEvaluation:
    """Convert a native result envelope into a ``BenchmarkEvaluation``.

    ``extra_events`` optionally maps a task id to trace events (e.g. the HLE
    runner attaching the prompt/answer exchange); Harbor runs pass nothing.
    """

    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    candidate = dict(payload.get("candidate") or {})
    raw_tasks = payload.get("tasks") or []
    events_by_task = dict(extra_events or {})

    traces: list[TraceRun] = []
    outcomes: list[TaskOutcome] = []
    total_tokens = 0
    total_cost = 0.0
    passed_count = 0
    score_sum = 0.0
    status_counts: dict[str, int] = {}

    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            continue
        task_id = str(raw.get("task_id") or f"task-{index}")
        score = float(raw.get("score") or 0.0)
        passed = bool(raw.get("passed"))
        status = str(raw.get("status") or "ok")
        prompt_tokens = int(raw.get("prompt_tokens") or 0)
        completion_tokens = int(raw.get("completion_tokens") or 0)
        task_tokens = int(raw.get("total_tokens") or (prompt_tokens + completion_tokens))
        cost = float(raw.get("cost_usd") or 0.0)

        total_tokens += task_tokens
        total_cost += cost
        passed_count += 1 if passed else 0
        score_sum += score
        status_counts[status] = status_counts.get(status, 0) + 1

        trace_id = f"{benchmark}:{split}:{candidate_id}:{task_id}"
        metrics = {
            key: raw[key] for key in _SAFE_TASK_METRIC_KEYS if raw.get(key) is not None
        }
        trace = TraceRun(
            trace_id=trace_id,
            benchmark=benchmark,
            split=split,
            candidate_id=candidate_id,
            task_id=task_id,
            score=score,
            passed=passed,
            status=status,
            input_summary=str(raw.get("question") or "")[:_SUMMARY_CHAR_LIMIT],
            output_summary=str(raw.get("prediction") or "")[:_SUMMARY_CHAR_LIMIT],
            events=tuple(events_by_task.get(task_id) or ()),
            artifact_paths=tuple(str(p) for p in raw.get("artifact_paths") or []),
            metrics=metrics,
        )
        traces.append(trace)
        outcomes.append(
            TaskOutcome(
                task_id=task_id,
                score=score,
                passed=passed,
                trace_id=trace_id,
                metadata={
                    "status": status,
                    "total_tokens": task_tokens,
                    "cost_usd": cost,
                },
            )
        )

    count = len(outcomes)
    trace_path = out_dir / "traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False))
            handle.write("\n")

    average_score = (
        float(candidate["average_score"])
        if candidate.get("average_score") is not None
        else (score_sum / count if count else 0.0)
    )
    passrate = (
        float(candidate["passrate"])
        if candidate.get("passrate") is not None
        else (passed_count / count if count else 0.0)
    )
    cost = (
        float(candidate["token_consuming"])
        if candidate.get("token_consuming") is not None
        else float(total_tokens)
    )

    evaluation = BenchmarkEvaluation(
        evaluation_id=f"{benchmark}:{split}:{candidate_id}",
        benchmark=benchmark,
        candidate_id=candidate_id,
        split=split,
        score=average_score,
        passrate=passrate,
        cost=cost,
        outcomes=tuple(outcomes),
        trace_path=str(trace_path),
        result_path=str(result_path),
        metadata={
            "count": count,
            "total_tokens": total_tokens,
            "monetary_cost": total_cost,
            "task_status_counts": status_counts,
            "job_stats": payload.get("job_stats") or {},
            "candidate_config": candidate.get("config") or {},
            "worker_error": payload.get("worker_error"),
        },
    )
    write_json(out_dir / "evaluation.json", evaluation.to_dict())
    return evaluation
