from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

from traceunit.io import write_json
from traceunit.models import BenchmarkEvaluation, TaskOutcome, TraceEvent, TraceRun


_ARTIFACT_CHAR_LIMIT = 100_000

_SAFE_TRACE_METRICS = {
    "agent_exit_code",
    "apply_error",
    "benchmark",
    "base_commit",
    "dry_run",
    "duration_s",
    "empty_patch",
    "error_tail",
    "eval_exit_code",
    "eval_returncodes",
    "evaluator_returncode",
    "exit_status",
    "ground_truth_isolated",
    "infra_error",
    "model_stats",
    "patch_bytes",
    "patch_exists",
    "patch_successfully_applied",
    "rep_successes",
    "resolved",
    "repo",
    "returncode",
    "run_returncodes",
    "run_status",
    "status_reason",
    "task_dump",
    "timed_out",
    "timeout",
    "timeout_s",
    "verdict",
}


@contextmanager
def worldcalib_import(root: Path) -> Iterator[None]:
    """Make the local WorldCalib evaluation substrate importable temporarily."""

    src = str((root / "src").resolve())
    inserted = src not in sys.path
    if inserted:
        sys.path.insert(0, src)
    try:
        yield
    finally:
        if inserted and src in sys.path:
            sys.path.remove(src)


def normalize_worldcalib_result(
    *,
    result_path: Path,
    benchmark: str,
    split: str,
    candidate_id: str,
    out_dir: Path,
) -> BenchmarkEvaluation:
    """Convert the task-result envelope and raw dumps into TraceRun JSONL."""

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = dict(payload.get("candidate") or {})
    raw_tasks = payload.get("tasks") or []
    traces: list[TraceRun] = []
    outcomes: list[TaskOutcome] = []
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            continue
        task_id = str(raw.get("task_id") or f"task-{index}")
        metadata = dict(raw.get("metadata") or {})
        artifact_paths: list[str] = []
        events: list[TraceEvent] = []
        dump_texts = [str(value) for value in metadata.get("task_dumps") or [] if value]
        if not dump_texts and metadata.get("task_dump"):
            dump_texts = [str(metadata["task_dump"])]
        evidence_names = metadata.get("dump_evidence_files") or []
        event_index = 0
        for repetition, dump_text in enumerate(dump_texts):
            dump = Path(dump_text)
            if not dump.is_dir():
                continue
            for name in evidence_names:
                evidence = dump / str(name)
                if not evidence.is_file():
                    continue
                resolved = evidence.resolve()
                artifact_paths.append(str(resolved))
                text = resolved.read_text(encoding="utf-8", errors="replace")
                was_truncated = len(text) > _ARTIFACT_CHAR_LIMIT
                if was_truncated:
                    half = _ARTIFACT_CHAR_LIMIT // 2
                    text = text[:half] + "\n...[middle truncated]...\n" + text[-half:]
                events.append(
                    TraceEvent(
                        event_id=f"artifact-{event_index}",
                        kind="artifact",
                        input={
                            "path": str(resolved),
                            "name": resolved.name,
                            "repetition": repetition,
                        },
                        output=text,
                        metadata={"truncated": was_truncated},
                    )
                )
                event_index += 1
        trace_id = f"{benchmark}:{split}:{candidate_id}:{task_id}"
        score = float(raw.get("score") or 0.0)
        passed = bool(raw.get("passed"))
        safe_metrics = {
            key: value for key, value in metadata.items() if key in _SAFE_TRACE_METRICS
        }
        trace = TraceRun(
            trace_id=trace_id,
            benchmark=benchmark,
            split=split,
            candidate_id=candidate_id,
            task_id=task_id,
            score=score,
            passed=passed,
            status=str(metadata.get("run_status") or "ok"),
            input_summary=str(raw.get("question") or "")[:20_000],
            output_summary=str(raw.get("prediction") or "")[:20_000],
            events=tuple(events),
            artifact_paths=tuple(artifact_paths),
            metrics=safe_metrics,
        )
        traces.append(trace)
        outcomes.append(
            TaskOutcome(
                task_id=task_id,
                score=score,
                passed=passed,
                trace_id=trace_id,
                metadata={"status": trace.status},
            )
        )
    trace_path = out_dir / "traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False))
            handle.write("\n")
    evaluation = BenchmarkEvaluation(
        evaluation_id=f"{benchmark}:{split}:{candidate_id}",
        benchmark=benchmark,
        candidate_id=candidate_id,
        split=split,
        score=float(candidate.get("average_score") or candidate.get("passrate") or 0.0),
        passrate=float(candidate.get("passrate") or 0.0),
        cost=float(candidate.get("token_consuming") or 0.0),
        outcomes=tuple(outcomes),
        trace_path=str(trace_path),
        result_path=str(result_path),
        metadata={
            "count": int(candidate.get("count") or len(outcomes)),
            "score_breakdown": payload.get("score_breakdown") or {},
            "candidate_config": candidate.get("config") or {},
        },
    )
    write_json(out_dir / "evaluation.json", evaluation.to_dict())
    return evaluation


def load_cached_evaluation(out_dir: Path) -> BenchmarkEvaluation | None:
    path = out_dir / "evaluation.json"
    if not path.is_file():
        return None
    try:
        return BenchmarkEvaluation.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception:
        return None
