from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping

from traceunit.io import read_json, read_jsonl, write_json
from traceunit.models import BenchmarkEvaluation
from traceunit.store import RunStore


class TraceEvidenceError(RuntimeError):
    pass


class NoFailureTraces(TraceEvidenceError):
    pass


def stage_search_trace_evidence(
    *,
    store: RunStore,
    candidate_id: str,
    destination: Path,
    max_failure_traces: int,
) -> Path:
    """Stage bounded raw search traces without exposing evaluator storage."""

    evaluation_path = store.evaluation_dir(candidate_id, "search") / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"missing cached search evaluation: {evaluation_path}")
    evaluation = BenchmarkEvaluation.from_dict(read_json(evaluation_path))
    rows = read_jsonl(Path(evaluation.trace_path))
    failed = [
        row
        for row in rows
        if not bool(row.get("passed"))
        and str(row.get("status") or "ok") in {"ok", "unresolved"}
    ]
    failed.sort(
        key=lambda row: (float(row.get("score") or 0.0), str(row.get("task_id")))
    )
    if not failed:
        invalid = [row for row in rows if not bool(row.get("passed"))]
        if invalid:
            statuses = sorted({str(row.get("status") or "unknown") for row in invalid})
            raise TraceEvidenceError(
                "search pool has failures but none are behavioral traces; "
                f"statuses={statuses}"
            )
        raise NoFailureTraces

    successful = [row for row in rows if bool(row.get("passed"))][:2]
    selected = failed[:max_failure_traces] + successful
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, Any]] = []
    for row in selected:
        copied = dict(row)
        digest = hashlib.sha256(str(row.get("trace_id")).encode()).hexdigest()[:16]
        trace_dir = destination / "artifacts" / digest
        trace_dir.mkdir(parents=True, exist_ok=True)
        staged_paths: list[str] = []
        for index, raw_path in enumerate(row.get("artifact_paths") or []):
            source = Path(str(raw_path))
            if not source.is_file():
                continue
            target = trace_dir / f"{index:02d}_{source.name}"
            shutil.copy2(source, target)
            staged_paths.append(str(target.relative_to(destination)))
        copied["artifact_paths"] = staged_paths
        copied["events"] = _sanitize_events(copied.get("events") or [], staged_paths)
        metrics = dict(copied.get("metrics") or {})
        metrics.pop("task_dump", None)
        copied["metrics"] = metrics
        staged.append(copied)
    manifest = destination / "manifest.json"
    write_json(manifest, {"traces": staged})
    return manifest


def _sanitize_events(
    events: list[Any], staged_paths: list[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    artifact_index = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        staged = dict(event)
        if event.get("kind") == "artifact":
            staged["input"] = {
                "staged_artifact": (
                    staged_paths[artifact_index]
                    if artifact_index < len(staged_paths)
                    else None
                )
            }
            artifact_index += 1
        result.append(staged)
    return result
