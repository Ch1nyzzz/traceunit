from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from traceunit.models import BenchmarkEvaluation


def paired_task_differences(
    parent: BenchmarkEvaluation, candidate: BenchmarkEvaluation
) -> list[float]:
    """Return deterministic paired deltas, rejecting incomplete or duplicate pairs."""

    parent_scores = _score_map(parent)
    candidate_scores = _score_map(candidate)
    if parent_scores.keys() != candidate_scores.keys():
        missing = sorted(parent_scores.keys() - candidate_scores.keys())
        extra = sorted(candidate_scores.keys() - parent_scores.keys())
        raise ValueError(
            "paired evaluations contain different task ids: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return [
        candidate_scores[task_id] - parent_scores[task_id]
        for task_id in sorted(parent_scores)
    ]


def paired_uncertainty(differences: Iterable[float]) -> float:
    values = list(differences)
    if not values:
        raise ValueError("paired evaluation contains no task outcomes")
    if len(values) < 2:
        return 1.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _score_map(evaluation: BenchmarkEvaluation) -> dict[str, float]:
    scores: dict[str, float] = {}
    for outcome in evaluation.outcomes:
        if not outcome.task_id:
            raise ValueError("evaluation outcome has an empty task id")
        if outcome.task_id in scores:
            raise ValueError(f"duplicate task outcome: {outcome.task_id}")
        scores[outcome.task_id] = outcome.score
    if not scores:
        raise ValueError(
            f"evaluation {evaluation.evaluation_id!r} contains no task outcomes"
        )
    return scores
