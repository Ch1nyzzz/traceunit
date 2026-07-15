"""Top-K passrate frontier.

The frontier is the set of candidates the proposer's ``pareto`` selection
policy may resample its patch base from. It is a strict top-K by
passrate, with ties at the K-th position broken by ``candidate_id``
(string comparison, descending — later iter ids win).

Single-objective by design: only ``passrate`` matters. Other ParetoPoint
fields (``average_score``, ``token_consuming``) remain on the dataclass
for downstream consumers that display them, but they do not influence
selection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FRONTIER_TOP_K = 3


@dataclass(frozen=True)
class ParetoPoint:
    """One point in the frontier. Only ``passrate`` and ``candidate_id``
    drive selection; the rest is metadata for display."""

    candidate_id: str
    scaffold_name: str
    passrate: float
    token_consuming: int
    avg_token_consuming: float
    average_score: float
    result_path: str
    config: dict


def dominates(a: ParetoPoint, b: ParetoPoint, *, quality_gap_threshold: float = 0.0) -> bool:
    """Strict passrate dominance: ``a`` dominates ``b`` iff ``a.passrate > b.passrate``.

    ``quality_gap_threshold`` is retained for API compatibility with
    older callers; it is no longer consulted.
    """

    del quality_gap_threshold  # deadcode for API compat
    return a.passrate > b.passrate


def pareto_frontier(
    points: Iterable[ParetoPoint],
    *,
    quality_gap_threshold: float = 0.0,
    top_k: int = FRONTIER_TOP_K,
) -> list[ParetoPoint]:
    """Strict top-K by passrate, ties broken by ``candidate_id`` desc.

    Returns ``min(top_k, len(points))`` points: ``[]`` when ``points`` is
    empty, the whole pool when ``len(points) < top_k``.

    ``quality_gap_threshold`` is retained for API compatibility; it is
    no longer consulted.
    """

    del quality_gap_threshold  # deadcode for API compat
    pool = list(points)
    if not pool:
        return []
    # Two-pass stable sort: secondary key first, primary key last.
    pool.sort(key=lambda item: item.candidate_id, reverse=True)
    pool.sort(key=lambda item: item.passrate, reverse=True)
    return pool[: max(0, int(top_k))]


def save_frontier(
    path: Path,
    points: Iterable[ParetoPoint],
    *,
    quality_gap_threshold: float = 0.0,
) -> None:
    """Write the frontier JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        asdict(point)
        for point in pareto_frontier(points, quality_gap_threshold=quality_gap_threshold)
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_frontier(path: Path) -> list[ParetoPoint]:
    """Load a frontier JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return [ParetoPoint(**_normalize_point(item)) for item in data]


def lex_better(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Lexicographic dominance: passrate primary, token_consuming tiebreaker.

    Used for benchmarks (graph-colouring) where the two evaluation axes are
    not co-equal: the harness wants whichever candidate scored *strictly*
    better on the primary axis, and falls back to the cost axis only when
    the primaries match exactly.

    Returns True iff ``a`` is strictly lex-better than ``b``.
    """

    if a.passrate > b.passrate:
        return True
    if a.passrate < b.passrate:
        return False
    return a.token_consuming < b.token_consuming


def lex_pareto_frontier(
    points: Iterable[ParetoPoint],
    *,
    top_k: int = FRONTIER_TOP_K,
) -> list[ParetoPoint]:
    """Top-K by lex order: passrate desc, then token_consuming asc.

    Ties on both axes break by ``candidate_id`` desc so later iterations win
    on identical metrics. This mirrors ``pareto_frontier``'s contract but
    uses the lex tiebreaker required by the graph-colouring evaluation
    rule (colors primary, runtime tiebreaker — encoded as passrate /
    token_consuming on the stored candidate dataclass).
    """

    pool = list(points)
    if not pool:
        return []
    pool.sort(key=lambda item: item.candidate_id, reverse=True)
    pool.sort(key=lambda item: item.token_consuming)
    pool.sort(key=lambda item: item.passrate, reverse=True)
    return pool[: max(0, int(top_k))]


def save_lex_frontier(
    path: Path,
    points: Iterable[ParetoPoint],
    *,
    top_k: int = FRONTIER_TOP_K,
) -> None:
    """Write a lex-ordered top-K frontier JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(point) for point in lex_pareto_frontier(points, top_k=top_k)]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_point(item: dict) -> dict:
    data = dict(item)
    if "scaffold_name" not in data and "seed_name" in data:
        data["scaffold_name"] = data.pop("seed_name")
    return data
