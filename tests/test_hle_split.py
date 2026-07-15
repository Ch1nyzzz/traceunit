from __future__ import annotations

from pathlib import Path

import pytest

from traceunit.benchmarks.hle import _cluster_key, _split_rows
from traceunit.config import load_config


def _source_rows(n_math: int = 70, n_physics: int = 30) -> list[dict[str, str]]:
    """A Math+Physics source pool where one cluster (Math) dominates."""

    rows = [
        {"id": f"m{i}", "category": "Math", "raw_subject": "Mathematics"}
        for i in range(n_math)
    ]
    rows += [
        {"id": f"p{i}", "category": "Physics", "raw_subject": "Physics"}
        for i in range(n_physics)
    ]
    return rows


def test_question_split_balances_a_dominant_cluster() -> None:
    rows = _source_rows()
    search, final = _split_rows(
        rows,
        seed=1729,
        search_fraction=0.5,
        cluster_key=_cluster_key,
        split_by="question",
    )
    assert search and final
    # Per-question assignment keeps the search fraction near 0.5 regardless of
    # the 70/30 cluster imbalance.
    assert 0.35 < len(search) / len(rows) < 0.65
    # The dominant Math cluster must be present on both sides, not wholesale on
    # one -- that is the whole point of splitting by question.
    math_in_search = sum(1 for row in search if row["category"] == "Math")
    assert 0 < math_in_search < 70


def test_cluster_split_assigns_whole_subjects_wholesale() -> None:
    rows = _source_rows()
    search, _ = _split_rows(
        rows,
        seed=1729,
        search_fraction=0.5,
        cluster_key=_cluster_key,
        split_by="cluster",
    )
    # With only two clusters, whole-cluster assignment puts every Math question
    # on a single side -- the imbalance that motivates question-level splitting.
    math_in_search = sum(1 for row in search if row["category"] == "Math")
    assert math_in_search in (0, 70)


def test_question_split_is_deterministic_and_order_independent() -> None:
    rows = _source_rows()
    first = _split_rows(
        rows,
        seed=1729,
        search_fraction=0.5,
        cluster_key=_cluster_key,
        split_by="question",
    )
    second = _split_rows(
        list(reversed(rows)),
        seed=1729,
        search_fraction=0.5,
        cluster_key=_cluster_key,
        split_by="question",
    )
    assert {row["id"] for row in first[0]} == {row["id"] for row in second[0]}
    assert {row["id"] for row in first[1]} == {row["id"] for row in second[1]}


def test_split_rejects_unknown_split_by() -> None:
    with pytest.raises(ValueError, match="unknown split_by"):
        _split_rows(
            _source_rows(),
            seed=1,
            search_fraction=0.5,
            cluster_key=_cluster_key,
            split_by="bogus",
        )


def test_config_accepts_question_split_by(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "loop:\n"
        "  run_dir: run\n"
        "benchmark:\n"
        "  name: hle\n"
        "  hle_split_by: question\n"
        f"  env_file: {tmp_path / 'absent.env'}\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.benchmark.hle_split_by == "question"


def test_config_rejects_bad_hle_split_by(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "loop:\n"
        "  run_dir: run\n"
        "benchmark:\n"
        "  name: hle\n"
        "  hle_split_by: bogus\n"
        f"  env_file: {tmp_path / 'absent.env'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hle_split_by"):
        load_config(config_path)
