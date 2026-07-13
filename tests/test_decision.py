from __future__ import annotations

from traceunit.config import DecisionConfig
from traceunit.decision import (
    ARCHIVE_SEARCH_IMPROVED_UNIT_FAILED,
    ARCHIVE_UNIT_PASSED_SEARCH_FLAT,
    DecisionPolicy,
    archive_kind,
    is_mismatch,
    unit_ok,
)
from traceunit.models import Decision, EvidenceRecord


def _evidence(**overrides: object) -> EvidenceRecord:
    values: dict[str, object] = {
        "iteration": 1,
        "candidate_id": "candidate",
        "target_capability": "evidence-before-mutation",
        "target_improved": True,
        "collateral_ok": True,
        "target_delta": 0.5,
        "collateral_delta": 0.0,
        "search_delta": 0.0,
        "metadata": {},
    }
    values.update(overrides)
    return EvidenceRecord(**values)  # type: ignore[arg-type]


def test_cell1_unit_pass_and_search_gain_promotes() -> None:
    result = DecisionPolicy(DecisionConfig()).decide(_evidence(search_delta=0.1))
    assert result.decision is Decision.PROMOTE


def test_cell2_unit_pass_and_flat_search_archives() -> None:
    config = DecisionConfig(noninferiority_margin=0.02)
    evidence = _evidence(search_delta=0.0)
    result = DecisionPolicy(config).decide(evidence)
    assert result.decision is Decision.ARCHIVE
    assert archive_kind(evidence, config) == ARCHIVE_UNIT_PASSED_SEARCH_FLAT
    assert not is_mismatch(evidence, config)
    # A small paired dip inside the noise band still counts as flat.
    assert (
        DecisionPolicy(config).decide(_evidence(search_delta=-0.01)).decision
        is Decision.ARCHIVE
    )


def test_cell3_unit_pass_and_search_regression_rejects_as_mismatch() -> None:
    config = DecisionConfig(noninferiority_margin=0.02)
    evidence = _evidence(search_delta=-0.1)
    result = DecisionPolicy(config).decide(evidence)
    assert result.decision is Decision.REJECT
    assert "deviates from the search distribution" in result.reason
    assert is_mismatch(evidence, config)
    assert archive_kind(evidence, config) is None


def test_cell4_search_gain_with_failed_unit_archives_as_mismatch() -> None:
    config = DecisionConfig()
    evidence = _evidence(search_delta=0.1, target_improved=False)
    result = DecisionPolicy(config).decide(evidence)
    assert result.decision is Decision.ARCHIVE
    assert "battery did not certify" in result.reason
    assert is_mismatch(evidence, config)
    assert archive_kind(evidence, config) == ARCHIVE_SEARCH_IMPROVED_UNIT_FAILED


def test_cell5_both_failed_rejects() -> None:
    config = DecisionConfig()
    for delta in (0.0, -0.1):
        evidence = _evidence(search_delta=delta, target_improved=False)
        result = DecisionPolicy(config).decide(evidence)
        assert result.decision is Decision.REJECT
        assert not is_mismatch(evidence, config)
        assert archive_kind(evidence, config) is None


def test_collateral_damage_counts_as_unit_failure() -> None:
    config = DecisionConfig()
    damaged = _evidence(collateral_ok=False, collateral_delta=-0.5)
    assert not unit_ok(damaged, config)
    policy = DecisionPolicy(config)
    result = policy.decide(damaged)
    assert result.decision is Decision.REJECT
    assert "damaged other capabilities" in result.reason
    # With a search gain it lands in cell 4, like any other unit failure.
    gained = _evidence(collateral_ok=False, search_delta=0.1)
    assert policy.decide(gained).decision is Decision.ARCHIVE
    assert is_mismatch(gained, config)


def test_missing_search_evidence_is_rejected() -> None:
    config = DecisionConfig()
    evidence = _evidence(search_delta=None)
    assert DecisionPolicy(config).decide(evidence).decision is Decision.REJECT
    assert not is_mismatch(evidence, config)
    assert archive_kind(evidence, config) is None
