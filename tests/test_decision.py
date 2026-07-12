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
        "packet_id": "packet",
        "public_gain": 1.0,
        "hidden_gain": 1.0,
        "bridge_gain": 1.0,
        "regression_loss": 0.0,
        "contract_passed": True,
        "bridge_contract_passed": True,
        "search_delta": 0.0,
        "metadata": {"has_bridge": True},
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
    assert "deviated from the search distribution" in result.reason
    assert is_mismatch(evidence, config)
    assert archive_kind(evidence, config) is None


def test_cell4_search_gain_with_failed_unit_archives_as_mismatch() -> None:
    config = DecisionConfig()
    evidence = _evidence(search_delta=0.1, contract_passed=False)
    result = DecisionPolicy(config).decide(evidence)
    assert result.decision is Decision.ARCHIVE
    assert "unit contract failed" in result.reason
    assert is_mismatch(evidence, config)
    assert archive_kind(evidence, config) == ARCHIVE_SEARCH_IMPROVED_UNIT_FAILED


def test_cell5_both_failed_rejects() -> None:
    config = DecisionConfig()
    for delta in (0.0, -0.1):
        evidence = _evidence(search_delta=delta, contract_passed=False)
        result = DecisionPolicy(config).decide(evidence)
        assert result.decision is Decision.REJECT
        assert not is_mismatch(evidence, config)
        assert archive_kind(evidence, config) is None


def test_preservation_and_regression_failures_count_as_unit_failures() -> None:
    config = DecisionConfig()
    broke_regression = _evidence(regression_loss=0.5)
    broke_preserved = _evidence(preservation_passed=False)
    assert not unit_ok(broke_regression, config)
    assert not unit_ok(broke_preserved, config)
    policy = DecisionPolicy(config)
    assert policy.decide(broke_regression).decision is Decision.REJECT
    assert policy.decide(broke_preserved).decision is Decision.REJECT
    # With a search gain they land in cell 4, like any other unit failure.
    gained = _evidence(preservation_passed=False, search_delta=0.1)
    assert policy.decide(gained).decision is Decision.ARCHIVE
    assert is_mismatch(gained, config)


def test_missing_search_evidence_is_rejected() -> None:
    config = DecisionConfig()
    evidence = _evidence(search_delta=None)
    assert DecisionPolicy(config).decide(evidence).decision is Decision.REJECT
    assert not is_mismatch(evidence, config)
    assert archive_kind(evidence, config) is None
