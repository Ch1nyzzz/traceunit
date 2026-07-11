from __future__ import annotations

from traceunit.config import DecisionConfig
from traceunit.decision import DecisionPolicy
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


def test_positive_search_with_frozen_contract_promotes() -> None:
    result = DecisionPolicy(DecisionConfig()).decide(_evidence(search_delta=0.1))
    assert result.decision is Decision.PROMOTE


def test_search_gain_without_contract_is_not_promoted() -> None:
    result = DecisionPolicy(DecisionConfig()).decide(
        _evidence(search_delta=0.1, contract_passed=False)
    )
    assert result.decision is Decision.REJECT
    assert "frozen TestPacket" in result.reason


def test_noninferior_certified_bridge_archives() -> None:
    result = DecisionPolicy(DecisionConfig(noninferiority_margin=0.02)).decide(
        _evidence(search_delta=0.0)
    )
    assert result.decision is Decision.ARCHIVE


def test_regressed_certified_bridge_is_quarantined() -> None:
    result = DecisionPolicy(DecisionConfig(noninferiority_margin=0.02)).decide(
        _evidence(search_delta=-0.1)
    )
    assert result.decision is Decision.QUARANTINE


def test_regression_and_preservation_failures_precede_search() -> None:
    policy = DecisionPolicy(DecisionConfig())
    assert policy.decide(_evidence(regression_loss=0.5)).decision is Decision.REJECT
    assert (
        policy.decide(_evidence(preservation_passed=False)).decision is Decision.REJECT
    )


def test_missing_search_evidence_is_rejected() -> None:
    assert (
        DecisionPolicy(DecisionConfig()).decide(_evidence(search_delta=None)).decision
        is Decision.REJECT
    )
