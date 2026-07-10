from __future__ import annotations

from traceunit.config import DecisionConfig
from traceunit.decision import DecisionPolicy
from traceunit.models import Decision, EvidenceRecord


def _evidence(**overrides):
    values = {
        "iteration": 1,
        "candidate_id": "candidate",
        "packet_id": "packet",
        "public_gain": 1.0,
        "hidden_gain": 1.0,
        "bridge_gain": 1.0,
        "regression_loss": 0.0,
        "admission_score": 1.0,
        "metadata": {"has_bridge": True},
    }
    values.update(overrides)
    return EvidenceRecord(**values)


def test_unit_supported_candidate_requests_search_evaluation() -> None:
    policy = DecisionPolicy(DecisionConfig())
    assert policy.decide(_evidence()).decision is Decision.EVALUATE_SEARCH


def test_positive_search_delta_promotes_without_hidden_natural_gate() -> None:
    policy = DecisionPolicy(DecisionConfig())
    result = policy.decide(_evidence(search_delta=0.1))
    assert result.decision is Decision.PROMOTE


def test_flat_bridged_edit_is_archived() -> None:
    policy = DecisionPolicy(DecisionConfig(noninferiority_margin=0.02))
    assert policy.decide(_evidence(search_delta=0.0)).decision is Decision.ARCHIVE


def test_negative_bridged_edit_is_quarantined() -> None:
    policy = DecisionPolicy(DecisionConfig(noninferiority_margin=0.02))
    assert policy.decide(_evidence(search_delta=-0.1)).decision is Decision.QUARANTINE


def test_regression_and_certificate_failures_precede_natural_evaluation() -> None:
    policy = DecisionPolicy(DecisionConfig())
    assert policy.decide(_evidence(regression_loss=0.5)).decision is Decision.REJECT
    assert (
        policy.decide(_evidence(archive_replay_passed=False)).decision
        is Decision.REJECT
    )
    assert (
        policy.decide(_evidence(preservation_passed=False)).decision is Decision.REJECT
    )
