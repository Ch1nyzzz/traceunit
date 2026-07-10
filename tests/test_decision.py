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


def test_decision_cascade_and_partial_archive() -> None:
    policy = DecisionPolicy(DecisionConfig(require_audit_for_promotion=True))
    assert policy.decide(_evidence()).decision == Decision.ESCALATE
    assert (
        policy.decide(_evidence(diagnostic_delta=0.0)).decision
        == Decision.PARTIAL_ARCHIVE
    )
    assert policy.decide(_evidence(diagnostic_delta=0.1)).decision == Decision.ESCALATE
    assert (
        policy.decide(_evidence(diagnostic_delta=0.1, canary_delta=0.0)).decision
        == Decision.ESCALATE
    )
    assert (
        policy.decide(
            _evidence(diagnostic_delta=0.1, canary_delta=0.0, audit_delta=0.1)
        ).decision
        == Decision.PROMOTE
    )


def test_regression_is_rejected_before_natural_evaluation() -> None:
    policy = DecisionPolicy(DecisionConfig())
    assert policy.decide(_evidence(regression_loss=0.5)).decision == Decision.REJECT


def test_nonadaptive_protocol_promotes_after_hidden_canary() -> None:
    policy = DecisionPolicy(DecisionConfig(require_audit_for_promotion=False))
    record = policy.decide(_evidence(diagnostic_delta=0.1, canary_delta=0.0))
    assert record.decision == Decision.PROMOTE
    assert record.evidence.audit_delta is None
