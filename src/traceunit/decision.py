from __future__ import annotations

from traceunit.config import DecisionConfig
from traceunit.models import Decision, DecisionRecord, EvidenceRecord


class DecisionPolicy:
    """Deterministic four-way gate; LLM agents never commit candidates."""

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config

    def decide(self, evidence: EvidenceRecord) -> DecisionRecord:
        cfg = self.config
        if evidence.admission_score < cfg.min_admission_score:
            return self._record(
                evidence,
                Decision.TEST_CHALLENGE,
                "test packet did not satisfy its own incumbent admission contract",
                evidence.admission_score,
            )
        if evidence.regression_loss > cfg.max_regression_loss:
            return self._record(
                evidence,
                Decision.REJECT,
                "candidate broke an incumbent-passing regression or admission control",
                1.0 - evidence.regression_loss,
            )
        unit_supported = (
            evidence.public_gain >= cfg.min_public_gain
            and evidence.hidden_gain >= cfg.min_hidden_gain
        )
        if not unit_supported:
            return self._record(
                evidence,
                Decision.REJECT,
                "candidate did not repair the frozen public and hidden mechanism tests",
                max(evidence.public_gain, evidence.hidden_gain),
            )
        if evidence.diagnostic_delta is None:
            return self._record(
                evidence,
                Decision.ESCALATE,
                "mechanism evidence passed; natural diagnostic evidence is still missing",
                min(evidence.public_gain, evidence.hidden_gain),
            )

        has_bridge = bool(evidence.metadata.get("has_bridge"))
        bridge_supported = has_bridge and evidence.bridge_gain >= cfg.min_bridge_gain
        if evidence.diagnostic_delta <= cfg.min_diagnostic_delta:
            if bridge_supported:
                return self._record(
                    evidence,
                    Decision.PARTIAL_ARCHIVE,
                    "hidden mechanism and downstream bridge improved while natural score stayed flat",
                    min(evidence.hidden_gain, evidence.bridge_gain),
                )
            return self._record(
                evidence,
                Decision.REJECT,
                "mechanism evidence did not propagate to a bridge or natural diagnostic gain",
                evidence.hidden_gain,
            )

        if evidence.canary_delta is None:
            return self._record(
                evidence,
                Decision.ESCALATE,
                "diagnostic score improved; hidden natural canary evidence is missing",
                min(1.0, 0.5 + evidence.diagnostic_delta),
            )
        if evidence.canary_delta < -cfg.noninferiority_margin:
            decision = Decision.PARTIAL_ARCHIVE if bridge_supported else Decision.REJECT
            return self._record(
                evidence,
                decision,
                "candidate improved visible tasks but regressed on the hidden canary pool",
                max(0.0, 1.0 + evidence.canary_delta),
            )
        if evidence.canary_delta < cfg.min_canary_delta:
            decision = Decision.PARTIAL_ARCHIVE if bridge_supported else Decision.REJECT
            return self._record(
                evidence,
                decision,
                "hidden canary was non-inferior but did not reach the required gain",
                max(0.0, 0.5 + evidence.canary_delta),
            )
        if cfg.require_audit_for_promotion and evidence.audit_delta is None:
            return self._record(
                evidence,
                Decision.ESCALATE,
                "candidate passed unit, diagnostic, and canary gates; sealed audit is required",
                0.8,
            )
        if (
            evidence.audit_delta is not None
            and evidence.audit_delta <= cfg.min_audit_delta
        ):
            decision = (
                Decision.PARTIAL_ARCHIVE
                if bridge_supported
                and evidence.audit_delta >= -cfg.noninferiority_margin
                else Decision.REJECT
            )
            return self._record(
                evidence,
                decision,
                "sealed audit did not show the required held-out improvement",
                max(0.0, 0.5 + evidence.audit_delta),
            )
        return self._record(
            evidence,
            Decision.PROMOTE,
            (
                "candidate passed mechanism, regression, diagnostic, canary, and audit gates"
                if cfg.require_audit_for_promotion
                else "candidate passed mechanism, regression, diagnostic, and hidden canary gates"
            ),
            0.95,
        )

    @staticmethod
    def _record(
        evidence: EvidenceRecord,
        decision: Decision,
        reason: str,
        confidence: float,
    ) -> DecisionRecord:
        return DecisionRecord(
            iteration=evidence.iteration,
            candidate_id=evidence.candidate_id,
            decision=decision,
            reason=reason,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence,
        )
