from __future__ import annotations

from traceunit.config import DecisionConfig
from traceunit.models import Decision, DecisionRecord, EvidenceRecord


class DecisionPolicy:
    """Pure policy over frozen unit, replay, and visible search evidence."""

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config

    def decide(self, evidence: EvidenceRecord) -> DecisionRecord:
        cfg = self.config
        if evidence.admission_score < cfg.min_admission_score:
            return self._record(
                evidence,
                Decision.CHALLENGE_PACKET,
                "the TestPacket failed its pre-edit admission contract",
                evidence.admission_score,
            )
        if evidence.regression_loss > cfg.max_regression_loss:
            return self._record(
                evidence,
                Decision.REJECT,
                "the candidate broke an incumbent-passing regression",
                1.0 - evidence.regression_loss,
            )
        if not evidence.archive_replay_passed:
            return self._record(
                evidence,
                Decision.REJECT,
                "the composition failed an original archive certificate",
                1.0,
            )
        if not evidence.preservation_passed:
            return self._record(
                evidence,
                Decision.REJECT,
                "the candidate broke a cumulative promoted capability",
                1.0,
            )

        unit_supported = (
            evidence.public_gain >= cfg.min_public_gain
            and evidence.hidden_gain >= cfg.min_hidden_gain
        )
        if not unit_supported:
            return self._record(
                evidence,
                Decision.REJECT,
                "the candidate did not repair the frozen public and hidden tests",
                max(evidence.public_gain, evidence.hidden_gain),
            )
        if evidence.search_delta is None:
            return self._record(
                evidence,
                Decision.EVALUATE_SEARCH,
                "unit evidence passed; paired search-task evidence is required",
                min(evidence.public_gain, evidence.hidden_gain),
            )
        if evidence.search_delta > cfg.min_search_delta:
            return self._record(
                evidence,
                Decision.PROMOTE,
                "the candidate passed unit, replay, preservation, and search gates",
                min(1.0, 0.5 + evidence.search_delta),
            )

        has_bridge = bool(evidence.metadata.get("has_bridge"))
        bridge_supported = has_bridge and evidence.bridge_gain >= cfg.min_bridge_gain
        if bridge_supported and evidence.search_delta >= -cfg.noninferiority_margin:
            return self._record(
                evidence,
                Decision.ARCHIVE,
                "local and bridge evidence passed while search performance was noninferior",
                min(evidence.hidden_gain, evidence.bridge_gain),
            )
        if bridge_supported and evidence.search_delta < -cfg.noninferiority_margin:
            return self._record(
                evidence,
                Decision.QUARANTINE,
                "the local mechanism passed but natural search performance regressed",
                max(0.0, 1.0 + evidence.search_delta),
            )
        return self._record(
            evidence,
            Decision.REJECT,
            "unit evidence did not propagate to a bridge or search-task improvement",
            evidence.hidden_gain,
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
