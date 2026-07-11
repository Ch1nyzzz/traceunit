from __future__ import annotations

from traceunit.config import DecisionConfig
from traceunit.models import Decision, DecisionRecord, EvidenceRecord


class DecisionPolicy:
    """Pure policy over frozen unit, replay, and visible search evidence."""

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config

    def decide(self, evidence: EvidenceRecord) -> DecisionRecord:
        cfg = self.config
        if evidence.regression_loss > cfg.max_regression_loss:
            return self._record(
                evidence,
                Decision.REJECT,
                "the candidate broke an incumbent-passing regression",
                1.0 - evidence.regression_loss,
            )
        if not evidence.preservation_passed:
            return self._record(
                evidence,
                Decision.REJECT,
                "the candidate broke a cumulative promoted capability",
                1.0,
            )
        if evidence.search_delta is None:
            return self._record(
                evidence,
                Decision.REJECT,
                "a mechanically valid candidate is missing paired search evidence",
                0.0,
            )
        if evidence.search_delta > cfg.min_search_delta:
            if not evidence.contract_passed:
                return self._record(
                    evidence,
                    Decision.REJECT,
                    "search improved but the candidate did not satisfy its frozen TestPacket",
                    0.0,
                )
            return self._record(
                evidence,
                Decision.PROMOTE,
                "the candidate satisfied its frozen contract and improved paired search",
                min(1.0, 0.5 + evidence.search_delta),
            )

        has_bridge = bool(evidence.metadata.get("has_bridge"))
        bridge_certified = has_bridge and evidence.bridge_contract_passed
        if (
            evidence.contract_passed
            and bridge_certified
            and evidence.search_delta >= -cfg.noninferiority_margin
        ):
            return self._record(
                evidence,
                Decision.ARCHIVE,
                "the frozen contract and bridge passed while paired search was noninferior",
                0.5,
            )
        if evidence.contract_passed and bridge_certified:
            return self._record(
                evidence,
                Decision.QUARANTINE,
                "the frozen contract passed but paired search regressed",
                max(0.0, 1.0 + evidence.search_delta),
            )
        if not evidence.contract_passed:
            reason = "the candidate did not satisfy its frozen TestPacket"
        elif has_bridge and not evidence.bridge_contract_passed:
            reason = "the candidate did not satisfy its frozen downstream bridge"
        else:
            reason = "paired search did not improve and no certified bridge supports archival"
        return self._record(
            evidence,
            Decision.REJECT,
            reason,
            0.0,
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
