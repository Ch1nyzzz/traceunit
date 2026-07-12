from __future__ import annotations

from traceunit.config import DecisionConfig
from traceunit.models import Decision, DecisionRecord, EvidenceRecord

ARCHIVE_UNIT_PASSED_SEARCH_FLAT = "unit_passed_search_flat"
ARCHIVE_SEARCH_IMPROVED_UNIT_FAILED = "search_improved_unit_failed"


def unit_ok(evidence: EvidenceRecord, config: DecisionConfig) -> bool:
    """The complete unit verdict: frozen contract, preserved contracts, regressions."""

    return (
        evidence.contract_passed
        and evidence.preservation_passed
        and evidence.regression_loss <= config.max_regression_loss
    )


def is_mismatch(evidence: EvidenceRecord, config: DecisionConfig) -> bool:
    """Unit verdict and paired search disagree: the UT design missed the search
    distribution (unit passed, search regressed) or missed the mechanism (search
    improved, unit failed). Either way the next Test Author must diagnose it."""

    if evidence.search_delta is None:
        return False
    if unit_ok(evidence, config):
        return evidence.search_delta < -config.noninferiority_margin
    return evidence.search_delta > config.min_search_delta


def archive_kind(evidence: EvidenceRecord, config: DecisionConfig) -> str | None:
    if evidence.search_delta is None:
        return None
    if unit_ok(evidence, config):
        if (
            evidence.search_delta <= config.min_search_delta
            and evidence.search_delta >= -config.noninferiority_margin
        ):
            return ARCHIVE_UNIT_PASSED_SEARCH_FLAT
        return None
    if evidence.search_delta > config.min_search_delta:
        return ARCHIVE_SEARCH_IMPROVED_UNIT_FAILED
    return None


class DecisionPolicy:
    """The five-cell decision table over the unit verdict and paired search.

    | unit \\ search | improved | flat        | regressed          |
    | passed         | promote  | archive     | reject (mismatch)  |
    | failed         | archive (mismatch) | reject | reject        |

    The unit contract is a cheap alignment check that the patch repaired the
    trace-diagnosed atomic problem; paired search is the real objective. When
    the two disagree, the candidate earns no promotion, but the disagreement
    itself is staged for the next Test Author to diagnose.
    """

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config

    def decide(self, evidence: EvidenceRecord) -> DecisionRecord:
        cfg = self.config
        if evidence.search_delta is None:
            return self._record(
                evidence,
                Decision.REJECT,
                "a mechanically valid candidate is missing paired search evidence",
                0.0,
            )
        delta = evidence.search_delta
        if unit_ok(evidence, cfg):
            if delta > cfg.min_search_delta:
                return self._record(
                    evidence,
                    Decision.PROMOTE,
                    "the unit contract passed and paired search improved",
                    min(1.0, 0.5 + delta),
                )
            if delta >= -cfg.noninferiority_margin:
                return self._record(
                    evidence,
                    Decision.ARCHIVE,
                    "the unit contract passed and paired search was noninferior; "
                    "kept as a record for later re-litigation",
                    0.5,
                )
            return self._record(
                evidence,
                Decision.REJECT,
                "the unit contract passed but paired search regressed: the UT "
                "design deviated from the search distribution",
                max(0.0, 1.0 + delta),
            )
        if delta > cfg.min_search_delta:
            return self._record(
                evidence,
                Decision.ARCHIVE,
                "paired search improved but the unit contract failed: the patch "
                "is kept as a record and the UT design needs diagnosis",
                0.5,
            )
        return self._record(
            evidence,
            Decision.REJECT,
            self._reject_reason(evidence),
            0.0,
        )

    def _reject_reason(self, evidence: EvidenceRecord) -> str:
        if not evidence.contract_passed:
            return "neither the unit contract nor paired search improved"
        if not evidence.preservation_passed:
            return (
                "the candidate broke a preserved contract and paired search "
                "did not improve"
            )
        return (
            "the candidate broke an incumbent-passing regression and paired "
            "search did not improve"
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
