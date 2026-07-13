from __future__ import annotations

from typing import Any

from traceunit.config import DecisionConfig
from traceunit.models import Decision, DecisionRecord, EvidenceRecord

ARCHIVE_UNIT_PASSED_SEARCH_FLAT = "unit_passed_search_flat"
ARCHIVE_SEARCH_IMPROVED_UNIT_FAILED = "search_improved_unit_failed"


def unit_ok(evidence: EvidenceRecord, config: DecisionConfig) -> bool:
    """The complete unit verdict from the capability battery: the diagnosed
    capability improved and no other capability was damaged."""

    del config  # collateral tolerance is applied when the evidence is built
    return evidence.target_improved and evidence.collateral_ok


def _clears_margin(delta: float, config: DecisionConfig) -> bool:
    return delta > 0 and delta >= config.noninferiority_margin


def search_improved(evidence: EvidenceRecord, config: DecisionConfig) -> bool:
    """Improvement means clearing the noise margin, not merely being positive.

    On a 45-task pool with one repeat, a one- or two-task swing is ordinary
    nondeterminism; treating it as signal floods the mismatch channel and
    dilutes the Test Author's attention.
    """

    if evidence.search_delta is None:
        return False
    return _clears_margin(evidence.search_delta, config)


def search_regressed(evidence: EvidenceRecord, config: DecisionConfig) -> bool:
    if evidence.search_delta is None:
        return False
    return evidence.search_delta < -config.noninferiority_margin


def confirmation_delta(evidence: EvidenceRecord) -> float | None:
    """The paired delta of the independent confirmation re-evaluation, if the
    evaluator ran one for this candidate."""

    search: dict[str, Any] = dict(evidence.metadata.get("search") or {})
    confirmation = search.get("confirmation")
    if not isinstance(confirmation, dict):
        return None
    value = confirmation.get("search_delta")
    return None if value is None else float(value)


def is_mismatch(evidence: EvidenceRecord, config: DecisionConfig) -> bool:
    """Battery verdict and paired search disagree beyond noise: the battery
    deviates from the search distribution (unit passed, search regressed) or
    missed the mechanism (search improved, unit failed). Either way the next
    Test Author must diagnose it before updating the battery. A search
    improvement that failed its confirmation re-run was noise, not a
    disagreement."""

    if evidence.search_delta is None:
        return False
    if unit_ok(evidence, config):
        return search_regressed(evidence, config)
    if not search_improved(evidence, config):
        return False
    confirmed = confirmation_delta(evidence)
    if confirmed is not None and not _clears_margin(confirmed, config):
        return False
    return True


def archive_kind(evidence: EvidenceRecord, config: DecisionConfig) -> str | None:
    if evidence.search_delta is None:
        return None
    if unit_ok(evidence, config):
        if not search_improved(evidence, config) and not search_regressed(
            evidence, config
        ):
            return ARCHIVE_UNIT_PASSED_SEARCH_FLAT
        return None
    if search_improved(evidence, config):
        return ARCHIVE_SEARCH_IMPROVED_UNIT_FAILED
    return None


class DecisionPolicy:
    """The five-cell decision table over the battery verdict and paired search.

    | unit \\ search | improved | flat        | regressed          |
    | passed         | promote  | archive     | reject (mismatch)  |
    | failed         | confirm -> promote / archive (mismatch) | reject | reject |

    The capability battery is a cheap proxy for the benchmark's capability
    requirements; paired search is the real objective. A battery miss must not
    become a permanent loss: when search clearly improves without battery
    certification, the evaluator re-evaluates the candidate once, and a
    confirmed improvement promotes while the disagreement is still staged for
    the next Test Author to diagnose. Search boundaries use the noise margin:
    a sub-margin delta is flat, not signal.
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
            if search_improved(evidence, cfg):
                return self._record(
                    evidence,
                    Decision.PROMOTE,
                    "the target capability improved on the battery and paired "
                    "search improved",
                    min(1.0, 0.5 + delta),
                )
            if not search_regressed(evidence, cfg):
                return self._record(
                    evidence,
                    Decision.ARCHIVE,
                    "the target capability improved on the battery while paired "
                    "search was noninferior; kept as a record for later "
                    "re-litigation",
                    0.5,
                )
            return self._record(
                evidence,
                Decision.REJECT,
                "the battery certified the edit but paired search regressed: "
                "the battery deviates from the search distribution",
                max(0.0, 1.0 + delta),
            )
        if search_improved(evidence, cfg):
            return self._decide_uncertified_improvement(evidence)
        return self._record(
            evidence,
            Decision.REJECT,
            self._reject_reason(evidence),
            0.0,
        )

    def _decide_uncertified_improvement(
        self, evidence: EvidenceRecord
    ) -> DecisionRecord:
        """Search cleared the margin but the battery did not certify.

        Without a confirmation re-run the candidate is archived (the evaluator
        runs the confirmation and decides again). A confirmed improvement
        promotes - the battery missed the mechanism and must catch up via the
        staged mismatch; an unconfirmed one was noise and stays a record.
        """

        confirmed = confirmation_delta(evidence)
        if confirmed is None:
            return self._record(
                evidence,
                Decision.ARCHIVE,
                "paired search improved but the battery did not certify the "
                "edit: the patch is kept as a record and the battery needs "
                "diagnosis",
                0.5,
            )
        if _clears_margin(confirmed, self.config):
            return self._record(
                evidence,
                Decision.PROMOTE,
                "paired search improvement held in an independent confirmation "
                "re-evaluation: the battery missed the mechanism and the "
                "mismatch is staged for diagnosis",
                min(1.0, 0.5 + min(evidence.search_delta or 0.0, confirmed)),
            )
        return self._record(
            evidence,
            Decision.ARCHIVE,
            "paired search improvement did not survive the confirmation "
            "re-evaluation: recorded as probable noise",
            0.3,
        )

    @staticmethod
    def _reject_reason(evidence: EvidenceRecord) -> str:
        if not evidence.target_improved:
            return (
                "neither the target capability nor paired search improved"
            )
        return (
            "the edit damaged other capabilities on the battery and paired "
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
