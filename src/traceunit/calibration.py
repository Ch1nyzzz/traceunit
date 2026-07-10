from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping

from traceunit.io import append_jsonl, read_jsonl


class CalibrationLabel(StrEnum):
    """Outcome of a paired natural-task calibration measurement."""

    POSITIVE = "positive"
    NONINFERIOR = "noninferior"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"


class TransferBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UncertaintyBand(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CalibrationStatus(StrEnum):
    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    CHALLENGED = "challenged"


class CalibrationTrigger(StrEnum):
    UNSEEN_FAMILY = "unseen_family"
    HIGH_UNCERTAINTY = "high_uncertainty"
    UNIT_SEARCH_DISAGREEMENT = "unit_search_disagreement"
    NOVEL_COMPOSITION = "novel_composition"


def classify_calibration_label(
    paired_delta: float,
    uncertainty: float,
    *,
    noninferiority_margin: float,
    positive_effect: float = 0.0,
) -> CalibrationLabel:
    """Classify an estimate and symmetric uncertainty radius into four states.

    uncertainty is a confidence radius supplied by the benchmark adapter, not
    a standard error interpreted by this module. The caller therefore controls
    the statistical coverage used by this deterministic classification rule.
    """

    _require_finite("paired_delta", paired_delta)
    _require_nonnegative_finite("uncertainty", uncertainty)
    _require_nonnegative_finite("noninferiority_margin", noninferiority_margin)
    _require_nonnegative_finite("positive_effect", positive_effect)

    lower = paired_delta - uncertainty
    upper = paired_delta + uncertainty
    if lower > positive_effect:
        return CalibrationLabel.POSITIVE
    if upper < -noninferiority_margin:
        return CalibrationLabel.NEGATIVE
    if lower >= -noninferiority_margin:
        return CalibrationLabel.NONINFERIOR
    return CalibrationLabel.INCONCLUSIVE


@dataclass(frozen=True)
class CalibrationObservation:
    """Private, append-only natural-transfer evidence for one candidate pair."""

    candidate_id: str
    parent_id: str
    lineage_id: str
    epoch: int
    shard_id: str
    stratum: str
    family_keys: tuple[str, ...]
    unit_profile: str
    paired_delta: float
    uncertainty: float
    label: CalibrationLabel

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "parent_id",
            "lineage_id",
            "shard_id",
            "stratum",
            "unit_profile",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.epoch < 1:
            raise ValueError("epoch must be at least 1")
        _require_finite("paired_delta", self.paired_delta)
        _require_nonnegative_finite("uncertainty", self.uncertainty)

        families = tuple(
            sorted({key.strip() for key in self.family_keys if key.strip()})
        )
        if not families:
            raise ValueError("family_keys must contain at least one non-empty key")
        object.__setattr__(self, "family_keys", families)
        object.__setattr__(self, "label", CalibrationLabel(self.label))

    @property
    def identity(self) -> tuple[str, str, int, str, str]:
        """Stable identity used to reject accidental duplicate appends."""

        return (
            self.candidate_id,
            self.parent_id,
            self.epoch,
            self.shard_id,
            self.stratum,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["family_keys"] = list(self.family_keys)
        value["label"] = self.label.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationObservation":
        return cls(
            candidate_id=str(value["candidate_id"]),
            parent_id=str(value["parent_id"]),
            lineage_id=str(value["lineage_id"]),
            epoch=int(value["epoch"]),
            shard_id=str(value["shard_id"]),
            stratum=str(value["stratum"]),
            family_keys=tuple(str(key) for key in value["family_keys"]),
            unit_profile=str(value["unit_profile"]),
            paired_delta=float(value["paired_delta"]),
            uncertainty=float(value["uncertainty"]),
            label=CalibrationLabel(str(value["label"])),
        )

    @classmethod
    def from_measurement(
        cls,
        *,
        candidate_id: str,
        parent_id: str,
        lineage_id: str,
        epoch: int,
        shard_id: str,
        stratum: str,
        family_keys: Iterable[str],
        unit_profile: str,
        paired_delta: float,
        uncertainty: float,
        noninferiority_margin: float,
        positive_effect: float = 0.0,
    ) -> "CalibrationObservation":
        return cls(
            candidate_id=candidate_id,
            parent_id=parent_id,
            lineage_id=lineage_id,
            epoch=epoch,
            shard_id=shard_id,
            stratum=stratum,
            family_keys=tuple(family_keys),
            unit_profile=unit_profile,
            paired_delta=paired_delta,
            uncertainty=uncertainty,
            label=classify_calibration_label(
                paired_delta,
                uncertainty,
                noninferiority_margin=noninferiority_margin,
                positive_effect=positive_effect,
            ),
        )


@dataclass(frozen=True)
class PublicCalibrationCard:
    """Sanitized feedback safe for the Test Author and Search Agent.

    Continuous posteriors, exact counts, candidates, lineages, shards, deltas,
    and task-level outcomes deliberately do not cross this boundary.
    """

    family_key: str
    unit_profile: str
    transfer_band: TransferBand
    uncertainty: UncertaintyBand
    support_bucket: str
    status: CalibrationStatus
    version: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "family_key": self.family_key,
            "unit_profile": self.unit_profile,
            "transfer_band": self.transfer_band.value,
            "uncertainty": self.uncertainty.value,
            "support_bucket": self.support_bucket,
            "status": self.status.value,
            "version": self.version,
        }


@dataclass(frozen=True)
class TriggerAssessment:
    triggered: bool
    reasons: tuple[CalibrationTrigger, ...]
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reasons": [reason.value for reason in self.reasons],
            "version": self.version,
        }


@dataclass
class _PosteriorEvidence:
    """Private Beta(1, 1) sufficient statistics for one family/context."""

    transfer_alpha: float = 1.0
    transfer_beta: float = 1.0
    harmful_alpha: float = 1.0
    harmful_beta: float = 1.0
    effective_n: float = 0.0

    def add(self, label: CalibrationLabel, weight: float) -> None:
        if label is CalibrationLabel.INCONCLUSIVE:
            return

        # NONINFERIOR is deliberately neutral, fractional evidence for the
        # positive-transfer target. It is not counted as a positive outcome.
        transfer_success = {
            CalibrationLabel.POSITIVE: 1.0,
            CalibrationLabel.NONINFERIOR: 0.5,
            CalibrationLabel.NEGATIVE: 0.0,
        }[label]
        harmful = float(label is CalibrationLabel.NEGATIVE)
        self.transfer_alpha += weight * transfer_success
        self.transfer_beta += weight * (1.0 - transfer_success)
        self.harmful_alpha += weight * harmful
        self.harmful_beta += weight * (1.0 - harmful)
        self.effective_n += weight

    @property
    def transfer_mean(self) -> float:
        return self.transfer_alpha / (self.transfer_alpha + self.transfer_beta)

    @property
    def harmful_mean(self) -> float:
        return self.harmful_alpha / (self.harmful_alpha + self.harmful_beta)

    @property
    def transfer_stddev(self) -> float:
        total = self.transfer_alpha + self.transfer_beta
        variance = (
            self.transfer_alpha * self.transfer_beta / (total * total * (total + 1.0))
        )
        return math.sqrt(variance)


class AlignmentCalibrator:
    """Append-only private calibration ledger with a sanitized public view.

    Every informative observation contributes total weight one across all its
    families, so a packet with many tags creates no more evidence than a packet
    with one tag. Each (family_key, unit_profile) cell has a Beta(1, 1) prior.
    """

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = ledger_path
        self._observations = tuple(
            CalibrationObservation.from_dict(row) for row in read_jsonl(ledger_path)
        )
        self._identities = {observation.identity for observation in self._observations}
        if len(self._identities) != len(self._observations):
            raise ValueError(f"duplicate calibration observations in {ledger_path}")

    @property
    def observations(self) -> tuple[CalibrationObservation, ...]:
        """Private observations; never pass this value to an agent prompt."""

        return self._observations

    @property
    def version(self) -> int:
        return max((observation.epoch for observation in self._observations), default=0)

    def append(self, observation: CalibrationObservation) -> None:
        """Append one observation without replacing or rewriting prior evidence."""

        if observation.identity in self._identities:
            raise ValueError(f"observation already exists: {observation.identity!r}")
        append_jsonl(self.ledger_path, observation.to_dict())
        self._observations = (*self._observations, observation)
        self._identities.add(observation.identity)

    def append_many(self, observations: Iterable[CalibrationObservation]) -> None:
        for observation in observations:
            self.append(observation)

    def public_cards(
        self,
        *,
        exclude_candidate: str | None = None,
        exclude_lineage: str | None = None,
        min_effective_n: float = 1.0,
    ) -> tuple[PublicCalibrationCard, ...]:
        """Return discrete, privacy-preserving calibration summaries.

        Exclusions implement delayed or leave-one-lineage-out feedback. Cells
        below min_effective_n are omitted rather than exposing exact support.
        Version names the full ledger epoch even when an exclusion changes view.
        """

        _require_nonnegative_finite("min_effective_n", min_effective_n)
        evidence = self._aggregate(
            observation
            for observation in self._observations
            if observation.candidate_id != exclude_candidate
            and observation.lineage_id != exclude_lineage
        )
        cards: list[PublicCalibrationCard] = []
        for (family_key, unit_profile), posterior in sorted(evidence.items()):
            if posterior.effective_n < min_effective_n:
                continue
            transfer_band = _transfer_band(posterior.transfer_mean)
            uncertainty = _uncertainty_band(posterior.transfer_stddev)
            if (
                posterior.harmful_mean >= 2.0 / 3.0
                and uncertainty is not UncertaintyBand.HIGH
            ):
                status = CalibrationStatus.CHALLENGED
            elif (
                transfer_band is TransferBand.HIGH
                and uncertainty is not UncertaintyBand.HIGH
            ):
                status = CalibrationStatus.SUPPORTED
            else:
                status = CalibrationStatus.UNCERTAIN
            cards.append(
                PublicCalibrationCard(
                    family_key=family_key,
                    unit_profile=unit_profile,
                    transfer_band=transfer_band,
                    uncertainty=uncertainty,
                    support_bucket=_support_bucket(posterior.effective_n),
                    status=status,
                    version=self.version,
                )
            )
        return tuple(cards)

    def assess_triggers(
        self,
        *,
        family_keys: Iterable[str],
        unit_profile: str,
        unit_positive: bool | None = None,
        search_positive: bool | None = None,
        composition_signature: str | None = None,
        known_composition_signatures: Collection[str] = (),
    ) -> TriggerAssessment:
        """Assess rotation triggers without reserving or consuming a shard."""

        requested_families = tuple(
            sorted({key.strip() for key in family_keys if key.strip()})
        )
        if not requested_families:
            raise ValueError("family_keys must contain at least one non-empty key")
        if not unit_profile.strip():
            raise ValueError("unit_profile must not be empty")

        known_families = {
            family_key
            for observation in self._observations
            for family_key in observation.family_keys
        }
        reasons: set[CalibrationTrigger] = set()
        seen_requested = [
            family_key
            for family_key in requested_families
            if family_key in known_families
        ]
        if len(seen_requested) != len(requested_families):
            reasons.add(CalibrationTrigger.UNSEEN_FAMILY)

        cards = {
            (card.family_key, card.unit_profile): card
            for card in self.public_cards(min_effective_n=0.0)
        }
        if any(
            (family_key, unit_profile) not in cards
            or cards[(family_key, unit_profile)].uncertainty is UncertaintyBand.HIGH
            for family_key in seen_requested
        ):
            reasons.add(CalibrationTrigger.HIGH_UNCERTAINTY)

        if (
            unit_positive is not None
            and search_positive is not None
            and unit_positive != search_positive
        ):
            reasons.add(CalibrationTrigger.UNIT_SEARCH_DISAGREEMENT)

        if (
            composition_signature is not None
            and composition_signature not in known_composition_signatures
        ):
            reasons.add(CalibrationTrigger.NOVEL_COMPOSITION)

        ordered = tuple(reason for reason in CalibrationTrigger if reason in reasons)
        return TriggerAssessment(
            triggered=bool(ordered),
            reasons=ordered,
            version=self.version,
        )

    @staticmethod
    def _aggregate(
        observations: Iterable[CalibrationObservation],
    ) -> dict[tuple[str, str], _PosteriorEvidence]:
        result: dict[tuple[str, str], _PosteriorEvidence] = {}
        for observation in observations:
            family_weight = 1.0 / len(observation.family_keys)
            for family_key in observation.family_keys:
                posterior = result.setdefault(
                    (family_key, observation.unit_profile), _PosteriorEvidence()
                )
                posterior.add(observation.label, family_weight)
        return result


def _transfer_band(mean: float) -> TransferBand:
    if mean >= 2.0 / 3.0:
        return TransferBand.HIGH
    if mean <= 1.0 / 3.0:
        return TransferBand.LOW
    return TransferBand.MEDIUM


def _uncertainty_band(stddev: float) -> UncertaintyBand:
    if stddev <= 0.12:
        return UncertaintyBand.LOW
    if stddev <= 0.20:
        return UncertaintyBand.MEDIUM
    return UncertaintyBand.HIGH


def _support_bucket(effective_n: float) -> str:
    if effective_n < 1.0:
        return "<1"
    if effective_n < 3.0:
        return "1-2"
    if effective_n < 6.0:
        return "3-5"
    if effective_n < 10.0:
        return "6-9"
    return "10+"


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
