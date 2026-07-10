from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceunit.calibration import (
    AlignmentCalibrator,
    CalibrationLabel,
    CalibrationObservation,
    CalibrationStatus,
    CalibrationTrigger,
    TransferBand,
    classify_calibration_label,
)


def _observation(
    candidate_id: str,
    *,
    parent_id: str = "parent",
    lineage_id: str = "lineage",
    epoch: int = 1,
    shard_id: str = "shard-1",
    stratum: str = "default",
    family_keys: tuple[str, ...] = ("planner.bridge",),
    unit_profile: str = "unit+|bridge+|search-flat",
    paired_delta: float = 0.1,
    uncertainty: float = 0.01,
    label: CalibrationLabel = CalibrationLabel.POSITIVE,
) -> CalibrationObservation:
    return CalibrationObservation(
        candidate_id=candidate_id,
        parent_id=parent_id,
        lineage_id=lineage_id,
        epoch=epoch,
        shard_id=shard_id,
        stratum=stratum,
        family_keys=family_keys,
        unit_profile=unit_profile,
        paired_delta=paired_delta,
        uncertainty=uncertainty,
        label=label,
    )


def test_four_state_label_classification() -> None:
    def classify(delta: float, uncertainty: float) -> CalibrationLabel:
        return classify_calibration_label(
            delta,
            uncertainty,
            noninferiority_margin=0.02,
        )

    assert classify(0.10, 0.01) is CalibrationLabel.POSITIVE
    assert classify(0.00, 0.01) is CalibrationLabel.NONINFERIOR
    assert classify(-0.10, 0.01) is CalibrationLabel.NEGATIVE
    assert classify(-0.02, 0.01) is CalibrationLabel.INCONCLUSIVE


def test_jsonl_ledger_is_append_only_and_reloadable(tmp_path: Path) -> None:
    path = tmp_path / "alignment-observations.jsonl"
    calibrator = AlignmentCalibrator(path)
    first = _observation("candidate-1")
    second = _observation(
        "candidate-2",
        epoch=2,
        shard_id="shard-2",
        label=CalibrationLabel.NONINFERIOR,
    )

    calibrator.append(first)
    first_line = path.read_text(encoding="utf-8")
    calibrator.append(second)

    assert path.read_text(encoding="utf-8").startswith(first_line)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert AlignmentCalibrator(path).observations == (first, second)
    assert calibrator.version == 2
    with pytest.raises(ValueError, match="already exists"):
        calibrator.append(first)


def test_packet_weight_is_shared_across_families(tmp_path: Path) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "ledger.jsonl")
    calibrator.append(
        _observation(
            "multi-family",
            family_keys=("family-a", "family-b"),
        )
    )

    # One packet contributes 1/2 to each family, not one full sample to both.
    assert calibrator.public_cards(min_effective_n=0.5)
    assert calibrator.public_cards(min_effective_n=0.51) == ()


def test_noninferior_is_neutral_soft_evidence_not_positive(tmp_path: Path) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "ledger.jsonl")
    for index in range(8):
        calibrator.append(
            _observation(
                f"candidate-{index}",
                shard_id=f"shard-{index}",
                label=CalibrationLabel.NONINFERIOR,
            )
        )

    card = calibrator.public_cards()[0]
    assert card.transfer_band is TransferBand.MEDIUM
    assert card.status is CalibrationStatus.UNCERTAIN


def test_public_cards_are_sanitized_and_support_exclusions(tmp_path: Path) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "ledger.jsonl")
    for index in range(3):
        calibrator.append(
            _observation(
                f"candidate-secret-{index}",
                parent_id="parent-secret",
                lineage_id="lineage-secret" if index == 0 else "lineage-public",
                shard_id=f"secret-shard-{index}",
                stratum="task-secret-natural-item",
                paired_delta=0.123456789,
            )
        )

    card = calibrator.public_cards()[0]
    public = card.to_dict()
    assert set(public) == {
        "family_key",
        "unit_profile",
        "transfer_band",
        "uncertainty",
        "support_bucket",
        "status",
        "version",
    }
    serialized = json.dumps(public)
    for forbidden in (
        "candidate-secret",
        "parent-secret",
        "lineage-secret",
        "secret-shard",
        "task-secret-natural-item",
        "paired_delta",
        "0.123456789",
        "effective_n",
    ):
        assert forbidden not in serialized

    assert (
        calibrator.public_cards(exclude_candidate="candidate-secret-0")[
            0
        ].support_bucket
        == "1-2"
    )
    assert (
        calibrator.public_cards(exclude_lineage="lineage-public")[0].support_bucket
        == "1-2"
    )
    assert (
        calibrator.public_cards(exclude_lineage="lineage-public", min_effective_n=2.0)
        == ()
    )


def test_positive_and_harmful_posteriors_drive_discrete_status(
    tmp_path: Path,
) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "ledger.jsonl")
    for index in range(3):
        calibrator.append(
            _observation(
                f"positive-{index}",
                shard_id=f"positive-shard-{index}",
            )
        )
        calibrator.append(
            _observation(
                f"negative-{index}",
                shard_id=f"negative-shard-{index}",
                family_keys=("retrieval.empty",),
                label=CalibrationLabel.NEGATIVE,
            )
        )

    cards = {card.family_key: card for card in calibrator.public_cards()}
    assert cards["planner.bridge"].status is CalibrationStatus.SUPPORTED
    assert cards["planner.bridge"].transfer_band is TransferBand.HIGH
    assert cards["retrieval.empty"].status is CalibrationStatus.CHALLENGED
    assert cards["retrieval.empty"].transfer_band is TransferBand.LOW


def test_trigger_assessment_is_read_only_and_reports_all_reasons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    calibrator = AlignmentCalibrator(path)
    calibrator.append(_observation("candidate-1"))
    before = path.read_bytes()

    assessment = calibrator.assess_triggers(
        family_keys=("planner.bridge", "unseen.family"),
        unit_profile="unit+|bridge+|search-flat",
        unit_positive=True,
        search_positive=False,
        composition_signature="archive:a+b",
        known_composition_signatures={"archive:a"},
    )

    assert assessment.triggered
    assert assessment.reasons == (
        CalibrationTrigger.UNSEEN_FAMILY,
        CalibrationTrigger.HIGH_UNCERTAINTY,
        CalibrationTrigger.UNIT_SEARCH_DISAGREEMENT,
        CalibrationTrigger.NOVEL_COMPOSITION,
    )
    assert path.read_bytes() == before
    assert len(calibrator.observations) == 1


def test_delayed_card_view_can_exclude_current_candidate(tmp_path: Path) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "ledger.jsonl")
    current = _observation("current-candidate")
    calibrator.append(current)

    assert calibrator.public_cards()
    assert calibrator.public_cards(exclude_candidate=current.candidate_id) == ()
