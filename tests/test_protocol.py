from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceunit.calibration import (
    AlignmentCalibrator,
    CalibrationLabel,
    CalibrationObservation,
)
from traceunit.config import AlignmentConfig
from traceunit.io import sha256_file, sha256_tree
from traceunit.models import (
    BenchmarkEvaluation,
    PoolRole,
    PoolSliceRef,
    TaskOutcome,
)
from traceunit.protocol import (
    AlignmentCheckpointRunner,
    CalibrationSubject,
    RotationScheduler,
)


def _shard(ordinal: int) -> PoolSliceRef:
    return PoolSliceRef(
        slice_id=f"calibration-{ordinal}",
        role=PoolRole.CALIBRATION,
        manifest_path=f"/sealed/calibration-{ordinal}.json",
        manifest_sha256=f"sha-{ordinal}",
        cluster_ids=(f"cluster-{ordinal}",),
        ordinal=ordinal,
    )


def _subject(root: Path, candidate_id: str = "candidate-secret") -> CalibrationSubject:
    parent_source = root / "sources" / "parent-secret"
    candidate_source = root / "sources" / candidate_id
    parent_source.mkdir(parents=True, exist_ok=True)
    candidate_source.mkdir(parents=True, exist_ok=True)
    (parent_source / "behavior.txt").write_text("parent", encoding="utf-8")
    (candidate_source / "behavior.txt").write_text(candidate_id, encoding="utf-8")
    decision_path = root / "decisions" / f"{candidate_id}.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text('{"decision":"frozen"}\n', encoding="utf-8")
    return CalibrationSubject(
        candidate_id=candidate_id,
        parent_id="parent-secret",
        lineage_id="lineage-secret",
        candidate_source=str(candidate_source),
        parent_source=str(parent_source),
        candidate_source_sha256=sha256_tree(candidate_source),
        parent_source_sha256=sha256_tree(parent_source),
        decision_path=str(decision_path),
        decision_sha256=sha256_file(decision_path),
        family_keys=("planner.bridge",),
        unit_profile="unit+|bridge+|search+",
        stratum="unit+|search+",
    )


def _evaluation(
    candidate_id: str, score: float, shard: PoolSliceRef
) -> BenchmarkEvaluation:
    outcomes = tuple(
        TaskOutcome(
            task_id=f"task-secret-{index}",
            score=score,
            passed=bool(score),
            trace_id=f"trace-secret-{candidate_id}-{index}",
        )
        for index in range(3)
    )
    return BenchmarkEvaluation(
        evaluation_id=f"evaluation-{candidate_id}",
        benchmark="fake",
        candidate_id=candidate_id,
        split=shard.slice_id,
        score=score,
        passrate=score,
        cost=1.0,
        outcomes=outcomes,
        trace_path="/private/traces.jsonl",
        result_path="/private/result.json",
    )


def test_rotation_decision_is_read_only_until_checkpoint_freeze(tmp_path: Path) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "observations.jsonl")
    config = AlignmentConfig(
        min_candidates_per_checkpoint=1,
        max_candidates_per_checkpoint=4,
    )
    scheduler = RotationScheduler(config=config, calibrator=calibrator)
    runner = AlignmentCheckpointRunner(
        root=tmp_path / "protocol",
        config=config,
        calibrator=calibrator,
    )
    shards = (_shard(0), _shard(1))

    decision = scheduler.decide(
        pending=(_subject(tmp_path),),
        available_shards=runner.available_shards(shards),
    )

    assert decision.open_checkpoint
    assert decision.shard == shards[0]
    assert "initial_calibration" in decision.reasons
    assert runner.available_shards(shards) == shards
    assert not runner.shard_state_path.exists()

    runner.freeze(
        iteration=3,
        shard=decision.shard,
        subjects=(_subject(tmp_path),),
    )
    assert runner.available_shards(shards) == (shards[1],)


def test_checkpoint_is_delayed_and_rotates_single_use_shards(tmp_path: Path) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "observations.jsonl")
    config = AlignmentConfig(
        min_candidates_per_checkpoint=1,
        max_candidates_per_checkpoint=4,
        min_effective_n=1.0,
    )
    runner = AlignmentCheckpointRunner(
        root=tmp_path / "protocol",
        config=config,
        calibrator=calibrator,
    )
    first, second = _shard(0), _shard(1)
    subject = _subject(tmp_path)

    checkpoint = runner.freeze(
        iteration=7,
        shard=first,
        subjects=(subject,),
    )
    checkpoint_payload = checkpoint.to_dict()
    assert checkpoint.card_version_before == 0
    assert checkpoint_payload["effective_from_iteration"] == 8
    assert runner.available_shards((first, second)) == (second,)
    with pytest.raises(RuntimeError, match="already reserved"):
        runner.freeze(iteration=7, shard=first, subjects=(subject,))

    calls: list[tuple[str, str]] = []

    def evaluate(
        source: Path,
        candidate_id: str,
        shard: PoolSliceRef,
        tag: str,
    ) -> BenchmarkEvaluation:
        del source
        calls.append((candidate_id, tag))
        score = 1.0 if candidate_id == subject.candidate_id else 0.0
        return _evaluation(candidate_id, score, shard)

    observations = runner.run(
        checkpoint,
        evaluate=evaluate,
        noninferiority_margin=0.0,
        positive_effect=0.0,
    )

    assert len(observations) == 1
    assert observations[0].label is CalibrationLabel.POSITIVE
    assert calls == [
        ("parent-secret", "calibration_parent_e1"),
        ("candidate-secret", "calibration_candidate_e1"),
    ]
    assert runner.available_shards((first, second)) == (second,)
    with pytest.raises(RuntimeError, match="already spent"):
        runner.freeze(iteration=8, shard=first, subjects=(subject,))

    result_path = runner.checkpoint_root / checkpoint.checkpoint_id / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["effective_from_iteration"] == 8
    assert result["card_version_after"] == 1

    calls.clear()
    assert (
        runner.run(
            checkpoint,
            evaluate=evaluate,
            noninferiority_margin=0.0,
            positive_effect=0.0,
        )
        == ()
    )
    assert calls == []
    assert len(calibrator.observations) == 1

    next_checkpoint = runner.freeze(
        iteration=8,
        shard=second,
        subjects=(_subject(tmp_path, "future-candidate"),),
    )
    assert next_checkpoint.epoch == 2
    assert next_checkpoint.card_version_before == 1
    assert next_checkpoint.to_dict()["effective_from_iteration"] == 9
    assert runner.available_shards((first, second)) == ()


def test_protocol_public_cards_do_not_expose_private_observations(
    tmp_path: Path,
) -> None:
    calibrator = AlignmentCalibrator(tmp_path / "observations.jsonl")
    config = AlignmentConfig(min_effective_n=1.0)
    runner = AlignmentCheckpointRunner(
        root=tmp_path / "protocol",
        config=config,
        calibrator=calibrator,
    )
    calibrator.append(
        # Deliberately use secrets in every private field that is not a card key.
        CalibrationObservation(
            candidate_id="candidate-secret",
            parent_id="parent-secret",
            lineage_id="lineage-secret",
            epoch=1,
            shard_id="shard-secret",
            stratum="task-secret",
            family_keys=("planner.bridge",),
            unit_profile="unit+|bridge+|search+",
            paired_delta=0.123456789,
            uncertainty=0.01,
            label=CalibrationLabel.POSITIVE,
        )
    )
    path = tmp_path / "public-cards.json"

    runner.write_public_cards(path)

    public = path.read_text(encoding="utf-8")
    for secret in (
        "candidate-secret",
        "parent-secret",
        "lineage-secret",
        "shard-secret",
        "task-secret",
        "0.123456789",
        "paired_delta",
    ):
        assert secret not in public
