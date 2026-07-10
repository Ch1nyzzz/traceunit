from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from traceunit.calibration import (
    AlignmentCalibrator,
    CalibrationObservation,
    CalibrationTrigger,
)
from traceunit.config import AlignmentConfig
from traceunit.io import read_json, sha256_file, sha256_tree, write_json
from traceunit.models import BenchmarkEvaluation, PoolSliceRef
from traceunit.paired import paired_task_differences, paired_uncertainty


@dataclass(frozen=True)
class CalibrationSubject:
    candidate_id: str
    parent_id: str
    lineage_id: str
    candidate_source: str
    parent_source: str
    candidate_source_sha256: str
    parent_source_sha256: str
    decision_path: str
    decision_sha256: str
    family_keys: tuple[str, ...]
    unit_profile: str
    stratum: str
    composition_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["family_keys"] = list(self.family_keys)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationSubject":
        return cls(
            candidate_id=str(value["candidate_id"]),
            parent_id=str(value["parent_id"]),
            lineage_id=str(value["lineage_id"]),
            candidate_source=str(value["candidate_source"]),
            parent_source=str(value["parent_source"]),
            candidate_source_sha256=str(value["candidate_source_sha256"]),
            parent_source_sha256=str(value["parent_source_sha256"]),
            decision_path=str(value["decision_path"]),
            decision_sha256=str(value["decision_sha256"]),
            family_keys=tuple(str(item) for item in value.get("family_keys") or []),
            unit_profile=str(value["unit_profile"]),
            stratum=str(value["stratum"]),
            composition_signature=str(value.get("composition_signature") or ""),
        )


@dataclass(frozen=True)
class CalibrationCheckpoint:
    checkpoint_id: str
    epoch: int
    created_after_iteration: int
    card_version_before: int
    shard: PoolSliceRef
    subjects: tuple[CalibrationSubject, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "epoch": self.epoch,
            "created_after_iteration": self.created_after_iteration,
            "card_version_before": self.card_version_before,
            "shard": self.shard.to_dict(),
            "subjects": [item.to_dict() for item in self.subjects],
            "effective_from_iteration": self.created_after_iteration + 1,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationCheckpoint":
        return cls(
            checkpoint_id=str(value["checkpoint_id"]),
            epoch=int(value["epoch"]),
            created_after_iteration=int(value["created_after_iteration"]),
            card_version_before=int(value["card_version_before"]),
            shard=PoolSliceRef.from_dict(dict(value["shard"])),
            subjects=tuple(
                CalibrationSubject.from_dict(item)
                for item in value.get("subjects") or []
            ),
        )


@dataclass(frozen=True)
class RotationDecision:
    open_checkpoint: bool
    reasons: tuple[str, ...]
    shard: PoolSliceRef | None


class RotationScheduler:
    """Open a fresh shard only when a predeclared proxy-drift trigger fires."""

    def __init__(
        self,
        *,
        config: AlignmentConfig,
        calibrator: AlignmentCalibrator,
    ) -> None:
        self.config = config
        self.calibrator = calibrator

    def decide(
        self,
        *,
        pending: tuple[CalibrationSubject, ...],
        available_shards: tuple[PoolSliceRef, ...],
        known_composition_signatures: Iterable[str] = (),
    ) -> RotationDecision:
        if not available_shards:
            return RotationDecision(False, (), None)
        if len(pending) < self.config.min_candidates_per_checkpoint:
            return RotationDecision(False, (), None)
        reasons: set[str] = set()
        if self.calibrator.version == 0:
            reasons.add("initial_calibration")
        for subject in pending:
            assessment = self.calibrator.assess_triggers(
                family_keys=subject.family_keys,
                unit_profile=subject.unit_profile,
                unit_positive=_unit_positive(subject.unit_profile),
                search_positive=_search_positive(subject.stratum),
                composition_signature=subject.composition_signature or None,
                known_composition_signatures=known_composition_signatures,
            )
            for reason in assessment.reasons:
                if _trigger_enabled(reason, self.config):
                    reasons.add(reason.value)
        if len(pending) >= self.config.max_candidates_per_checkpoint:
            reasons.add("pending_capacity")
        ordered = tuple(sorted(reasons))
        return RotationDecision(
            bool(ordered), ordered, available_shards[0] if ordered else None
        )


class AlignmentCheckpointRunner:
    """Freeze decisions first, then label them on one single-use natural shard."""

    def __init__(
        self,
        *,
        root: Path,
        config: AlignmentConfig,
        calibrator: AlignmentCalibrator,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.calibrator = calibrator
        self.checkpoint_root = self.root / "checkpoints"
        self.shard_state_path = self.root / "shards.json"

    def freeze(
        self,
        *,
        iteration: int,
        shard: PoolSliceRef,
        subjects: Iterable[CalibrationSubject],
    ) -> CalibrationCheckpoint:
        selected = tuple(subjects)[: self.config.max_candidates_per_checkpoint]
        if not selected:
            raise ValueError("a calibration checkpoint requires subjects")
        states = self._shard_states()
        status = states.get(shard.slice_id, "sealed")
        if status != "sealed":
            raise RuntimeError(
                f"calibration shard {shard.slice_id} is already {status}"
            )
        epoch = self.calibrator.version + 1
        identity = {
            "epoch": epoch,
            "created_after_iteration": iteration,
            "card_version_before": self.calibrator.version,
            "shard": shard.to_dict(),
            "subjects": [item.to_dict() for item in selected],
        }
        checkpoint_id = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        checkpoint = CalibrationCheckpoint(
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            created_after_iteration=iteration,
            card_version_before=self.calibrator.version,
            shard=shard,
            subjects=selected,
        )
        path = self.checkpoint_root / checkpoint_id / "checkpoint.json"
        write_json(path, checkpoint.to_dict())
        states[shard.slice_id] = "reserved"
        write_json(self.shard_state_path, states)
        return checkpoint

    def run(
        self,
        checkpoint: CalibrationCheckpoint,
        *,
        evaluate: Callable[[Path, str, PoolSliceRef, str], BenchmarkEvaluation],
        noninferiority_margin: float,
        positive_effect: float,
    ) -> tuple[CalibrationObservation, ...]:
        states = self._shard_states()
        status = states.get(checkpoint.shard.slice_id)
        result_path = self.checkpoint_root / checkpoint.checkpoint_id / "result.json"
        if status == "spent":
            if not result_path.is_file():
                raise RuntimeError(
                    f"spent calibration checkpoint has no result: {checkpoint.checkpoint_id}"
                )
            return ()
        if status != "reserved":
            raise RuntimeError(
                f"calibration shard {checkpoint.shard.slice_id} is not reserved"
            )
        for subject in checkpoint.subjects:
            if sha256_file(Path(subject.decision_path)) != subject.decision_sha256:
                raise RuntimeError(
                    f"frozen decision changed before calibration: {subject.candidate_id}"
                )
            if (
                _source_sha256(Path(subject.parent_source))
                != subject.parent_source_sha256
            ):
                raise RuntimeError(
                    f"parent source changed before calibration: {subject.parent_id}"
                )
            if (
                _source_sha256(Path(subject.candidate_source))
                != subject.candidate_source_sha256
            ):
                raise RuntimeError(
                    f"candidate source changed before calibration: {subject.candidate_id}"
                )
        observations: list[CalibrationObservation] = []
        existing = {item.identity for item in self.calibrator.observations}
        evaluations: dict[tuple[str, str], BenchmarkEvaluation] = {}

        def evaluate_once(
            source: str, candidate_id: str, tag: str
        ) -> BenchmarkEvaluation:
            key = (source, candidate_id)
            if key not in evaluations:
                evaluations[key] = evaluate(
                    Path(source), candidate_id, checkpoint.shard, tag
                )
            return evaluations[key]

        for subject in checkpoint.subjects:
            identity_prefix = (
                subject.candidate_id,
                subject.parent_id,
                checkpoint.epoch,
                checkpoint.shard.slice_id,
                subject.stratum,
            )
            parent = evaluate_once(
                subject.parent_source,
                subject.parent_id,
                f"calibration_parent_e{checkpoint.epoch}",
            )
            candidate = evaluate_once(
                subject.candidate_source,
                subject.candidate_id,
                f"calibration_candidate_e{checkpoint.epoch}",
            )
            if identity_prefix in existing:
                continue
            differences = paired_task_differences(parent, candidate)
            delta = statistics.fmean(differences) if differences else 0.0
            uncertainty = paired_uncertainty(differences)
            observation = CalibrationObservation.from_measurement(
                candidate_id=subject.candidate_id,
                parent_id=subject.parent_id,
                lineage_id=subject.lineage_id,
                epoch=checkpoint.epoch,
                shard_id=checkpoint.shard.slice_id,
                stratum=subject.stratum,
                family_keys=subject.family_keys,
                unit_profile=subject.unit_profile,
                paired_delta=delta,
                uncertainty=uncertainty,
                noninferiority_margin=noninferiority_margin,
                positive_effect=positive_effect,
            )
            self.calibrator.append(observation)
            observations.append(observation)
        checkpoint_observations = [
            item
            for item in self.calibrator.observations
            if item.epoch == checkpoint.epoch
            and item.shard_id == checkpoint.shard.slice_id
            and item.candidate_id
            in {subject.candidate_id for subject in checkpoint.subjects}
        ]
        write_json(
            result_path,
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "observation_count": len(checkpoint_observations),
                "card_version_after": self.calibrator.version,
                "effective_from_iteration": checkpoint.created_after_iteration + 1,
                "cost": sum(item.cost for item in evaluations.values()),
            },
        )
        states[checkpoint.shard.slice_id] = "spent"
        write_json(self.shard_state_path, states)
        return tuple(observations)

    def reserved_checkpoint(self) -> CalibrationCheckpoint | None:
        """Return the sole incomplete checkpoint, if a crash interrupted one."""

        reserved = {
            shard_id
            for shard_id, status in self._shard_states().items()
            if status == "reserved"
        }
        if not reserved:
            return None
        matches = []
        for path in sorted(self.checkpoint_root.glob("*/checkpoint.json")):
            checkpoint = CalibrationCheckpoint.from_dict(read_json(path))
            if checkpoint.shard.slice_id in reserved:
                matches.append(checkpoint)
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one reserved calibration checkpoint, found {len(matches)}"
            )
        return matches[0]

    def write_public_cards(self, path: Path) -> None:
        cards = self.calibrator.public_cards(
            min_effective_n=self.config.min_effective_n
        )
        write_json(
            path,
            {
                "version": self.calibrator.version,
                "cards": [card.to_dict() for card in cards],
            },
        )

    def available_shards(
        self, shards: Iterable[PoolSliceRef]
    ) -> tuple[PoolSliceRef, ...]:
        states = self._shard_states()
        return tuple(
            shard
            for shard in shards
            if states.get(shard.slice_id, "sealed") == "sealed"
        )

    def _shard_states(self) -> dict[str, str]:
        if not self.shard_state_path.is_file():
            return {}
        return {
            str(key): str(value)
            for key, value in read_json(self.shard_state_path).items()
        }


def decision_file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def _trigger_enabled(trigger: CalibrationTrigger, config: AlignmentConfig) -> bool:
    return {
        CalibrationTrigger.UNSEEN_FAMILY: config.trigger_on_new_family,
        CalibrationTrigger.HIGH_UNCERTAINTY: True,
        CalibrationTrigger.UNIT_SEARCH_DISAGREEMENT: config.trigger_on_disagreement,
        CalibrationTrigger.NOVEL_COMPOSITION: config.trigger_on_novel_composition,
    }[trigger]


def _unit_positive(profile: str) -> bool | None:
    if "unit+" in profile:
        return True
    if "unit-" in profile:
        return False
    return None


def _search_positive(stratum: str) -> bool | None:
    if "search+" in stratum:
        return True
    if "search-" in stratum or "search0" in stratum:
        return False
    return None


def _source_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sha256_tree(path)
