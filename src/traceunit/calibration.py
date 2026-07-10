from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from traceunit.io import read_json, write_json
from traceunit.models import DecisionRecord, EvidenceRecord


@dataclass
class CalibrationBin:
    count: int = 0
    audit_positive: int = 0
    audit_noninferior: int = 0
    promotion_correct: int = 0

    def to_dict(self) -> dict[str, Any]:
        # Beta(1,1) smoothing avoids an unjustified 0/1 estimate early on.
        return {
            "count": self.count,
            "audit_positive": self.audit_positive,
            "audit_noninferior": self.audit_noninferior,
            "promotion_correct": self.promotion_correct,
            "p_audit_positive": (self.audit_positive + 1) / (self.count + 2),
            "p_audit_noninferior": (self.audit_noninferior + 1) / (self.count + 2),
        }


@dataclass
class CalibrationState:
    version: int = 1
    bins: dict[str, CalibrationBin] = field(default_factory=dict)
    family_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)


class CrossLayerCalibrator:
    """Small online calibration ledger, intentionally not a learned reward model."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = self._load()

    def update(
        self,
        *,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
        family_ids: set[str],
    ) -> None:
        key = self._bin_key(evidence)
        record = {
            "iteration": evidence.iteration,
            "candidate_id": evidence.candidate_id,
            "bin": key,
            "evidence": evidence.to_dict(),
            "decision": decision.decision.value,
            "families": sorted(family_ids),
        }
        self.state.records = [
            item
            for item in self.state.records
            if not (
                int(item.get("iteration") or -1) == evidence.iteration
                and str(item.get("candidate_id") or "") == evidence.candidate_id
            )
        ]
        self.state.records.append(record)
        self.state.records.sort(
            key=lambda item: (
                int(item.get("iteration") or 0),
                str(item.get("candidate_id") or ""),
            )
        )
        self._recompute()
        self.save()

    def _recompute(self) -> None:
        self.state.bins = {}
        self.state.family_stats = {}
        for record in self.state.records:
            values = dict(record.get("evidence") or {})
            audit_delta = values.get("audit_delta")
            if audit_delta is None:
                continue
            audit_delta = float(audit_delta)
            key = str(record.get("bin") or "unknown")
            bin_stats = self.state.bins.setdefault(key, CalibrationBin())
            bin_stats.count += 1
            bin_stats.audit_positive += int(audit_delta > 0)
            bin_stats.audit_noninferior += int(audit_delta >= 0)
            bin_stats.promotion_correct += int(
                record.get("decision") == "promote" and audit_delta > 0
            )
            unit_positive = (
                float(values.get("public_gain") or 0.0) > 0
                and float(values.get("hidden_gain") or 0.0) > 0
            )
            for family_id in record.get("families") or []:
                stats = self.state.family_stats.setdefault(
                    str(family_id),
                    {"audited": 0, "audit_positive": 0, "unit_positive": 0},
                )
                stats["audited"] += 1
                stats["unit_positive"] += int(unit_positive)
                stats["audit_positive"] += int(audit_delta > 0)

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.state.version,
            "bins": {key: value.to_dict() for key, value in self.state.bins.items()},
            "family_stats": self.state.family_stats,
            "records": self.state.records,
            "cross_level_metrics": self._cross_level_metrics(),
        }

    def _cross_level_metrics(self) -> dict[str, Any]:
        audited = [
            record
            for record in self.state.records
            if (record.get("evidence") or {}).get("audit_delta") is not None
        ]
        if not audited:
            return {
                "n": 0,
                "target": "audit_delta > 0",
                "baseline_features": ["diagnostic_delta_sign"],
                "proxy_features": [
                    "diagnostic_delta_sign",
                    "public_gain_sign",
                    "hidden_gain_sign",
                ],
                "conditional_information_bits": None,
            }

        labels = [
            int(float((record.get("evidence") or {}).get("audit_delta")) > 0)
            for record in audited
        ]

        def train_key(record: Mapping[str, Any]) -> str:
            evidence = dict(record.get("evidence") or {})
            delta = evidence.get("diagnostic_delta")
            if delta is None:
                return "train?"
            return "train+" if float(delta) > 0 else "train0-"

        def proxy_key(record: Mapping[str, Any]) -> str:
            evidence = dict(record.get("evidence") or {})
            public = float(evidence.get("public_gain") or 0.0) > 0
            hidden = float(evidence.get("hidden_gain") or 0.0) > 0
            unit = "unit+" if public and hidden else "unit-"
            return f"{train_key(record)}|{unit}"

        def loo_predictions(key_fn: Any) -> list[float]:
            totals: dict[str, list[int]] = {}
            for record, label in zip(audited, labels, strict=True):
                key = key_fn(record)
                stats = totals.setdefault(key, [0, 0])
                stats[0] += 1
                stats[1] += label
            predictions: list[float] = []
            for record, label in zip(audited, labels, strict=True):
                count, positives = totals[key_fn(record)]
                predictions.append((positives - label + 1) / (count - 1 + 2))
            return predictions

        def scores(predictions: list[float]) -> tuple[float, float]:
            log_loss = -sum(
                label * math.log(max(1e-9, prediction))
                + (1 - label) * math.log(max(1e-9, 1 - prediction))
                for label, prediction in zip(labels, predictions, strict=True)
            ) / len(labels)
            brier = sum(
                (prediction - label) ** 2
                for label, prediction in zip(labels, predictions, strict=True)
            ) / len(labels)
            return log_loss, brier

        baseline_log_loss, baseline_brier = scores(loo_predictions(train_key))
        proxy_log_loss, proxy_brier = scores(loo_predictions(proxy_key))
        promoted = [
            (record, label)
            for record, label in zip(audited, labels, strict=True)
            if record.get("decision") == "promote"
        ]
        return {
            "n": len(audited),
            "target": "audit_delta > 0",
            "protocol": "leave_one_edit_out_beta_1_1",
            "baseline_features": ["diagnostic_delta_sign"],
            "proxy_features": [
                "diagnostic_delta_sign",
                "public_gain_sign",
                "hidden_gain_sign",
            ],
            "baseline_log_loss": baseline_log_loss,
            "proxy_log_loss": proxy_log_loss,
            "baseline_brier": baseline_brier,
            "proxy_brier": proxy_brier,
            "conditional_information_bits": (baseline_log_loss - proxy_log_loss)
            / math.log(2),
            "promotions": len(promoted),
            "false_promotions": sum(1 for _, label in promoted if not label),
        }

    def save(self) -> None:
        write_json(self.path, self.summary())

    def _load(self) -> CalibrationState:
        if not self.path.is_file():
            return CalibrationState()
        raw = read_json(self.path)
        bins = {
            str(key): CalibrationBin(
                count=int(value.get("count") or 0),
                audit_positive=int(value.get("audit_positive") or 0),
                audit_noninferior=int(value.get("audit_noninferior") or 0),
                promotion_correct=int(value.get("promotion_correct") or 0),
            )
            for key, value in dict(raw.get("bins") or {}).items()
        }
        return CalibrationState(
            version=int(raw.get("version") or 1),
            bins=bins,
            family_stats={
                str(key): {str(k): int(v) for k, v in dict(value).items()}
                for key, value in dict(raw.get("family_stats") or {}).items()
            },
            records=list(raw.get("records") or []),
        )

    @staticmethod
    def _bin_key(evidence: EvidenceRecord) -> str:
        unit = (
            "unit+"
            if evidence.public_gain > 0 and evidence.hidden_gain > 0
            else "unit-"
        )
        train = (
            "diag?"
            if evidence.diagnostic_delta is None
            else ("diag+" if evidence.diagnostic_delta > 0 else "diag0-")
        )
        bridge = "bridge+" if evidence.bridge_gain > 0 else "bridge0"
        return f"{unit}|{bridge}|{train}"
