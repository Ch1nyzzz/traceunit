from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from traceunit.calibration import CalibrationLabel, CalibrationObservation
from traceunit.io import read_json, read_jsonl
from traceunit.models import EvidenceRecord


_BASELINE_FEATURES = (
    "search_delta",
    "search_missing",
)
_UNIT_FEATURES = (
    "public_gain",
    "hidden_gain",
    "bridge_gain",
    "regression_loss",
    "admission_score",
    "archive_replay_passed",
    "preservation_passed",
    "has_bridge",
    "composition_present",
    "family_count",
)


@dataclass(frozen=True)
class ProxyExample:
    run_dir: Path
    group_id: str
    observation: CalibrationObservation
    evidence: EvidenceRecord

    @property
    def target(self) -> int | None:
        if self.observation.label is CalibrationLabel.INCONCLUSIVE:
            return None
        return int(self.observation.label is CalibrationLabel.POSITIVE)


@dataclass(frozen=True)
class _LoadedCohort:
    run_dirs: tuple[Path, ...]
    examples: tuple[ProxyExample, ...]
    total_observations: int
    missing_evidence: int
    label_counts: dict[str, int]
    benchmark: str


@dataclass(frozen=True)
class _FeatureEncoder:
    names: tuple[str, ...]
    include_unit: bool

    @classmethod
    def fit(
        cls,
        examples: Sequence[ProxyExample],
        *,
        include_unit: bool,
        min_category_support: int,
    ) -> "_FeatureEncoder":
        names = list(_BASELINE_FEATURES)
        if include_unit:
            names.extend(_UNIT_FEATURES)
            profile_support = Counter(
                example.observation.unit_profile for example in examples
            )
            family_support = Counter(
                family
                for example in examples
                for family in example.observation.family_keys
            )
            names.extend(
                f"profile={profile}"
                for profile, support in sorted(profile_support.items())
                if support >= min_category_support
            )
            names.extend(
                f"family={family}"
                for family, support in sorted(family_support.items())
                if support >= min_category_support
            )
        return cls(names=tuple(names), include_unit=include_unit)

    def transform(self, example: ProxyExample) -> list[float]:
        raw = _raw_features(example)
        values = [float(raw.get(name, 0.0)) for name in self.names]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"non-finite proxy feature for {example.observation.candidate_id}"
            )
        return values


@dataclass(frozen=True)
class _LogisticModel:
    weights: tuple[float, ...]
    constant_probability: float | None = None

    def predict(self, row: Sequence[float]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability
        score = self.weights[0] + sum(
            weight * value for weight, value in zip(self.weights[1:], row)
        )
        return _sigmoid(score)


def analyze_proxy(
    run_dirs: Iterable[Path],
    *,
    min_train_examples: int = 4,
    min_category_support: int = 2,
    l2: float = 1.0,
    selection_thresholds: Sequence[float] = (0.1, 0.25, 0.5),
    audit_rate: float = 0.1,
    alignment_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate unit evidence as an out-of-sample natural-transfer proxy.

    Final results never enter features, labels, model fitting, thresholds, or
    predictions. If an already sealed final report exists, it is copied only
    into the separate optimization-effect section.
    """

    if min_train_examples < 1:
        raise ValueError("min_train_examples must be positive")
    if min_category_support < 1:
        raise ValueError("min_category_support must be positive")
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be a nonnegative finite value")
    thresholds = tuple(sorted(set(float(value) for value in selection_thresholds)))
    if not thresholds or any(
        not math.isfinite(value) or value <= 0.0 or value >= 1.0
        for value in thresholds
    ):
        raise ValueError("selection thresholds must be finite values between 0 and 1")
    if not math.isfinite(audit_rate) or not 0.0 <= audit_rate <= 1.0:
        raise ValueError("audit_rate must be between 0 and 1")
    if alignment_bins < 2:
        raise ValueError("alignment_bins must be at least 2")

    cohort = _load_cohort(run_dirs)
    analyzed = tuple(example for example in cohort.examples if example.target is not None)
    groups = sorted({example.group_id for example in analyzed})
    if len(groups) < 2:
        raise ValueError(
            "out-of-sample analysis requires at least two groups with informative "
            f"labels; found {len(groups)}"
        )
    for heldout_group in groups:
        train_size = sum(
            example.group_id != heldout_group for example in analyzed
        )
        if train_size < min_train_examples:
            raise ValueError(
                f"held-out group {heldout_group!r} leaves only {train_size} training "
                f"examples; require at least {min_train_examples}"
            )

    baseline, baseline_predictions = _evaluate_oof(
        analyzed,
        groups=groups,
        include_unit=False,
        min_category_support=min_category_support,
        l2=l2,
    )
    proxy, proxy_predictions = _evaluate_oof(
        analyzed,
        groups=groups,
        include_unit=True,
        min_category_support=min_category_support,
        l2=l2,
    )
    unit_predictions = [
        float(example.observation.unit_profile.split("|", 1)[0] == "unit+")
        for example in analyzed
    ]
    unit_gate = _metrics(analyzed, unit_predictions)
    unit_gate["label_cross_tab"] = _unit_label_cross_tab(cohort.examples)

    return {
        "analysis": "candidate_out_of_sample_proxy",
        "benchmark": cohort.benchmark,
        "split": {
            "method": "leave_one_group_out",
            "group_by": "lineage",
            "groups": groups,
            "folds": len(groups),
        },
        "target": {
            "positive": CalibrationLabel.POSITIVE.value,
            "negative_class": [
                CalibrationLabel.NONINFERIOR.value,
                CalibrationLabel.NEGATIVE.value,
            ],
            "excluded": [CalibrationLabel.INCONCLUSIVE.value],
        },
        "cohort": {
            "run_count": len({str(example.run_dir) for example in cohort.examples}),
            "total_calibration_observations": cohort.total_observations,
            "joined_with_unit_evidence": len(cohort.examples),
            "analyzed_informative": len(analyzed),
            "excluded_inconclusive": sum(
                example.target is None for example in cohort.examples
            ),
            "missing_evidence": cohort.missing_evidence,
            "label_counts": cohort.label_counts,
        },
        "model": {
            "kind": "l2_logistic_regression",
            "l2": l2,
            "min_category_support_per_training_fold": min_category_support,
            "baseline_features": list(_BASELINE_FEATURES),
            "proxy_numeric_features": list((*_BASELINE_FEATURES, *_UNIT_FEATURES)),
            "proxy_categorical_features": ["unit_profile", "family_key"],
            "vocabulary_fit": "training_fold_only",
        },
        "baseline_search_only": baseline,
        "proxy_search_plus_unit": proxy,
        "incremental_value": {
            "brier_reduction": baseline["brier"] - proxy["brier"],
            "log_loss_reduction": baseline["log_loss"] - proxy["log_loss"],
            "pairwise_accuracy_lift": _optional_difference(
                proxy["pairwise_accuracy"], baseline["pairwise_accuracy"]
            ),
            "selection_regret_reduction": _optional_difference(
                baseline["mean_selection_regret"],
                proxy["mean_selection_regret"],
            ),
        },
        "unit_gate_diagnostic": unit_gate,
        "proxy_alignment_curve": {
            "interpretation": (
                "OOF predicted positive-transfer probability versus observed "
                "positive-transfer frequency"
            ),
            "bins": alignment_bins,
            "baseline_search_only": _alignment_curve(
                analyzed, baseline_predictions, bins=alignment_bins
            ),
            "proxy_search_plus_unit": _alignment_curve(
                analyzed, proxy_predictions, bins=alignment_bins
            ),
        },
        "selective_full_evaluation": {
            "interpretation": (
                "candidates below threshold skip the full natural-task evaluation; "
                "audit_rate is the random audit probability among skipped candidates"
            ),
            "threshold_selection_warning": (
                "choose and freeze a threshold on an earlier cohort before using it "
                "for prospective decisions"
            ),
            "audit_rate": audit_rate,
            "baseline_search_only": _selective_evaluation_curve(
                analyzed,
                baseline_predictions,
                thresholds=thresholds,
                audit_rate=audit_rate,
            ),
            "proxy_search_plus_unit": _selective_evaluation_curve(
                analyzed,
                proxy_predictions,
                thresholds=thresholds,
                audit_rate=audit_rate,
            ),
            "frozen_unit_gate": _selective_evaluation_point(
                analyzed,
                unit_predictions,
                threshold=0.5,
                audit_rate=audit_rate,
            ),
        },
        "prediction_count_check": {
            "baseline": len(baseline_predictions),
            "proxy": len(proxy_predictions),
            "expected": len(analyzed),
        },
        "optimization_effect": _optimization_effect(cohort.run_dirs),
        "information_boundary": (
            "proxy_models_use_calibration_and_iteration_artifacts_only; "
            "sealed_final_is_reported_only_as_terminal_outcome"
        ),
    }


def _load_cohort(run_dirs: Iterable[Path]) -> _LoadedCohort:
    roots = tuple(dict.fromkeys(Path(path).resolve() for path in run_dirs))
    if not roots:
        raise ValueError("at least one run directory is required")

    examples: list[ProxyExample] = []
    label_counts: Counter[str] = Counter()
    total_observations = 0
    missing_evidence = 0
    benchmarks: set[str] = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {root}")
        benchmark = _benchmark_name(root)
        if benchmark:
            benchmarks.add(benchmark)
        evidence_by_candidate = _evidence_by_candidate(root)
        seen_candidates: set[str] = set()
        observations = [
            CalibrationObservation.from_dict(row)
            for row in read_jsonl(root / "calibration" / "private_observations.jsonl")
        ]
        total_observations += len(observations)
        for observation in observations:
            label_counts[observation.label.value] += 1
            if observation.candidate_id in seen_candidates:
                raise ValueError(
                    "candidate has multiple calibration observations in one run: "
                    f"{root}:{observation.candidate_id}"
                )
            seen_candidates.add(observation.candidate_id)
            evidence = evidence_by_candidate.get(observation.candidate_id)
            if evidence is None:
                missing_evidence += 1
                continue
            examples.append(
                ProxyExample(
                    run_dir=root,
                    group_id=observation.lineage_id,
                    observation=observation,
                    evidence=evidence,
                )
            )
    if len(benchmarks) > 1:
        raise ValueError(
            "proxy analysis cannot mix benchmarks: " + ", ".join(sorted(benchmarks))
        )
    if not examples:
        raise ValueError("no calibration observations could be joined to unit evidence")
    return _LoadedCohort(
        run_dirs=roots,
        examples=tuple(examples),
        total_observations=total_observations,
        missing_evidence=missing_evidence,
        label_counts=dict(sorted(label_counts.items())),
        benchmark=next(iter(benchmarks), "unknown"),
    )


def _benchmark_name(root: Path) -> str:
    state_path = root / "run_state.json"
    if state_path.is_file():
        return str(read_json(state_path).get("benchmark") or "")
    config_path = root / "config.snapshot.json"
    if config_path.is_file():
        return str((read_json(config_path).get("benchmark") or {}).get("name") or "")
    return ""


def _evidence_by_candidate(root: Path) -> dict[str, EvidenceRecord]:
    result: dict[str, EvidenceRecord] = {}
    for path in sorted((root / "iterations").glob("iter_*/evidence.json")):
        evidence = EvidenceRecord.from_dict(read_json(path))
        if evidence.candidate_id in result:
            raise ValueError(
                f"duplicate candidate evidence under {root}: {evidence.candidate_id}"
            )
        result[evidence.candidate_id] = evidence
    return result


def _raw_features(example: ProxyExample) -> dict[str, float]:
    evidence = example.evidence
    metadata = evidence.metadata
    result = {
        "search_delta": float(evidence.search_delta or 0.0),
        "search_missing": float(evidence.search_delta is None),
        "public_gain": evidence.public_gain,
        "hidden_gain": evidence.hidden_gain,
        "bridge_gain": evidence.bridge_gain,
        "regression_loss": evidence.regression_loss,
        "admission_score": evidence.admission_score,
        "archive_replay_passed": float(evidence.archive_replay_passed),
        "preservation_passed": float(evidence.preservation_passed),
        "has_bridge": float(bool(metadata.get("has_bridge"))),
        "composition_present": float(bool(metadata.get("composition_ids"))),
        "family_count": float(len(example.observation.family_keys)),
        f"profile={example.observation.unit_profile}": 1.0,
    }
    result.update(
        {f"family={family}": 1.0 for family in example.observation.family_keys}
    )
    return result


def _evaluate_oof(
    examples: Sequence[ProxyExample],
    *,
    groups: Sequence[str],
    include_unit: bool,
    min_category_support: int,
    l2: float,
) -> tuple[dict[str, Any], list[float]]:
    prediction_by_index: dict[int, float] = {}
    fold_reports: list[dict[str, Any]] = []
    for heldout_group in groups:
        train_indices = [
            index
            for index, example in enumerate(examples)
            if example.group_id != heldout_group
        ]
        test_indices = [
            index
            for index, example in enumerate(examples)
            if example.group_id == heldout_group
        ]
        training = [examples[index] for index in train_indices]
        encoder = _FeatureEncoder.fit(
            training,
            include_unit=include_unit,
            min_category_support=min_category_support,
        )
        train_rows = [encoder.transform(example) for example in training]
        train_targets = [_target(example) for example in training]
        model = _fit_logistic(train_rows, train_targets, l2=l2)
        fold_predictions = [
            model.predict(encoder.transform(examples[index])) for index in test_indices
        ]
        prediction_by_index.update(zip(test_indices, fold_predictions))
        fold_reports.append(
            {
                "heldout_group": heldout_group,
                "train_examples": len(train_indices),
                "test_examples": len(test_indices),
                "train_positive_rate": sum(train_targets) / len(train_targets),
                "feature_count": len(encoder.names),
                "constant_target_model": model.constant_probability is not None,
            }
        )
    predictions = [prediction_by_index[index] for index in range(len(examples))]
    report = _metrics(examples, predictions)
    report["folds"] = fold_reports
    return report, predictions


def _fit_logistic(
    rows: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    l2: float,
    learning_rate: float = 0.1,
    max_iterations: int = 4000,
) -> _LogisticModel:
    if not rows or len(rows) != len(targets):
        raise ValueError("logistic regression requires aligned non-empty rows and targets")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("feature rows have inconsistent widths")
    smoothed_rate = (sum(targets) + 1.0) / (len(targets) + 2.0)
    if len(set(targets)) == 1:
        return _LogisticModel(weights=(), constant_probability=smoothed_rate)

    weights = [math.log(smoothed_rate / (1.0 - smoothed_rate)), *([0.0] * width)]
    count = float(len(rows))
    for _ in range(max_iterations):
        gradients = [0.0] * len(weights)
        for row, target in zip(rows, targets):
            probability = _sigmoid(
                weights[0]
                + sum(weight * value for weight, value in zip(weights[1:], row))
            )
            error = probability - target
            gradients[0] += error
            for index, value in enumerate(row, start=1):
                gradients[index] += error * value
        gradients[0] /= count
        for index in range(1, len(gradients)):
            gradients[index] = gradients[index] / count + (l2 / count) * weights[index]
        if max(abs(value) for value in gradients) < 1e-8:
            break
        for index, gradient in enumerate(gradients):
            weights[index] -= learning_rate * gradient
    return _LogisticModel(weights=tuple(weights))


def _metrics(
    examples: Sequence[ProxyExample], predictions: Sequence[float]
) -> dict[str, Any]:
    if len(examples) != len(predictions) or not examples:
        raise ValueError("metrics require aligned non-empty examples and predictions")
    targets = [_target(example) for example in examples]
    clipped = [min(1.0 - 1e-12, max(1e-12, value)) for value in predictions]
    brier = sum((probability - target) ** 2 for probability, target in zip(clipped, targets)) / len(targets)
    log_loss = -sum(
        target * math.log(probability)
        + (1 - target) * math.log(1.0 - probability)
        for probability, target in zip(clipped, targets)
    ) / len(targets)
    predicted = [int(probability >= 0.5) for probability in clipped]
    tp = sum(actual == 1 and guess == 1 for actual, guess in zip(targets, predicted))
    tn = sum(actual == 0 and guess == 0 for actual, guess in zip(targets, predicted))
    fp = sum(actual == 0 and guess == 1 for actual, guess in zip(targets, predicted))
    fn = sum(actual == 1 and guess == 0 for actual, guess in zip(targets, predicted))
    return {
        "examples": len(examples),
        "positive_rate": sum(targets) / len(targets),
        "brier": brier,
        "log_loss": log_loss,
        "accuracy_at_0_5": (tp + tn) / len(targets),
        "precision_at_0_5": _safe_ratio(tp, tp + fp),
        "recall_at_0_5": _safe_ratio(tp, tp + fn),
        "false_positive_rate_at_0_5": _safe_ratio(fp, fp + tn),
        "false_negative_rate_at_0_5": _safe_ratio(fn, fn + tp),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "pairwise_accuracy": _pairwise_accuracy(examples, clipped),
        "mean_selection_regret": _mean_selection_regret(examples, clipped),
    }


def _pairwise_accuracy(
    examples: Sequence[ProxyExample], predictions: Sequence[float]
) -> float | None:
    grouped: dict[str, list[tuple[ProxyExample, float]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions):
        grouped[example.group_id].append((example, prediction))
    correct = 0.0
    pairs = 0
    for values in grouped.values():
        for left_index, (left, left_prediction) in enumerate(values):
            for right, right_prediction in values[left_index + 1 :]:
                left_delta = left.observation.paired_delta
                right_delta = right.observation.paired_delta
                if left_delta == right_delta:
                    continue
                pairs += 1
                expected = left_delta > right_delta
                if left_prediction == right_prediction:
                    correct += 0.5
                elif (left_prediction > right_prediction) == expected:
                    correct += 1.0
    return correct / pairs if pairs else None


def _mean_selection_regret(
    examples: Sequence[ProxyExample], predictions: Sequence[float]
) -> float | None:
    grouped: dict[str, list[tuple[ProxyExample, float]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions):
        grouped[example.group_id].append((example, prediction))
    regrets: list[float] = []
    for values in grouped.values():
        if len(values) < 2:
            continue
        oracle_delta = max(item.observation.paired_delta for item, _ in values)
        chosen, _ = max(
            values,
            key=lambda item: (item[1], item[0].observation.candidate_id),
        )
        regrets.append(max(0.0, oracle_delta - chosen.observation.paired_delta))
    return sum(regrets) / len(regrets) if regrets else None


def _unit_label_cross_tab(examples: Sequence[ProxyExample]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        unit = example.observation.unit_profile.split("|", 1)[0]
        table[unit][example.observation.label.value] += 1
    return {
        unit: dict(sorted(counts.items())) for unit, counts in sorted(table.items())
    }


def _selective_evaluation_curve(
    examples: Sequence[ProxyExample],
    predictions: Sequence[float],
    *,
    thresholds: Sequence[float],
    audit_rate: float,
) -> list[dict[str, float | int | None]]:
    return [
        _selective_evaluation_point(
            examples,
            predictions,
            threshold=threshold,
            audit_rate=audit_rate,
        )
        for threshold in thresholds
    ]


def _selective_evaluation_point(
    examples: Sequence[ProxyExample],
    predictions: Sequence[float],
    *,
    threshold: float,
    audit_rate: float,
) -> dict[str, float | int | None]:
    if len(examples) != len(predictions) or not examples:
        raise ValueError(
            "selective evaluation requires aligned non-empty examples and predictions"
        )
    selected = [prediction >= threshold for prediction in predictions]
    targets = [_target(example) for example in examples]
    selected_count = sum(selected)
    skipped_count = len(examples) - selected_count
    positives = sum(targets)
    selected_positives = sum(
        choose and target for choose, target in zip(selected, targets)
    )
    strict_negative = [
        example.observation.label is CalibrationLabel.NEGATIVE for example in examples
    ]
    strict_negative_count = sum(strict_negative)
    skipped_strict_negative = sum(
        (not choose) and negative
        for choose, negative in zip(selected, strict_negative)
    )
    expected_audited = audit_rate * skipped_count
    expected_positive_coverage = selected_positives + audit_rate * (
        positives - selected_positives
    )
    expected_evaluations = selected_count + expected_audited
    return {
        "threshold": threshold,
        "candidates": len(examples),
        "selected_without_audit": selected_count,
        "skipped_without_audit": skipped_count,
        "full_evaluation_rate_without_audit": selected_count / len(examples),
        "avoided_full_evaluation_rate_without_audit": skipped_count / len(examples),
        "positive_recall_without_audit": _safe_ratio(
            selected_positives, positives
        ),
        "strict_negative_skip_rate_without_audit": _safe_ratio(
            skipped_strict_negative, strict_negative_count
        ),
        "expected_full_evaluation_rate_with_audit": (
            expected_evaluations / len(examples)
        ),
        "expected_avoided_full_evaluation_rate_with_audit": (
            1.0 - expected_evaluations / len(examples)
        ),
        "expected_positive_label_coverage_with_audit": (
            expected_positive_coverage / positives if positives else None
        ),
    }


def _alignment_curve(
    examples: Sequence[ProxyExample],
    predictions: Sequence[float],
    *,
    bins: int,
) -> dict[str, Any]:
    if len(examples) != len(predictions) or not examples:
        raise ValueError("alignment curve requires aligned non-empty data")
    grouped: list[list[tuple[ProxyExample, float]]] = [[] for _ in range(bins)]
    for example, prediction in zip(examples, predictions):
        probability = min(1.0, max(0.0, prediction))
        index = min(bins - 1, int(probability * bins))
        grouped[index].append((example, probability))

    points: list[dict[str, float | int | None]] = []
    expected_calibration_error = 0.0
    for index, values in enumerate(grouped):
        lower = index / bins
        upper = (index + 1) / bins
        if not values:
            points.append(
                {
                    "bin": index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_predicted_probability": None,
                    "observed_positive_rate": None,
                    "mean_natural_paired_delta": None,
                }
            )
            continue
        mean_probability = sum(prediction for _, prediction in values) / len(values)
        observed_rate = sum(_target(example) for example, _ in values) / len(values)
        mean_delta = sum(
            example.observation.paired_delta for example, _ in values
        ) / len(values)
        expected_calibration_error += (
            len(values) / len(examples) * abs(mean_probability - observed_rate)
        )
        points.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": len(values),
                "mean_predicted_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "mean_natural_paired_delta": mean_delta,
            }
        )
    return {
        "expected_calibration_error": expected_calibration_error,
        "points": points,
    }


def _optimization_effect(run_dirs: Sequence[Path]) -> dict[str, Any]:
    runs = [_run_trajectory(root) for root in run_dirs]
    available = [run for run in runs if run["trajectory_available"]]
    final = [
        run["final_outcome"]
        for run in runs
        if run["final_outcome"]["status"] == "evaluated"
    ]
    return {
        "interpretation": {
            "iteration_score": "incumbent search score after each iteration",
            "cost_score": (
                "incumbent search score versus cumulative search+calibration "
                "natural-task cost units; final-evaluation cost is separate"
            ),
            "primary_final_effect": (
                "sealed final baseline-versus-terminal paired delta when available"
            ),
        },
        "summary": {
            "runs_requested": len(runs),
            "runs_with_search_trajectory": len(available),
            "mean_baseline_search_score": _mean_optional(
                [run["baseline_search_score"] for run in available]
            ),
            "mean_terminal_search_score": _mean_optional(
                [run["terminal_search_score"] for run in available]
            ),
            "mean_search_score_gain": _mean_optional(
                [run["search_score_gain"] for run in available]
            ),
            "runs_with_sealed_final": len(final),
            "mean_final_paired_delta": _mean_optional(
                [outcome["paired_delta"] for outcome in final]
            ),
            "mean_final_baseline_score": _mean_optional(
                [outcome["baseline_score"] for outcome in final]
            ),
            "mean_final_terminal_score": _mean_optional(
                [outcome["terminal_score"] for outcome in final]
            ),
        },
        "by_condition": _optimization_by_condition(runs),
        "runs": runs,
    }


def _optimization_by_condition(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("condition") or "unknown")].append(run)
    result: dict[str, Any] = {}
    for condition, condition_runs in sorted(grouped.items()):
        available = [run for run in condition_runs if run["trajectory_available"]]
        final = [
            run["final_outcome"]
            for run in condition_runs
            if run["final_outcome"]["status"] == "evaluated"
        ]
        iteration_scores: defaultdict[int, list[float]] = defaultdict(list)
        for run in available:
            for point in run["curves"]["iteration_score"]:
                iteration_scores[int(point["iteration"])].append(
                    float(point["incumbent_search_score"])
                )
        result[condition] = {
            "runs": len(condition_runs),
            "runs_with_search_trajectory": len(available),
            "mean_search_score_gain": _mean_optional(
                [run["search_score_gain"] for run in available]
            ),
            "runs_with_sealed_final": len(final),
            "mean_final_paired_delta": _mean_optional(
                [outcome["paired_delta"] for outcome in final]
            ),
            "mean_final_baseline_score": _mean_optional(
                [outcome["baseline_score"] for outcome in final]
            ),
            "mean_final_terminal_score": _mean_optional(
                [outcome["terminal_score"] for outcome in final]
            ),
            "aggregate_iteration_score_curve": [
                {
                    "iteration": iteration,
                    "runs": len(scores),
                    "mean_incumbent_search_score": sum(scores) / len(scores),
                    "min_incumbent_search_score": min(scores),
                    "max_incumbent_search_score": max(scores),
                }
                for iteration, scores in sorted(iteration_scores.items())
            ],
            "cost_score_curve": "see per-run curves; cost grids are irregular",
        }
    return result


def _run_trajectory(root: Path) -> dict[str, Any]:
    identity = _run_identity(root)
    final_outcome = _final_outcome(root)
    baseline_evaluation = _baseline_search_evaluation(root)
    if baseline_evaluation is None:
        return {
            **identity,
            "trajectory_available": False,
            "unavailable_reason": "missing baseline search evaluation",
            "baseline_search_score": None,
            "terminal_search_score": None,
            "search_score_gain": None,
            "trajectory": [],
            "curves": {"iteration_score": [], "cost_score": []},
            "final_outcome": final_outcome,
        }

    baseline_score = float(baseline_evaluation.get("score") or 0.0)
    baseline_cost = float(baseline_evaluation.get("cost") or 0.0)
    score = baseline_score
    cumulative_search_cost = baseline_cost
    cumulative_calibration_cost = 0.0
    cumulative_unit_seconds = 0.0
    points: list[dict[str, Any]] = [
        _trajectory_point(
            iteration=0,
            candidate_id="baseline",
            decision="baseline",
            search_delta=None,
            incumbent_search_score=score,
            cumulative_search_cost=cumulative_search_cost,
            cumulative_calibration_cost=cumulative_calibration_cost,
            cumulative_unit_seconds=cumulative_unit_seconds,
        )
    ]
    calibration_costs = _calibration_cost_by_iteration(root)
    iteration_dirs = {
        int(path.name.rsplit("_", 1)[-1]): path
        for path in (root / "iterations").glob("iter_*")
        if path.is_dir() and path.name.rsplit("_", 1)[-1].isdigit()
    }
    iterations = sorted(set(iteration_dirs) | set(calibration_costs))
    for iteration in iterations:
        iteration_dir = iteration_dirs.get(iteration)
        decision_payload: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        status_payload: dict[str, Any] = {}
        if iteration_dir is not None:
            decision_path = iteration_dir / "decision.json"
            evidence_path = iteration_dir / "evidence.json"
            status_path = iteration_dir / "iteration_status.json"
            if decision_path.is_file():
                decision_payload = read_json(decision_path)
            if evidence_path.is_file():
                evidence = read_json(evidence_path)
            elif isinstance(decision_payload.get("evidence"), dict):
                evidence = dict(decision_payload["evidence"])
            if status_path.is_file():
                status_payload = read_json(status_path)
        decision = str(
            decision_payload.get("decision")
            or status_payload.get("status")
            or "no_decision"
        )
        search_delta_raw = evidence.get("search_delta")
        search_delta = (
            None if search_delta_raw is None else float(search_delta_raw)
        )
        if decision == "promote" and search_delta is not None:
            score += search_delta
        cumulative_search_cost += float(evidence.get("total_cost") or 0.0)
        metadata = dict(evidence.get("metadata") or {})
        costs = dict(metadata.get("costs") or {})
        cumulative_unit_seconds += float(costs.get("unit_test_wall_seconds") or 0.0)
        cumulative_calibration_cost += calibration_costs.get(iteration, 0.0)
        points.append(
            _trajectory_point(
                iteration=iteration,
                candidate_id=str(
                    decision_payload.get("candidate_id")
                    or evidence.get("candidate_id")
                    or status_payload.get("candidate_id")
                    or ""
                ),
                decision=decision,
                search_delta=search_delta,
                incumbent_search_score=score,
                cumulative_search_cost=cumulative_search_cost,
                cumulative_calibration_cost=cumulative_calibration_cost,
                cumulative_unit_seconds=cumulative_unit_seconds,
            )
        )

    reported_terminal = _reported_terminal_search_score(root)
    return {
        **identity,
        "trajectory_available": True,
        "unavailable_reason": None,
        "baseline_search_score": baseline_score,
        "terminal_search_score": score,
        "reported_terminal_search_score": reported_terminal,
        "terminal_score_matches_reported": (
            None
            if reported_terminal is None
            else math.isclose(score, reported_terminal, rel_tol=1e-9, abs_tol=1e-12)
        ),
        "search_score_gain": score - baseline_score,
        "trajectory": points,
        "curves": {
            "iteration_score": [
                {
                    "iteration": point["iteration"],
                    "incumbent_search_score": point["incumbent_search_score"],
                }
                for point in points
            ],
            "cost_score": [
                {
                    "cumulative_natural_task_cost": point[
                        "cumulative_natural_task_cost"
                    ],
                    "incumbent_search_score": point["incumbent_search_score"],
                    "iteration": point["iteration"],
                }
                for point in points
            ],
        },
        "final_outcome": final_outcome,
    }


def _trajectory_point(
    *,
    iteration: int,
    candidate_id: str,
    decision: str,
    search_delta: float | None,
    incumbent_search_score: float,
    cumulative_search_cost: float,
    cumulative_calibration_cost: float,
    cumulative_unit_seconds: float,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "candidate_id": candidate_id,
        "decision": decision,
        "search_delta": search_delta,
        "incumbent_search_score": incumbent_search_score,
        "cumulative_search_cost": cumulative_search_cost,
        "cumulative_calibration_cost": cumulative_calibration_cost,
        "cumulative_natural_task_cost": (
            cumulative_search_cost + cumulative_calibration_cost
        ),
        "cumulative_unit_test_wall_seconds": cumulative_unit_seconds,
    }


def _run_identity(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    config_path = root / "config.snapshot.json"
    config = read_json(config_path) if config_path.is_file() else {}
    loop = dict(config.get("loop") or {})
    protocol = dict(config.get("protocol") or {})
    return {
        "run_dir": str(root),
        "run_id": str(summary.get("run_id") or loop.get("run_id") or root.name),
        "condition": str(summary.get("protocol") or protocol.get("condition") or ""),
    }


def _baseline_search_evaluation(root: Path) -> dict[str, Any] | None:
    plan_path = root / "benchmark_data" / "plan.json"
    search_slice = "search"
    if plan_path.is_file():
        search = dict(read_json(plan_path).get("search") or {})
        search_slice = str(search.get("slice_id") or "search")
    preferred = root / "evaluations" / "baseline" / search_slice / "evaluation.json"
    if preferred.is_file():
        return read_json(preferred)
    candidates = sorted((root / "evaluations" / "baseline").glob("*/evaluation.json"))
    for path in candidates:
        payload = read_json(path)
        if str(payload.get("split") or "") == search_slice:
            return payload
    return read_json(candidates[0]) if len(candidates) == 1 else None


def _calibration_cost_by_iteration(root: Path) -> dict[int, float]:
    result: defaultdict[int, float] = defaultdict(float)
    for path in sorted((root / "calibration" / "checkpoints").glob("*/result.json")):
        payload = read_json(path)
        effective = int(payload.get("effective_from_iteration") or 1)
        result[max(0, effective - 1)] += float(payload.get("cost") or 0.0)
    return dict(result)


def _reported_terminal_search_score(root: Path) -> float | None:
    for name in ("summary.json", "run_state.json"):
        path = root / name
        if not path.is_file():
            continue
        value = read_json(path).get("incumbent_search_score")
        if value is not None:
            return float(value)
    return None


def _final_outcome(root: Path) -> dict[str, Any]:
    path = root / "sealed" / "final" / "report.json"
    if not path.is_file():
        return {"status": "not_opened"}
    payload = read_json(path)
    return {
        "status": "evaluated",
        "baseline_score": float(payload.get("baseline_score") or 0.0),
        "terminal_score": float(payload.get("terminal_score") or 0.0),
        "paired_delta": float(payload.get("paired_delta") or 0.0),
        "matched_tasks": int(payload.get("matched_tasks") or 0),
        "final_evaluation_cost": float(payload.get("cost") or 0.0),
        "terminal_candidate_id": str(payload.get("terminal_candidate_id") or ""),
    }


def _mean_optional(values: Sequence[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _target(example: ProxyExample) -> int:
    target = example.target
    if target is None:
        raise ValueError("inconclusive observations cannot be used as binary targets")
    return target


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _optional_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right
