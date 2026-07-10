from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceunit.calibration import CalibrationLabel, CalibrationObservation
from traceunit.cli import main
from traceunit.io import append_jsonl, write_json
from traceunit.models import EvidenceRecord
from traceunit.proxy_analysis import analyze_proxy


def _write_run(
    root: Path,
    *,
    lineage: str,
    labels: tuple[CalibrationLabel, ...] = (
        CalibrationLabel.POSITIVE,
        CalibrationLabel.NEGATIVE,
        CalibrationLabel.POSITIVE,
        CalibrationLabel.NEGATIVE,
    ),
) -> None:
    write_json(
        root / "config.snapshot.json",
        {"loop": {"run_id": lineage}, "benchmark": {"name": "fake-benchmark"}},
    )
    for index, label in enumerate(labels, start=1):
        positive = label is CalibrationLabel.POSITIVE
        candidate_id = f"{lineage}-candidate-{index}"
        evidence = EvidenceRecord(
            iteration=index,
            candidate_id=candidate_id,
            packet_id=f"packet-{index}",
            public_gain=1.0 if positive else 0.0,
            hidden_gain=1.0 if positive else 0.0,
            bridge_gain=1.0 if positive else 0.0,
            regression_loss=0.0 if positive else 1.0,
            admission_score=1.0,
            search_delta=0.0,
            metadata={
                "has_bridge": True,
                "composition_ids": [],
                "family_keys": ["planner.recovery"],
            },
        )
        write_json(
            root / "iterations" / f"iter_{index:03d}" / "evidence.json",
            evidence.to_dict(),
        )
        append_jsonl(
            root / "calibration" / "private_observations.jsonl",
            CalibrationObservation(
                candidate_id=candidate_id,
                parent_id=f"{lineage}-parent",
                lineage_id=lineage,
                epoch=1,
                shard_id=f"{lineage}-shard",
                stratum="search0|composition0",
                family_keys=("planner.recovery",),
                unit_profile=(
                    "unit+|bridge+|regression+"
                    if positive
                    else "unit-|bridge0|regression-"
                ),
                paired_delta=0.2 if positive else -0.2,
                uncertainty=0.01,
                label=label,
            ).to_dict(),
        )


def _write_trajectory_artifacts(root: Path) -> None:
    write_json(
        root / "evaluations" / "baseline" / "search" / "evaluation.json",
        {"split": "search", "score": 0.2, "cost": 10.0},
    )
    deltas = {1: 0.1, 2: -0.1, 3: 0.2, 4: None}
    costs = {1: 3.0, 2: 3.0, 3: 4.0, 4: 0.0}
    decisions = {1: "promote", 2: "reject", 3: "promote", 4: "reject"}
    unit_seconds = {1: 1.0, 2: 1.0, 3: 2.0, 4: 1.0}
    for iteration in range(1, 5):
        evidence_path = root / "iterations" / f"iter_{iteration:03d}" / "evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["search_delta"] = deltas[iteration]
        evidence["total_cost"] = costs[iteration]
        evidence.setdefault("metadata", {}).setdefault("costs", {})[
            "unit_test_wall_seconds"
        ] = unit_seconds[iteration]
        write_json(evidence_path, evidence)
        write_json(
            evidence_path.parent / "decision.json",
            {
                "iteration": iteration,
                "candidate_id": evidence["candidate_id"],
                "decision": decisions[iteration],
                "evidence": evidence,
            },
        )
    write_json(
        root / "calibration" / "checkpoints" / "checkpoint-1" / "result.json",
        {"effective_from_iteration": 4, "cost": 5.0},
    )
    write_json(
        root / "summary.json",
        {
            "run_id": "trajectory-run",
            "protocol": "c3_full",
            "incumbent_search_score": 0.5,
        },
    )
    write_json(
        root / "sealed" / "final" / "report.json",
        {
            "baseline_score": 0.1,
            "terminal_score": 0.4,
            "paired_delta": 0.3,
            "matched_tasks": 20,
            "cost": 12.0,
            "terminal_candidate_id": "trajectory-candidate",
        },
    )


def test_oof_unit_evidence_improves_prediction_and_selective_gate(
    tmp_path: Path,
) -> None:
    runs = [tmp_path / f"run-{index}" for index in range(3)]
    for index, run in enumerate(runs):
        _write_run(run, lineage=f"lineage-{index}")

    report = analyze_proxy(runs, audit_rate=0.1)

    assert report["split"]["folds"] == 3
    assert report["cohort"]["analyzed_informative"] == 12
    assert report["prediction_count_check"] == {
        "baseline": 12,
        "proxy": 12,
        "expected": 12,
    }
    assert report["incremental_value"]["brier_reduction"] > 0
    assert report["baseline_search_only"]["pairwise_accuracy"] == 0.5
    assert report["proxy_search_plus_unit"]["pairwise_accuracy"] == 1.0
    baseline_alignment = report["proxy_alignment_curve"]["baseline_search_only"]
    proxy_alignment = report["proxy_alignment_curve"]["proxy_search_plus_unit"]
    assert sum(point["count"] for point in proxy_alignment["points"]) == 12
    assert len([point for point in baseline_alignment["points"] if point["count"]]) == 1
    observed_proxy_rates = {
        point["observed_positive_rate"]
        for point in proxy_alignment["points"]
        if point["count"]
    }
    assert observed_proxy_rates == {0.0, 1.0}

    unit_gate = report["selective_full_evaluation"]["frozen_unit_gate"]
    assert unit_gate["full_evaluation_rate_without_audit"] == 0.5
    assert unit_gate["positive_recall_without_audit"] == 1.0
    assert unit_gate["expected_full_evaluation_rate_with_audit"] == pytest.approx(0.55)
    assert "sealed_final_is_reported_only" in report["information_boundary"]


def test_iteration_cost_and_final_effect_are_reconstructed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, lineage="lineage-first")
    _write_run(second, lineage="lineage-second")
    _write_trajectory_artifacts(first)

    report = analyze_proxy([first, second])
    optimization = report["optimization_effect"]
    run = next(
        item for item in optimization["runs"] if item["run_id"] == "trajectory-run"
    )

    assert optimization["summary"]["runs_with_search_trajectory"] == 1
    assert optimization["summary"]["runs_with_sealed_final"] == 1
    assert optimization["summary"]["mean_final_paired_delta"] == pytest.approx(0.3)
    condition = optimization["by_condition"]["c3_full"]
    assert condition["mean_final_paired_delta"] == pytest.approx(0.3)
    assert condition["aggregate_iteration_score_curve"][-1][
        "mean_incumbent_search_score"
    ] == pytest.approx(0.5)
    assert run["baseline_search_score"] == pytest.approx(0.2)
    assert run["terminal_search_score"] == pytest.approx(0.5)
    assert run["search_score_gain"] == pytest.approx(0.3)
    assert run["terminal_score_matches_reported"] is True
    assert [point["iteration"] for point in run["trajectory"]] == [0, 1, 2, 3, 4]
    assert run["trajectory"][3]["cumulative_search_cost"] == pytest.approx(20.0)
    assert run["trajectory"][3]["cumulative_calibration_cost"] == pytest.approx(5.0)
    assert run["trajectory"][3]["cumulative_natural_task_cost"] == pytest.approx(25.0)
    assert run["trajectory"][-1]["cumulative_unit_test_wall_seconds"] == pytest.approx(5.0)
    assert run["curves"]["iteration_score"][-1] == {
        "iteration": 4,
        "incumbent_search_score": pytest.approx(0.5),
    }
    assert run["curves"]["cost_score"][-1][
        "cumulative_natural_task_cost"
    ] == pytest.approx(25.0)
    assert run["final_outcome"]["paired_delta"] == pytest.approx(0.3)


def test_analysis_requires_multiple_out_of_sample_groups(tmp_path: Path) -> None:
    run = tmp_path / "only-run"
    _write_run(run, lineage="only-lineage")

    with pytest.raises(ValueError, match="at least two groups"):
        analyze_proxy([run])


def test_analyze_proxy_cli_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "proxy-report.json"
    _write_run(first, lineage="lineage-first")
    _write_run(second, lineage="lineage-second")

    assert (
        main(
            [
                "analyze-proxy",
                "--run-dir",
                str(first),
                "--run-dir",
                str(second),
                "--output",
                str(output),
                "--skip-below",
                "0.25",
                "--audit-rate",
                "0.2",
            ]
        )
        == 0
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["selective_full_evaluation"]["audit_rate"] == 0.2
    assert len(written["selective_full_evaluation"]["proxy_search_plus_unit"]) == 1
