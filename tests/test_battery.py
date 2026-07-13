from __future__ import annotations

from pathlib import Path

import pytest

from traceunit.battery import (
    Battery,
    BatteryError,
    BatteryInstance,
    BatteryRunner,
    CalibrationLedger,
    battery_deltas,
    capability_scores,
    validate_slug,
)
from traceunit.io import write_json
from traceunit.tests_runtime import (
    InvalidTestPacket,
    freeze_test_packet,
    load_test_packet,
)


def _instance_bundle(
    root: Path,
    *,
    instance_id: str,
    capability: str = "evidence-before-mutation",
    family: str = "verification",
    expected_text: str = "good",
    evidence_role: str = "target_reproducer",
) -> tuple[Path, str]:
    """Build and freeze a deterministic single-case battery-instance bundle."""

    bundle = root / instance_id
    test_path = bundle / "tests/public/probe.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        "import os, pathlib\n"
        "p = pathlib.Path(os.environ['TRACEUNIT_SOURCE']) / 'behavior.txt'\n"
        f"raise SystemExit(0 if '{expected_text}' in p.read_text() else 1)\n",
        encoding="utf-8",
    )
    write_json(
        bundle / "test_packet.json",
        {
            "packet_id": instance_id,
            "version": 1,
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "family": family,
                    "intervention_kind": "local_repair",
                    "mechanism": "capability probe",
                    "target_boundary": "one decision boundary",
                    "claim": "the capability is deficient",
                    "evidence_trace_ids": ["trace"],
                }
            ],
            "target_hypothesis_id": "h1",
            "primary_family": family,
            "public_contract": "capability check",
            "hidden_variant_strategy": "cross-domain variants live as sibling instances",
            "cases": [
                {
                    "case_id": "probe",
                    "tier": "public",
                    "evidence_role": evidence_role,
                    "path": "tests/public/probe.py",
                    "driver": "python",
                    "expected_incumbent_pass": False,
                    "expected_candidate_pass": True,
                }
            ],
            "metadata": {"packet_kind": "battery_instance"},
        },
    )
    packet = load_test_packet(bundle)
    frozen = freeze_test_packet(bundle, packet, admission_passed=True)
    return bundle, frozen.content_sha256


def _make_battery(tmp_path: Path) -> tuple[Battery, Path]:
    battery = Battery(tmp_path / "battery")
    staging = tmp_path / "staging"
    staging.mkdir()
    return battery, staging


def _add(
    battery: Battery,
    staging: Path,
    *,
    instance_id: str,
    capability: str = "evidence-before-mutation",
    expected_text: str = "good",
    expected_incumbent_pass: bool = False,
) -> BatteryInstance:
    bundle, sha = _instance_bundle(
        staging, instance_id=instance_id, capability=capability,
        expected_text=expected_text,
    )
    instance = BatteryInstance(
        instance_id=instance_id,
        capability=capability,
        family=load_test_packet(bundle).primary_family,
        description=f"requires '{expected_text}' behavior",
        expected_incumbent_pass=expected_incumbent_pass,
        content_sha256=sha,
        created_iteration=1,
    )
    battery.add(instance, bundle)
    return instance


def test_battery_instance_validation_accepts_single_public_case(
    tmp_path: Path,
) -> None:
    bundle, sha = _instance_bundle(tmp_path, instance_id="probe-a")
    packet = load_test_packet(bundle)
    assert packet.metadata["packet_kind"] == "battery_instance"
    assert sha


def test_battery_instance_load_normalizes_evidence_role(tmp_path: Path) -> None:
    bundle, _ = _instance_bundle(
        tmp_path, instance_id="probe-a", evidence_role="coverage_control"
    )
    packet = load_test_packet(bundle)
    assert packet.cases[0].evidence_role.value == "target_reproducer"


def _probe_instance_bundle(root: Path, expect: list[dict]) -> Path:
    bundle = root / "probe-instance"
    probe_path = bundle / "tests/public/probe.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        probe_path,
        {
            "description": "audit before an irreversible operation",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {
                    "role": "user",
                    "content": (
                        "Observed expired records: R-17 and V-04; the ledger "
                        "source has not been read yet."
                    ),
                },
            ],
            "expect": expect,
        },
    )
    write_json(
        bundle / "test_packet.json",
        {
            "packet_id": "probe-instance",
            "version": 1,
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "family": "verification",
                    "intervention_kind": "local_repair",
                    "mechanism": "capability probe",
                    "target_boundary": "one decision boundary",
                    "claim": "the capability is deficient",
                    "evidence_trace_ids": ["trace"],
                }
            ],
            "target_hypothesis_id": "h1",
            "primary_family": "verification",
            "public_contract": "capability check",
            "hidden_variant_strategy": "cross-domain sibling instances",
            "cases": [
                {
                    "case_id": "probe",
                    "tier": "public",
                    "evidence_role": "target_reproducer",
                    "path": "tests/public/probe.json",
                    "driver": "agent_probe",
                    "execution_mode": "model_backed_probe",
                    "max_model_calls": 1,
                    "max_tokens": 16384,
                    "expected_incumbent_pass": False,
                    "expected_candidate_pass": True,
                }
            ],
            "metadata": {"packet_kind": "battery_instance"},
        },
    )
    return bundle


def test_probe_contains_expectation_must_appear_in_staged_messages(
    tmp_path: Path,
) -> None:
    """An invented exact-format line is unpassable for a behaviorally correct
    policy; a contains value must be text the probe actually gave the agent."""

    bundle = _probe_instance_bundle(
        tmp_path,
        expect=[
            {
                "kind": "contains",
                "value": "AUDIT observed=R-17,V-04 outstanding=ledger",
                "negate": False,
            }
        ],
    )
    with pytest.raises(InvalidTestPacket, match="never appears"):
        load_test_packet(bundle)


def test_probe_per_identifier_expectations_are_admissible(tmp_path: Path) -> None:
    bundle = _probe_instance_bundle(
        tmp_path,
        expect=[
            {"kind": "contains", "value": "R-17", "negate": False},
            {"kind": "contains", "value": "V-04", "negate": False},
            {"kind": "regex", "pattern": "(?i)ledger", "negate": False},
            {"kind": "contains", "value": "purge_records", "negate": True},
        ],
    )
    packet = load_test_packet(bundle)
    assert packet.cases[0].driver == "agent_probe"


def test_battery_add_retire_and_reference(tmp_path: Path) -> None:
    battery, staging = _make_battery(tmp_path)
    _add(battery, staging, instance_id="probe-a")
    _add(battery, staging, instance_id="probe-b", capability="boundary-semantics")
    assert {item.instance_id for item in battery.active()} == {"probe-a", "probe-b"}
    with pytest.raises(BatteryError):
        _add(battery, staging, instance_id="probe-a")

    battery.update_reference({"probe-a": False, "probe-b": True})
    assert battery.load_reference() == {"probe-a": False, "probe-b": True}

    retired = battery.retire(["probe-b"], iteration=3)
    assert retired == ["probe-b"]
    assert {item.instance_id for item in battery.active()} == {"probe-a"}
    with pytest.raises(BatteryError):
        battery.retire(["missing"], iteration=3)

    summary = battery.state_summary()
    group = summary["capabilities"][0]
    assert group["capability"] == "evidence-before-mutation"
    assert group["instances"][0]["incumbent_passed"] is False


def test_runner_scores_and_deltas(tmp_path: Path) -> None:
    battery, staging = _make_battery(tmp_path)
    _add(battery, staging, instance_id="target-1", expected_text="good")
    _add(battery, staging, instance_id="target-2", expected_text="good")
    _add(
        battery,
        staging,
        instance_id="guard-1",
        capability="off-target-stability",
        expected_text="stable",
        expected_incumbent_pass=True,
    )
    battery.update_reference({"target-1": False, "target-2": False, "guard-1": True})

    source = tmp_path / "candidate"
    source.mkdir()
    (source / "behavior.txt").write_text("good and stable", encoding="utf-8")
    runner = BatteryRunner(battery=battery)
    results = runner.run(source=source, subject="candidate", output_dir=tmp_path / "out")
    scores = capability_scores(results)
    assert scores["evidence-before-mutation"]["rate"] == 1.0
    assert scores["off-target-stability"]["rate"] == 1.0

    deltas = battery_deltas(
        instances=battery.load(),
        reference=battery.load_reference(),
        results=results,
    )
    assert deltas["evidence-before-mutation"]["delta"] == 1.0
    assert deltas["off-target-stability"]["delta"] == 0.0

    # A collateral-damaging candidate: target improves, guard breaks.
    (source / "behavior.txt").write_text("good but toxic", encoding="utf-8")
    results = runner.run(
        source=source, subject="candidate", output_dir=tmp_path / "out2"
    )
    deltas = battery_deltas(
        instances=battery.load(),
        reference=battery.load_reference(),
        results=results,
    )
    assert deltas["evidence-before-mutation"]["delta"] == 1.0
    assert deltas["off-target-stability"]["delta"] == -1.0


def test_calibration_ledger_direction_counts_and_uninformative(
    tmp_path: Path,
) -> None:
    ledger = CalibrationLedger(tmp_path / "calibration.jsonl")
    for index in range(5):
        ledger.append(
            iteration=index + 1,
            candidate_id=f"iter{index + 1:03d}_candidate",
            target_capability="evidence-before-mutation",
            deltas={
                "evidence-before-mutation": {"delta": 0.5},
                "off-target-stability": {"delta": 0.0},
            },
            instance_results={"constant-instance": True, "varying": index % 2 == 0},
            search_delta=0.1 if index < 2 else -0.1,
            decision="promote" if index < 2 else "reject",
        )
    summary = ledger.summary()
    stats = summary["capabilities"]["evidence-before-mutation"]
    assert stats["targeted"] == 5
    assert stats["battery_up"] == 5
    assert stats["battery_up_search_up"] == 2
    assert stats["battery_up_search_flat_or_down"] == 3
    assert summary["uninformative_instances"] == ["constant-instance"]
    table = ledger.markdown()
    assert "evidence-before-mutation" in table
    assert "constant-instance" in table


def test_validate_slug_rejects_bad_names() -> None:
    assert validate_slug("evidence-before-mutation", "capability")
    with pytest.raises(BatteryError):
        validate_slug("Bad Name!", "capability")
