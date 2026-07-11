from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import traceunit.tests_runtime as runtime
from traceunit.io import write_json
from traceunit.models import (
    EvidenceRole,
    TestExecution as ExecutionResult,
    TestExecutionMode as ExecutionMode,
    TestTier as Tier,
)
from traceunit.ontology import ontology_ref
from traceunit.tests_runtime import (
    InvalidTestPacket,
    TestSandboxUnavailable as SandboxUnavailable,
    admission_contract,
    freeze_test_packet,
    load_test_packet,
    paired_test_metrics,
    run_test_cases,
    verify_frozen_packet,
)


def _write_packet(bundle: Path) -> None:
    public = bundle / "tests/public/target.py"
    hidden = bundle / "tests/hidden/sibling.py"
    bridge = bundle / "tests/hidden/bridge.py"
    regression = bundle / "tests/hidden/regression.py"
    witness = bundle / "tests/hidden/positive_witness.py"
    for path in (public, hidden, bridge):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import os, pathlib\n"
            "root = pathlib.Path(os.environ['TRACEUNIT_SOURCE'])\n"
            "raise SystemExit(0 if (root / 'behavior.txt').read_text().strip() == 'good' else 1)\n",
            encoding="utf-8",
        )
    regression.write_text(
        "import os, pathlib\n"
        "assert 'DEEPSEEK_API_KEY' not in os.environ\n"
        "root = pathlib.Path(os.environ['TRACEUNIT_SOURCE'])\n"
        "raise SystemExit(0 if (root / 'behavior.txt').exists() else 1)\n",
        encoding="utf-8",
    )
    witness.write_text(
        "# Controlled witness: the asserted good mechanism is satisfiable.\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    cases = [
        {
            "case_id": "public",
            "tier": "public",
            "evidence_role": "target_reproducer",
            "path": "tests/public/target.py",
            "expected_incumbent_pass": False,
            "expected_candidate_pass": True,
        },
        {
            "case_id": "hidden",
            "tier": "hidden",
            "evidence_role": "structural_sibling",
            "path": "tests/hidden/sibling.py",
            "expected_incumbent_pass": False,
            "expected_candidate_pass": True,
        },
        {
            "case_id": "bridge",
            "tier": "bridge",
            "evidence_role": "downstream_bridge",
            "path": "tests/hidden/bridge.py",
            "expected_incumbent_pass": False,
            "expected_candidate_pass": True,
        },
        {
            "case_id": "regression",
            "tier": "regression",
            "evidence_role": "off_target_control",
            "path": "tests/hidden/regression.py",
            "expected_incumbent_pass": True,
            "expected_candidate_pass": True,
        },
        {
            "case_id": "positive_witness",
            "tier": "admission",
            "evidence_role": "positive_witness",
            "path": "tests/hidden/positive_witness.py",
            "expected_incumbent_pass": True,
            "expected_candidate_pass": True,
        },
    ]
    write_json(
        bundle / "test_packet.json",
        {
            "packet_id": "packet",
            "version": 1,
            "source_trace_ids": ["trace"],
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "family": "verification",
                    "intervention_kind": "local_repair",
                    "mechanism": "behavior",
                    "target_boundary": "file",
                    "claim": "behavior becomes good",
                    "evidence_trace_ids": ["trace"],
                    "alternatives": ["h2"],
                    "confidence": 0.8,
                },
                {
                    "hypothesis_id": "h2",
                    "family": "context",
                    "intervention_kind": "orchestration_change",
                    "mechanism": "unrelated file absence",
                    "target_boundary": "file existence",
                    "claim": "the behavior file is missing",
                    "evidence_trace_ids": ["trace"],
                    "alternatives": ["h1"],
                    "confidence": 0.2,
                },
            ],
            "target_hypothesis_id": "h1",
            "primary_family": "verification",
            "public_contract": "behavior must be good",
            "hidden_variant_strategy": "equivalent sibling",
            "cases": cases,
            "metadata": {},
        },
    )


def test_admission_freeze_and_paired_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    incumbent = tmp_path / "incumbent"
    candidate = tmp_path / "candidate"
    incumbent.mkdir()
    candidate.mkdir()
    (incumbent / "behavior.txt").write_text("bad", encoding="utf-8")
    (candidate / "behavior.txt").write_text("good", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-leak")

    packet = load_test_packet(bundle)
    baseline = run_test_cases(
        packet=packet,
        bundle=bundle,
        source=incumbent,
        subject="incumbent",
        output_dir=tmp_path / "base-results",
    )
    admitted, reasons = admission_contract(packet, baseline)
    assert admitted is True
    assert reasons == []
    packet = freeze_test_packet(bundle, packet, admission_passed=admitted)
    assert verify_frozen_packet(bundle, packet)

    proposed = run_test_cases(
        packet=packet,
        bundle=bundle,
        source=candidate,
        subject="candidate",
        output_dir=tmp_path / "candidate-results",
    )
    metrics = paired_test_metrics(packet, baseline, proposed)
    assert metrics == {
        "public_gain": 1.0,
        "hidden_gain": 1.0,
        "bridge_gain": 1.0,
        "regression_loss": 0.0,
    }


def test_admitted_packet_is_bound_to_frozen_ontology(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    proposed = load_test_packet(bundle)

    frozen = freeze_test_packet(bundle, proposed, admission_passed=True)

    assert frozen.metadata["ontology"] == ontology_ref()
    assert load_test_packet(bundle).metadata["ontology"] == ontology_ref()
    payload = json.loads((bundle / "test_packet.json").read_text(encoding="utf-8"))
    payload["metadata"]["ontology"]["version"] = "tampered"
    write_json(bundle / "test_packet.json", payload)
    with pytest.raises(InvalidTestPacket, match="unknown L0 ontology"):
        load_test_packet(bundle)


def test_model_probe_is_host_controlled_and_budget_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    packet_path = bundle / "test_packet.json"
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    payload["cases"][0].update(
        {
            "path": "tests/public/probe.json",
            "driver": "agent_probe",
            "execution_mode": "model_backed_probe",
            "max_model_calls": 2,
            "max_tokens": 100,
        }
    )
    write_json(bundle / "tests/public/probe.json", {"probe": "counterexample"})
    write_json(packet_path, payload)
    packet = load_test_packet(bundle)
    source = tmp_path / "source"
    source.mkdir()
    (source / "behavior.txt").write_text("bad", encoding="utf-8")
    calls: list[str] = []

    def probe_runner(
        case: runtime.TestCaseSpec,
        test_bundle: Path,
        subject_source: Path,
        subject: str,
        output_dir: Path,
    ) -> ExecutionResult:
        assert test_bundle == bundle.resolve()
        assert subject_source == source.resolve()
        calls.append(case.case_id)
        return ExecutionResult(
            case_id=case.case_id,
            tier=Tier.PUBLIC,
            evidence_role=EvidenceRole.TARGET_REPRODUCER,
            execution_mode=ExecutionMode.MODEL_BACKED_PROBE,
            subject=subject,
            passed=True,
            returncode=0,
            duration_s=0.1,
            stdout_path=str(output_dir / "probe.stdout"),
            stderr_path=str(output_dir / "probe.stderr"),
            model_calls=2,
            tokens=90,
        )

    monkeypatch.setattr(
        runtime,
        "_sandbox_backend",
        lambda: pytest.fail("model-only probe initialized a code sandbox"),
    )
    results = run_test_cases(
        packet=packet,
        bundle=bundle,
        source=source,
        subject="candidate",
        output_dir=tmp_path / "probe-results",
        tiers={Tier.PUBLIC},
        probe_runner=probe_runner,
    )
    assert calls == ["public"]
    assert results[0].model_calls == 2
    assert results[0].tokens == 90

    def over_budget(*args: object) -> ExecutionResult:
        result = probe_runner(*args)  # type: ignore[arg-type]
        return replace(result, tokens=101)

    with pytest.raises(SandboxUnavailable, match="exceeded max_tokens"):
        run_test_cases(
            packet=packet,
            bundle=bundle,
            source=source,
            subject="candidate",
            output_dir=tmp_path / "over-budget-results",
            tiers={Tier.PUBLIC},
            probe_runner=over_budget,
        )


def test_packet_rejects_evaluator_access(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    (bundle / "tests/hidden/sibling.py").write_text(
        "world.evaluate()\n", encoding="utf-8"
    )
    with pytest.raises(InvalidTestPacket, match="evaluator access"):
        load_test_packet(bundle)


def test_packet_requires_competing_hypothesis_and_positive_witness(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    path = bundle / "test_packet.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hypotheses"] = payload["hypotheses"][:1]
    payload["hypotheses"][0]["alternatives"] = []
    write_json(path, payload)
    with pytest.raises(InvalidTestPacket, match="competing failure hypotheses"):
        load_test_packet(bundle)

    _write_packet(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"] = [
        case for case in payload["cases"] if case["tier"] != "admission"
    ]
    write_json(path, payload)
    with pytest.raises(InvalidTestPacket, match="positive_witness"):
        load_test_packet(bundle)


def test_isolated_snapshot_blocks_source_bundle_mutation_and_env_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    public = bundle / "tests/public/target.py"
    public.write_text(
        "import os, pathlib\n"
        "assert 'HOST_SECRET' not in os.environ\n"
        "home = pathlib.Path(os.environ['HOME'])\n"
        "assert home.is_dir() and list(home.iterdir()) == []\n"
        "source = pathlib.Path(os.environ['TRACEUNIT_SOURCE']) / 'behavior.txt'\n"
        "packet = pathlib.Path(os.environ['TRACEUNIT_TEST_BUNDLE']) / 'test_packet.json'\n"
        "for target in (source, packet):\n"
        "    try:\n"
        "        target.write_text('mutated')\n"
        "    except OSError:\n"
        "        pass\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "behavior.txt").write_text("bad", encoding="utf-8")
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    (real_home / "credential.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HOST_SECRET", "must-not-leak")

    packet = load_test_packet(bundle)
    packet = freeze_test_packet(bundle, packet, admission_passed=True)
    results = run_test_cases(
        packet=packet,
        bundle=bundle,
        source=source,
        subject="incumbent",
        output_dir=tmp_path / "isolated-results",
    )

    assert results[0].passed
    assert (source / "behavior.txt").read_text(encoding="utf-8") == "bad"
    assert verify_frozen_packet(bundle, packet)
    assert (real_home / "credential.txt").read_text(encoding="utf-8") == "secret"


def test_packet_rejects_protected_environment_override(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    path = bundle / "test_packet.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["environment"] = {"HOME": "{source}"}
    write_json(path, payload)

    with pytest.raises(InvalidTestPacket, match="protected key"):
        load_test_packet(bundle)


def test_real_execution_fails_closed_without_docker_or_bwrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    marker = tmp_path / "executed.txt"
    public = bundle / "tests/public/target.py"
    public.write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['MARKER']).write_text('executed')\n",
        encoding="utf-8",
    )
    packet_path = bundle / "test_packet.json"
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    payload["cases"][0]["environment"] = {"MARKER": str(marker)}
    write_json(packet_path, payload)
    packet = load_test_packet(bundle)
    source = tmp_path / "source"
    source.mkdir()
    (source / "behavior.txt").write_text("bad", encoding="utf-8")

    monkeypatch.delenv("TRACEUNIT_TEST_SANDBOX_MODE", raising=False)
    monkeypatch.setattr(runtime, "_docker_available", lambda: False)
    monkeypatch.setattr(runtime, "_bwrap_available", lambda: False)
    with pytest.raises(SandboxUnavailable, match="no generated-test sandbox"):
        run_test_cases(
            packet=packet,
            bundle=bundle,
            source=source,
            subject="incumbent",
            output_dir=tmp_path / "closed-results",
        )
    assert not marker.exists()


def test_docker_sandbox_hides_host_and_blocks_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not runtime._docker_available():
        pytest.skip("cached TraceUnit Docker image is unavailable")

    bundle = tmp_path / "bundle"
    _write_packet(bundle)
    marker = Path("/tmp") / (
        f"traceunit-host-marker-{tmp_path.parent.name}-{tmp_path.name}"
    )
    marker.unlink(missing_ok=True)
    public = bundle / "tests/public/target.py"
    public.write_text(
        "import os, pathlib\n"
        "assert 'HOST_SECRET' not in os.environ\n"
        "home = pathlib.Path(os.environ['HOME'])\n"
        "assert home.is_dir() and list(home.iterdir()) == []\n"
        "source = pathlib.Path(os.environ['TRACEUNIT_SOURCE']) / 'behavior.txt'\n"
        "packet = pathlib.Path(os.environ['TRACEUNIT_TEST_BUNDLE']) / 'test_packet.json'\n"
        "for target in (source, packet):\n"
        "    try:\n"
        "        target.write_text('mutated')\n"
        "    except OSError:\n"
        "        pass\n"
        "assert source.read_text() == 'bad'\n"
        "assert '\"packet_id\"' in packet.read_text()\n"
        "pathlib.Path(os.environ['HOST_MARKER']).write_text('container-only')\n",
        encoding="utf-8",
    )
    packet_path = bundle / "test_packet.json"
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    payload["cases"][0]["environment"] = {"HOST_MARKER": str(marker)}
    write_json(packet_path, payload)
    source = tmp_path / "source"
    source.mkdir()
    (source / "behavior.txt").write_text("bad", encoding="utf-8")

    packet = load_test_packet(bundle)
    packet = freeze_test_packet(bundle, packet, admission_passed=True)
    monkeypatch.setenv("TRACEUNIT_TEST_SANDBOX_MODE", "docker")
    monkeypatch.setenv("HOST_SECRET", "must-not-leak")
    try:
        results = run_test_cases(
            packet=packet,
            bundle=bundle,
            source=source,
            subject="incumbent",
            output_dir=tmp_path / "docker-results",
            tiers={runtime.TestTier.PUBLIC},
        )
    finally:
        marker_leaked = marker.exists()
        marker.unlink(missing_ok=True)

    assert len(results) == 1 and results[0].passed
    assert not marker_leaked
    assert (source / "behavior.txt").read_text(encoding="utf-8") == "bad"
    assert verify_frozen_packet(bundle, packet)
