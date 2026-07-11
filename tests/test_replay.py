from __future__ import annotations

from pathlib import Path

import pytest

from traceunit.io import write_json
from traceunit.replay import (
    FrozenPacketRef,
    PacketReplayer,
    ReplayError,
    copy_packet_into_store,
)
from traceunit.tests_runtime import freeze_test_packet, load_test_packet


def _write_source(root: Path, value: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "behavior.txt").write_text(f"{value}\n", encoding="utf-8")


def _frozen_packet_bundle(bundle: Path) -> str:
    """Author and freeze a packet whose candidate contract requires 'good'."""

    public = bundle / "tests/public/target.py"
    hidden = bundle / "tests/hidden/sibling.py"
    witness = bundle / "tests/hidden/positive_witness.py"
    for path in (public, hidden):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import os, pathlib\n"
            "root = pathlib.Path(os.environ['TRACEUNIT_SOURCE'])\n"
            "raise SystemExit("
            "0 if (root / 'behavior.txt').read_text().strip() == 'good' else 1)\n",
            encoding="utf-8",
        )
    witness.write_text("raise SystemExit(0)\n", encoding="utf-8")
    write_json(
        bundle / "test_packet.json",
        {
            "packet_id": "latent-packet",
            "version": 1,
            "source_trace_ids": ["trace"],
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "family": "verification",
                    "intervention_kind": "local_repair",
                    "mechanism": "behavior",
                    "target_boundary": "behavior file",
                    "claim": "behavior becomes good",
                    "evidence_trace_ids": ["trace"],
                    "alternatives": ["h2"],
                    "confidence": 0.8,
                },
                {
                    "hypothesis_id": "h2",
                    "family": "context",
                    "intervention_kind": "orchestration_change",
                    "mechanism": "alternative",
                    "target_boundary": "elsewhere",
                    "claim": "competing claim",
                    "evidence_trace_ids": ["trace"],
                    "alternatives": ["h1"],
                    "confidence": 0.2,
                },
            ],
            "target_hypothesis_id": "h1",
            "primary_family": "verification",
            "public_contract": "behavior must be good",
            "hidden_variant_strategy": "structural variant",
            "cases": [
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
                    "case_id": "positive_witness",
                    "tier": "admission",
                    "evidence_role": "positive_witness",
                    "path": "tests/hidden/positive_witness.py",
                    "expected_incumbent_pass": True,
                    "expected_candidate_pass": True,
                },
            ],
            "metadata": {},
        },
    )
    packet = load_test_packet(bundle)
    frozen = freeze_test_packet(bundle, packet, admission_passed=True)
    return frozen.content_sha256


def _stored_ref(tmp_path: Path) -> tuple[Path, FrozenPacketRef]:
    bundle = tmp_path / "authored"
    content_sha256 = _frozen_packet_bundle(bundle)
    packet_root = tmp_path / "frozen_packets"
    relative = copy_packet_into_store(
        packet_root=packet_root,
        packet_bundle=bundle,
        content_sha256=content_sha256,
    )
    ref = FrozenPacketRef(
        packet_id="latent-packet",
        path=relative,
        content_sha256=content_sha256,
    )
    return packet_root, ref


def test_replay_reports_realization_per_packet(tmp_path: Path) -> None:
    packet_root, ref = _stored_ref(tmp_path)
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _write_source(good, "good")
    _write_source(bad, "bad")
    replayer = PacketReplayer(packet_root=packet_root)

    realized = replayer.replay(
        refs=(ref,), candidate_source=good, output_dir=tmp_path / "out-good"
    )
    missed = replayer.replay(
        refs=(ref,), candidate_source=bad, output_dir=tmp_path / "out-bad"
    )

    assert [item.contract_passed for item in realized] == [True]
    assert realized[0].primary_family == "verification"
    assert realized[0].content_sha256 == ref.content_sha256
    assert [item.contract_passed for item in missed] == [False]
    assert missed[0].reasons


def test_replay_deduplicates_identical_packet_contents(tmp_path: Path) -> None:
    packet_root, ref = _stored_ref(tmp_path)
    good = tmp_path / "good"
    _write_source(good, "good")

    results = PacketReplayer(packet_root=packet_root).replay(
        refs=(ref, ref),
        candidate_source=good,
        output_dir=tmp_path / "out",
    )

    assert len(results) == 1


def test_modified_stored_packet_fails_closed(tmp_path: Path) -> None:
    packet_root, ref = _stored_ref(tmp_path)
    good = tmp_path / "good"
    _write_source(good, "good")
    tampered = packet_root / ref.path / "tests" / "hidden" / "sibling.py"
    tampered.write_text("raise SystemExit(0)\n", encoding="utf-8")

    with pytest.raises(ReplayError, match="modified"):
        PacketReplayer(packet_root=packet_root).replay(
            refs=(ref,),
            candidate_source=good,
            output_dir=tmp_path / "out",
        )


def test_missing_stored_packet_fails_closed(tmp_path: Path) -> None:
    packet_root = tmp_path / "frozen_packets"
    packet_root.mkdir()
    ref = FrozenPacketRef(
        packet_id="ghost",
        path="packets/" + "0" * 64,
        content_sha256="0" * 64,
    )

    with pytest.raises(ReplayError, match="missing"):
        PacketReplayer(packet_root=packet_root).replay(
            refs=(ref,),
            candidate_source=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_packet_store_rejects_path_escape() -> None:
    with pytest.raises(ValueError, match="portable path"):
        FrozenPacketRef(
            packet_id="escape",
            path="../outside",
            content_sha256="0" * 64,
        )
