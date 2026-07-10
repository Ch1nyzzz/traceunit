from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from traceunit.archive import (
    ApplicationMode,
    ArchiveCatalog,
    ArchiveKind,
    ArchiveManifest,
    ArchiveValidationError,
    ComponentSelection,
    CompositionPlan,
    DependencyCycleError,
    FrozenPacketRef,
    LocalCertificate,
    UnknownComponentError,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _certificate(label: str, *, packet_id: str | None = None) -> LocalCertificate:
    return LocalCertificate(
        frozen_packet_refs=(
            FrozenPacketRef(
                packet_id=packet_id or f"packet-{label}",
                path=f"packets/{label}.json",
                content_sha256=_sha(f"packet:{label}"),
            ),
        ),
        family_keys=(f"family.{label}",),
        public_passed=True,
        hidden_passed=True,
        bridge_passed=True,
        regression_passed=True,
        public_score=1.0,
        hidden_score=0.9,
        bridge_score=0.8,
        regression_loss=0.0,
        case_results_path=f"evidence/{label}.json",
        case_results_sha256=_sha(f"evidence:{label}"),
    )


def _atomic(
    label: str,
    *,
    dependencies: tuple[str, ...] = (),
    certificate: LocalCertificate | None = None,
) -> ArchiveManifest:
    return ArchiveManifest(
        kind=ArchiveKind.ATOMIC,
        parent_source_sha256=_sha(f"parent:{label}"),
        candidate_source_sha256=_sha(f"candidate:{label}"),
        patch_path=f"patches/{label}.diff",
        patch_sha256=_sha(f"patch:{label}"),
        certificate=certificate or _certificate(label),
        mechanism=f"mechanism-{label}",
        target_boundary=f"boundary-{label}",
        dependencies=dependencies,
        trace_signature=(f"trace:{label}",),
        applicability=(f"has:{label}",),
    )


def _composite(
    label: str,
    *,
    dependencies: tuple[str, ...],
    constituents: tuple[str, ...],
) -> ArchiveManifest:
    return ArchiveManifest(
        kind=ArchiveKind.COMPOSITE,
        parent_source_sha256=_sha(f"parent:{label}"),
        candidate_source_sha256=_sha(f"candidate:{label}"),
        patch_path=f"patches/{label}.diff",
        patch_sha256=_sha(f"patch:{label}"),
        certificate=_certificate(label),
        mechanism=f"mechanism-{label}",
        target_boundary=f"boundary-{label}",
        dependencies=dependencies,
        constituents=constituents,
        integration_residual_path=f"patches/{label}.residual.diff",
        integration_residual_sha256=_sha(f"residual:{label}"),
    )


def test_manifest_is_content_addressed_and_portable_paths_do_not_change_id() -> None:
    original = _atomic("a")
    moved_certificate = replace(
        original.certificate,
        frozen_packet_refs=(
            replace(
                original.certificate.frozen_packet_refs[0],
                path="relocated/packet.json",
            ),
        ),
        case_results_path="relocated/evidence.json",
    )
    moved = replace(
        original,
        patch_path="relocated/edit.diff",
        certificate=moved_certificate,
        archive_id="",
    )

    assert original.archive_id == original.content_sha256
    assert moved.archive_id == original.archive_id
    assert original.to_dict()["patch_path"] != moved.to_dict()["patch_path"]

    round_trip = ArchiveManifest.from_dict(original.to_dict())
    assert round_trip == original

    tampered = original.to_dict()
    tampered["mechanism"] = "different-mechanism"
    with pytest.raises(ArchiveValidationError, match="archive_id does not match"):
        ArchiveManifest.from_dict(tampered)


def test_manifest_validates_atomic_and_composite_shapes() -> None:
    atomic = _atomic("a")

    with pytest.raises(ArchiveValidationError, match="cannot have constituents"):
        replace(atomic, constituents=(atomic.archive_id,), archive_id="")

    with pytest.raises(ArchiveValidationError, match="at least one constituent"):
        ArchiveManifest(
            kind=ArchiveKind.COMPOSITE,
            parent_source_sha256=_sha("parent"),
            candidate_source_sha256=_sha("candidate"),
            patch_path="patch.diff",
            patch_sha256=_sha("patch"),
            certificate=_certificate("composite"),
            mechanism="compose",
            target_boundary="planner",
        )

    with pytest.raises(ArchiveValidationError, match="provided together"):
        replace(
            atomic,
            integration_residual_path="residual.diff",
            archive_id="",
        )

    with pytest.raises(ArchiveValidationError, match="portable path"):
        replace(atomic, patch_path="../escape.diff", archive_id="")


def test_certificate_is_immutable_mechanical_evidence() -> None:
    certificate = _certificate("valid")

    assert certificate.mechanically_valid
    assert len(certificate.certificate_sha256) == 64

    failed = replace(certificate, bridge_passed=False)
    assert not failed.mechanically_valid
    assert failed.certificate_sha256 != certificate.certificate_sha256

    tampered = certificate.to_dict()
    tampered["hidden_score"] = 0.1
    with pytest.raises(ArchiveValidationError, match="certificate content hash"):
        LocalCertificate.from_dict(tampered)


def test_catalog_dependency_closure_deduplicates_and_orders() -> None:
    first = _atomic("first")
    second = _atomic("second", dependencies=(first.archive_id,))
    third = _atomic("third", dependencies=(first.archive_id,))
    composite = _composite(
        "combined",
        dependencies=(third.archive_id, second.archive_id),
        constituents=(first.archive_id, second.archive_id, third.archive_id),
    )
    catalog = ArchiveCatalog((composite, third, first, second))

    closure = catalog.dependency_closure(
        (composite.archive_id, second.archive_id, composite.archive_id)
    )
    ids = tuple(item.archive_id for item in closure)

    assert ids[-1] == composite.archive_id
    assert ids.count(first.archive_id) == 1
    assert ids.index(first.archive_id) < ids.index(second.archive_id)
    assert ids.index(first.archive_id) < ids.index(third.archive_id)
    assert ids.index(second.archive_id) < ids.index(composite.archive_id)
    assert ids.index(third.archive_id) < ids.index(composite.archive_id)


def test_catalog_loads_manifests_and_rejects_missing_references(
    tmp_path: Path,
) -> None:
    first = _atomic("first")
    second = _atomic("second", dependencies=(first.archive_id,))
    for index, manifest in enumerate((second, first)):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(manifest.to_dict()), encoding="utf-8"
        )

    loaded = ArchiveCatalog.load(tmp_path)
    assert len(loaded) == 2
    assert loaded.get(first.archive_id) == first
    assert loaded.register(first) == first.archive_id
    assert len(loaded) == 2

    incomplete = ArchiveCatalog((second,))
    with pytest.raises(UnknownComponentError):
        incomplete.validate()


def test_catalog_detects_corrupted_reference_cycles() -> None:
    first = _atomic("first")
    second = _atomic("second")
    catalog = ArchiveCatalog((first, second))

    # Valid content-addressed manifests cannot normally create a reference cycle:
    # their IDs commit to their references.  Mutating the frozen objects emulates
    # a corrupted in-memory catalog and exercises the defensive graph check.
    object.__setattr__(first, "dependencies", (second.archive_id,))
    object.__setattr__(second, "dependencies", (first.archive_id,))

    with pytest.raises(DependencyCycleError, match="reference cycle"):
        catalog.validate()


def test_make_plan_supports_many_components_modes_and_integration_edit() -> None:
    components = [_atomic("zero")]
    for index in range(1, 5):
        components.append(
            _atomic(
                str(index),
                dependencies=(components[index - 1].archive_id,),
            )
        )
    catalog = ArchiveCatalog(reversed(components))
    requested = ComponentSelection(
        component_id=components[-1].archive_id,
        mode=ApplicationMode.SEMANTIC,
        semantic_instructions="Port the behavior to the current planner boundary.",
        rationale="Matches the current trace.",
    )

    plan = catalog.make_plan(
        base_source_sha256=_sha("base"),
        selections=(requested,),
        integration_edit_path="composition/glue.diff",
        integration_edit_sha256=_sha("glue"),
        integration_instructions="Resolve the interaction between components.",
    )

    assert len(plan.selections) == 5
    assert plan.selections[-1] == requested
    assert all(
        selection.mode is ApplicationMode.EXACT for selection in plan.selections[:-1]
    )
    catalog.validate_plan(plan)
    assert len(plan.attempt_fingerprint) == 64

    equivalent = replace(
        plan,
        selections=tuple(
            replace(item, rationale="different explanation") for item in plan.selections
        ),
        integration_edit_path="moved/glue.diff",
    )
    assert equivalent.attempt_fingerprint == plan.attempt_fingerprint

    changed_mode = replace(
        plan.selections[0],
        mode=ApplicationMode.SEMANTIC,
        semantic_instructions="Reimplement this dependency.",
    )
    changed = replace(plan, selections=(changed_mode, *plan.selections[1:]))
    assert changed.attempt_fingerprint != plan.attempt_fingerprint


def test_plan_allows_zero_components_and_rejects_invalid_order() -> None:
    empty = CompositionPlan(base_source_sha256=_sha("base"))
    assert empty.component_ids == ()

    with pytest.raises(ArchiveValidationError, match="semantic_instructions"):
        ComponentSelection(
            component_id=_sha("component"),
            mode=ApplicationMode.SEMANTIC,
        )

    dependency = _atomic("dependency")
    dependant = _atomic("dependant", dependencies=(dependency.archive_id,))
    catalog = ArchiveCatalog((dependency, dependant))

    omitted = CompositionPlan(
        base_source_sha256=_sha("base"),
        selections=(ComponentSelection(dependant.archive_id),),
    )
    with pytest.raises(ArchiveValidationError, match="omits dependency"):
        catalog.validate_plan(omitted)

    reversed_plan = CompositionPlan(
        base_source_sha256=_sha("base"),
        selections=(
            ComponentSelection(dependant.archive_id),
            ComponentSelection(dependency.archive_id),
        ),
    )
    with pytest.raises(ArchiveValidationError, match="must precede"):
        catalog.validate_plan(reversed_plan)


def test_frozen_packet_replay_includes_composite_constituents_and_deduplicates() -> (
    None
):
    shared_packet_hash = _sha("shared-packet")
    first_certificate = replace(
        _certificate("first", packet_id="shared"),
        frozen_packet_refs=(
            FrozenPacketRef("shared", "packets/first.json", shared_packet_hash),
        ),
    )
    second_certificate = replace(
        _certificate("second", packet_id="shared"),
        frozen_packet_refs=(
            FrozenPacketRef("shared", "packets/second.json", shared_packet_hash),
        ),
    )
    first = _atomic("first", certificate=first_certificate)
    second = _atomic("second", certificate=second_certificate)
    composite = _composite(
        "combined",
        dependencies=(),
        constituents=(first.archive_id, second.archive_id),
    )
    catalog = ArchiveCatalog((composite, first, second))
    plan = catalog.make_plan(
        base_source_sha256=_sha("base"),
        selections=(ComponentSelection(composite.archive_id),),
    )

    refs = plan.frozen_packet_refs(catalog)

    assert {ref.packet_id for ref in refs} == {"shared", "packet-combined"}
    assert len(refs) == 2

    conflicting_second = replace(
        second,
        certificate=replace(
            second.certificate,
            frozen_packet_refs=(
                FrozenPacketRef("shared", "packets/bad.json", _sha("different")),
            ),
        ),
        archive_id="",
    )
    conflicting_composite = _composite(
        "conflicting",
        dependencies=(),
        constituents=(first.archive_id, conflicting_second.archive_id),
    )
    conflicting_catalog = ArchiveCatalog(
        (first, conflicting_second, conflicting_composite)
    )
    with pytest.raises(ArchiveValidationError, match="conflicting frozen contents"):
        conflicting_catalog.frozen_packet_refs((conflicting_composite.archive_id,))


def test_manifest_registration_is_idempotent_across_artifact_locations() -> None:
    original = _atomic("portable")
    relocated = replace(original, patch_path="elsewhere/edit.diff", archive_id="")
    catalog = ArchiveCatalog((original,))

    assert catalog.register(relocated) == original.archive_id
    assert len(catalog) == 1
