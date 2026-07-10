from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from traceunit.archive import (
    ApplicationMode,
    ArchiveCatalog,
    ArchiveKind,
    ArchiveManifest,
    ComponentSelection,
    FrozenPacketRef,
    LocalCertificate,
)
from traceunit.composition import CompositionError, CompositionExecutor
from traceunit.io import sha256_file, sha256_tree


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_source(root: Path, value: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "agent.txt").write_text(f"{value}\n", encoding="utf-8")


def _write_replace_patch(
    path: Path,
    *,
    before: str,
    after: str,
    filename: str = "agent.txt",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"diff --git a/{filename} b/{filename}",
                f"--- a/{filename}",
                f"+++ b/{filename}",
                "@@ -1 +1 @@",
                f"-{before}",
                f"+{after}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _certificate(label: str) -> LocalCertificate:
    return LocalCertificate(
        frozen_packet_refs=(
            FrozenPacketRef(
                packet_id=f"packet-{label}",
                path=f"packets/{label}",
                content_sha256=_sha(f"packet:{label}"),
            ),
        ),
        family_keys=(f"family.{label}",),
        public_passed=True,
        hidden_passed=True,
        bridge_passed=True,
        regression_passed=True,
    )


def _manifest(
    *,
    label: str,
    archive_root: Path,
    patch_path: Path,
    parent_source_sha256: str,
    candidate_source_sha256: str,
    dependencies: tuple[str, ...] = (),
) -> ArchiveManifest:
    return ArchiveManifest(
        kind=ArchiveKind.ATOMIC,
        parent_source_sha256=parent_source_sha256,
        candidate_source_sha256=candidate_source_sha256,
        patch_path=patch_path.relative_to(archive_root).as_posix(),
        patch_sha256=sha256_file(patch_path),
        certificate=_certificate(label),
        mechanism=f"mechanism-{label}",
        target_boundary="agent.txt",
        dependencies=dependencies,
    )


def test_exact_patch_is_really_applied_and_receipted(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    expected = tmp_path / "expected"
    archive_root = tmp_path / "archive"
    destination = tmp_path / "materialized"
    _write_source(parent, "before")
    _write_source(expected, "after")
    patch = archive_root / "patches" / "change.diff"
    _write_replace_patch(patch, before="before", after="after")
    manifest = _manifest(
        label="change",
        archive_root=archive_root,
        patch_path=patch,
        parent_source_sha256=sha256_tree(parent),
        candidate_source_sha256=sha256_tree(expected),
    )
    catalog = ArchiveCatalog((manifest,))
    plan = catalog.make_plan(
        base_source_sha256=sha256_tree(parent),
        selections=(ComponentSelection(manifest.archive_id),),
    )

    receipt = CompositionExecutor(archive_root, catalog).materialize(
        plan=plan,
        parent_source=parent,
        destination=destination,
    )

    assert (destination / "agent.txt").read_text(encoding="utf-8") == "after\n"
    assert receipt.materialized_source_sha256 == sha256_tree(expected)
    assert receipt.semantic_component_ids == ()
    assert len(receipt.components) == 1
    component = receipt.components[0]
    assert component.component_id == manifest.archive_id
    assert component.mode == ApplicationMode.EXACT.value
    assert component.applied
    assert component.before_source_sha256 == sha256_tree(parent)
    assert component.after_source_sha256 == sha256_tree(expected)


def test_more_than_three_components_apply_in_dependency_order(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    expected = tmp_path / "expected"
    archive_root = tmp_path / "archive"
    destination = tmp_path / "materialized"
    _write_source(parent, "0")
    _write_source(expected, "0")

    manifests: list[ArchiveManifest] = []
    for index in range(5):
        before_hash = sha256_tree(expected)
        patch = archive_root / "patches" / f"step-{index}.diff"
        _write_replace_patch(patch, before=str(index), after=str(index + 1))
        _write_source(expected, str(index + 1))
        dependencies = (manifests[-1].archive_id,) if manifests else ()
        manifests.append(
            _manifest(
                label=f"step-{index}",
                archive_root=archive_root,
                patch_path=patch,
                parent_source_sha256=before_hash,
                candidate_source_sha256=sha256_tree(expected),
                dependencies=dependencies,
            )
        )

    catalog = ArchiveCatalog(reversed(manifests))
    plan = catalog.make_plan(
        base_source_sha256=sha256_tree(parent),
        selections=(ComponentSelection(manifests[-1].archive_id),),
    )
    receipt = CompositionExecutor(archive_root, catalog).materialize(
        plan=plan,
        parent_source=parent,
        destination=destination,
    )

    assert len(plan.selections) == 5
    assert plan.component_ids == tuple(item.archive_id for item in manifests)
    assert len(receipt.components) == 5
    assert all(item.applied for item in receipt.components)
    assert (destination / "agent.txt").read_text(encoding="utf-8") == "5\n"
    assert receipt.materialized_source_sha256 == sha256_tree(expected)


def test_base_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    expected = tmp_path / "expected"
    archive_root = tmp_path / "archive"
    destination = tmp_path / "materialized"
    _write_source(parent, "before")
    _write_source(expected, "after")
    _write_source(destination, "existing-output")
    patch = archive_root / "patches" / "change.diff"
    _write_replace_patch(patch, before="before", after="after")
    manifest = _manifest(
        label="change",
        archive_root=archive_root,
        patch_path=patch,
        parent_source_sha256=sha256_tree(parent),
        candidate_source_sha256=sha256_tree(expected),
    )
    catalog = ArchiveCatalog((manifest,))
    plan = catalog.make_plan(
        base_source_sha256=_sha("not-the-parent-tree"),
        selections=(ComponentSelection(manifest.archive_id),),
    )

    with pytest.raises(CompositionError, match="base hash does not match"):
        CompositionExecutor(archive_root, catalog).materialize(
            plan=plan,
            parent_source=parent,
            destination=destination,
        )

    assert (parent / "agent.txt").read_text(encoding="utf-8") == "before\n"
    assert (destination / "agent.txt").read_text(encoding="utf-8") == (
        "existing-output\n"
    )


def test_patch_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    expected = tmp_path / "expected"
    archive_root = tmp_path / "archive"
    destination = tmp_path / "materialized"
    _write_source(parent, "before")
    _write_source(expected, "after")
    _write_source(destination, "existing-output")
    patch = archive_root / "patches" / "change.diff"
    _write_replace_patch(patch, before="before", after="after")
    manifest = _manifest(
        label="change",
        archive_root=archive_root,
        patch_path=patch,
        parent_source_sha256=sha256_tree(parent),
        candidate_source_sha256=sha256_tree(expected),
    )
    catalog = ArchiveCatalog((manifest,))
    plan = catalog.make_plan(
        base_source_sha256=sha256_tree(parent),
        selections=(ComponentSelection(manifest.archive_id),),
    )
    patch.write_text("tampered patch\n", encoding="utf-8")

    with pytest.raises(CompositionError, match="patch hash mismatch"):
        CompositionExecutor(archive_root, catalog).materialize(
            plan=plan,
            parent_source=parent,
            destination=destination,
        )

    assert (parent / "agent.txt").read_text(encoding="utf-8") == "before\n"
    assert (destination / "agent.txt").read_text(encoding="utf-8") == (
        "existing-output\n"
    )


def test_semantic_selection_is_recorded_but_never_applied_as_exact(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    expected_if_applied = tmp_path / "expected-if-applied"
    archive_root = tmp_path / "archive"
    destination = tmp_path / "materialized"
    _write_source(parent, "before")
    _write_source(expected_if_applied, "semantic-change")
    patch = archive_root / "patches" / "semantic.diff"
    _write_replace_patch(patch, before="before", after="semantic-change")
    manifest = _manifest(
        label="semantic",
        archive_root=archive_root,
        patch_path=patch,
        parent_source_sha256=sha256_tree(parent),
        candidate_source_sha256=sha256_tree(expected_if_applied),
    )
    catalog = ArchiveCatalog((manifest,))
    plan = catalog.make_plan(
        base_source_sha256=sha256_tree(parent),
        selections=(
            ComponentSelection(
                component_id=manifest.archive_id,
                mode=ApplicationMode.SEMANTIC,
                semantic_instructions="Port the behavior without replaying this diff.",
            ),
        ),
    )

    # Even a missing exact patch must not turn a semantic selection into replay.
    patch.unlink()
    receipt = CompositionExecutor(archive_root, catalog).materialize(
        plan=plan,
        parent_source=parent,
        destination=destination,
    )

    assert (destination / "agent.txt").read_text(encoding="utf-8") == "before\n"
    assert receipt.materialized_source_sha256 == sha256_tree(parent)
    assert receipt.semantic_component_ids == (manifest.archive_id,)
    assert len(receipt.components) == 1
    component = receipt.components[0]
    assert component.mode == ApplicationMode.SEMANTIC.value
    assert not component.applied
    assert component.before_source_sha256 == component.after_source_sha256


def test_late_patch_failure_does_not_publish_partial_output(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    archive_root = tmp_path / "archive"
    destination = tmp_path / "materialized"
    _write_source(parent, "before")
    _write_source(destination, "existing-output")

    first_patch = archive_root / "patches" / "first.diff"
    _write_replace_patch(first_patch, before="before", after="intermediate")
    first = _manifest(
        label="first",
        archive_root=archive_root,
        patch_path=first_patch,
        parent_source_sha256=sha256_tree(parent),
        candidate_source_sha256=_sha("intermediate-tree"),
    )

    failing_patch = archive_root / "patches" / "failing.diff"
    _write_replace_patch(failing_patch, before="not-intermediate", after="final")
    failing = _manifest(
        label="failing",
        archive_root=archive_root,
        patch_path=failing_patch,
        parent_source_sha256=_sha("intermediate-tree"),
        candidate_source_sha256=_sha("final-tree"),
        dependencies=(first.archive_id,),
    )
    catalog = ArchiveCatalog((failing, first))
    plan = catalog.make_plan(
        base_source_sha256=sha256_tree(parent),
        selections=(ComponentSelection(failing.archive_id),),
    )

    with pytest.raises(CompositionError, match="does not apply cleanly"):
        CompositionExecutor(archive_root, catalog).materialize(
            plan=plan,
            parent_source=parent,
            destination=destination,
        )

    assert (parent / "agent.txt").read_text(encoding="utf-8") == "before\n"
    assert (destination / "agent.txt").read_text(encoding="utf-8") == (
        "existing-output\n"
    )
    assert not any(tmp_path.glob(".traceunit-compose-*"))
