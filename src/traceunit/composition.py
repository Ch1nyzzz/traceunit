from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from traceunit.archive import (
    ApplicationMode,
    ArchiveCatalog,
    CompositionPlan,
    FrozenPacketRef,
)
from traceunit.io import copy_source, safe_relative_path, sha256_file, sha256_tree
from traceunit.tests_runtime import (
    candidate_contract,
    load_test_packet,
    run_test_cases,
    verify_frozen_packet,
)


class CompositionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentReceipt:
    component_id: str
    mode: str
    before_source_sha256: str
    after_source_sha256: str
    applied: bool


@dataclass(frozen=True)
class MaterializationReceipt:
    attempt_fingerprint: str
    base_source_sha256: str
    materialized_source_sha256: str
    components: tuple[ComponentReceipt, ...]
    semantic_component_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["components"] = [asdict(item) for item in self.components]
        return value


@dataclass(frozen=True)
class ReplayResult:
    passed: bool
    packet_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompositionExecutor:
    """Apply exact archive patches in an isolated staging tree.

    Semantic selections are never applied implicitly. They are recorded for the
    Candidate Editor, whose resulting integration diff is verified later.
    """

    def __init__(self, archive_root: Path, catalog: ArchiveCatalog) -> None:
        self.archive_root = archive_root.resolve()
        self.catalog = catalog

    def materialize(
        self,
        *,
        plan: CompositionPlan,
        parent_source: Path,
        destination: Path,
    ) -> MaterializationReceipt:
        self.catalog.validate_plan(plan)
        base_hash = sha256_tree(parent_source)
        if base_hash != plan.base_source_sha256:
            raise CompositionError(
                "composition base hash does not match the frozen parent source"
            )

        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".traceunit-compose-", dir=str(destination.parent)
        ) as raw_stage:
            stage = Path(raw_stage) / "source"
            copy_source(parent_source, stage)
            receipts: list[ComponentReceipt] = []
            semantic: list[str] = []
            for selection in plan.selections:
                before = sha256_tree(stage)
                if selection.mode is ApplicationMode.SEMANTIC:
                    semantic.append(selection.component_id)
                    receipts.append(
                        ComponentReceipt(
                            component_id=selection.component_id,
                            mode=selection.mode.value,
                            before_source_sha256=before,
                            after_source_sha256=before,
                            applied=False,
                        )
                    )
                    continue
                manifest = self.catalog.get(selection.component_id)
                patch_path = safe_relative_path(self.archive_root, manifest.patch_path)
                if not patch_path.is_file():
                    raise CompositionError(f"archive patch is missing: {patch_path}")
                if sha256_file(patch_path) != manifest.patch_sha256:
                    raise CompositionError(
                        f"archive patch hash mismatch: {selection.component_id}"
                    )
                self._apply_patch(stage, patch_path, selection.component_id)
                receipts.append(
                    ComponentReceipt(
                        component_id=selection.component_id,
                        mode=selection.mode.value,
                        before_source_sha256=before,
                        after_source_sha256=sha256_tree(stage),
                        applied=True,
                    )
                )
            self._publish(stage, destination)

        return MaterializationReceipt(
            attempt_fingerprint=plan.attempt_fingerprint,
            base_source_sha256=base_hash,
            materialized_source_sha256=sha256_tree(destination),
            components=tuple(receipts),
            semantic_component_ids=tuple(semantic),
        )

    @staticmethod
    def _publish(stage: Path, destination: Path) -> None:
        """Publish a staged tree transactionally on the destination filesystem."""

        backup = destination.parent / (
            f".{destination.name}.traceunit-backup-{uuid.uuid4().hex}"
        )
        moved_existing = False
        try:
            if destination.exists():
                os.replace(destination, backup)
                moved_existing = True
            os.replace(stage, destination)
        except OSError as exc:
            if moved_existing and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise CompositionError(
                f"could not publish materialized composition: {exc}"
            ) from exc
        if moved_existing:
            shutil.rmtree(backup)

    @staticmethod
    def _apply_patch(source: Path, patch_path: Path, component_id: str) -> None:
        base = [
            "git",
            "apply",
            "--unsafe-paths",
            "--whitespace=nowarn",
            str(patch_path),
        ]
        checked = subprocess.run(
            [*base[:2], "--check", *base[2:]],
            cwd=source,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if checked.returncode != 0:
            raise CompositionError(
                f"exact patch {component_id} does not apply cleanly: "
                f"{checked.stdout[-2000:]}"
            )
        applied = subprocess.run(
            base,
            cwd=source,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if applied.returncode != 0:
            raise CompositionError(
                f"exact patch {component_id} failed after check: "
                f"{applied.stdout[-2000:]}"
            )


class CertificateReplayer:
    def __init__(self, *, archive_root: Path, python: Path | None = None) -> None:
        self.archive_root = archive_root.resolve()
        self.python = python

    def replay(
        self,
        *,
        refs: Iterable[FrozenPacketRef],
        candidate_source: Path,
        output_dir: Path,
    ) -> ReplayResult:
        packet_ids: list[str] = []
        reasons: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            if ref.content_sha256 in seen:
                continue
            seen.add(ref.content_sha256)
            packet_ids.append(ref.packet_id)
            bundle = safe_relative_path(self.archive_root, ref.path)
            packet = load_test_packet(bundle)
            if packet.content_sha256 != ref.content_sha256:
                reasons.append(f"{ref.packet_id}: declared packet hash mismatch")
                continue
            if not verify_frozen_packet(bundle, packet):
                reasons.append(f"{ref.packet_id}: frozen packet was modified")
                continue
            results = run_test_cases(
                packet=packet,
                bundle=bundle,
                source=candidate_source,
                subject="candidate",
                output_dir=output_dir / ref.content_sha256[:16],
                python=self.python,
            )
            passed, packet_reasons = candidate_contract(packet, results)
            reasons.extend(f"{ref.packet_id}: {reason}" for reason in packet_reasons)
            if not passed and not packet_reasons:
                reasons.append(f"{ref.packet_id}: candidate contract failed")
        return ReplayResult(
            passed=not reasons,
            packet_ids=tuple(packet_ids),
            reasons=tuple(reasons),
        )


def copy_packet_into_archive(
    *,
    archive_root: Path,
    packet_bundle: Path,
    content_sha256: str,
) -> str:
    """Copy a frozen packet into private archive storage and return a portable path."""

    relative = Path("packets") / content_sha256
    destination = archive_root / relative
    if destination.exists():
        packet = load_test_packet(destination)
        if not verify_frozen_packet(destination, packet):
            raise CompositionError(
                f"existing archived packet is corrupt: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(packet_bundle, destination)
    return relative.as_posix()
