"""Content-addressed manifests for certified partial edits.

This module deliberately stops at data modelling and validation.  It does not
apply patches, modify source trees, invoke Git, or decide whether a component
should be promoted.  Those side effects belong to the search controller.

An archive component has two kinds of references:

* ``dependencies`` must be applied before the component;
* ``constituents`` record the provenance of a composite component.

The component's canonical ``patch`` is the complete replay artifact.  A
composite may additionally retain an ``integration_residual`` describing the
glue edit made after its constituents were combined.  Constituents therefore
participate in integrity checks and certificate replay, but are not
automatically applied unless they are also declared as dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArchiveValidationError(ValueError):
    """Raised when an archive manifest or composition plan is malformed."""


class UnknownComponentError(KeyError):
    """Raised when a catalog reference does not resolve to a manifest."""


class DependencyCycleError(ArchiveValidationError):
    """Raised when dependency or composite provenance references form a cycle."""


class ArchiveKind(StrEnum):
    ATOMIC = "atomic"
    COMPOSITE = "composite"


class ApplicationMode(StrEnum):
    EXACT = "exact"
    SEMANTIC = "semantic"


def _require_text(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ArchiveValidationError(f"{field_name} must not be empty")
    return result


def _require_sha256(value: str, field_name: str) -> str:
    result = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise ArchiveValidationError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest"
        )
    return result


def _optional_artifact(path: str, digest: str, *, field_name: str) -> tuple[str, str]:
    normalized_path = str(path).strip()
    normalized_digest = str(digest).strip().lower()
    if bool(normalized_path) != bool(normalized_digest):
        raise ArchiveValidationError(
            f"{field_name}_path and {field_name}_sha256 must be provided together"
        )
    if not normalized_path:
        return "", ""
    return (
        _require_relative_path(normalized_path, f"{field_name}_path"),
        _require_sha256(normalized_digest, f"{field_name}_sha256"),
    )


def _require_relative_path(value: str, field_name: str) -> str:
    raw = _require_text(value, field_name).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveValidationError(
            f"{field_name} must be a portable path relative to the archive root"
        )
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ArchiveValidationError(f"{field_name} must name an artifact")
    return normalized


def _unique_strings(
    values: Iterable[str], *, field_name: str, sort: bool = True
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _require_text(str(raw), field_name)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    if sort:
        result.sort()
    return tuple(result)


def _sha256_strings(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    return tuple(sorted({_require_sha256(str(value), field_name) for value in values}))


def _finite(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ArchiveValidationError(f"{field_name} must be finite")
    return result


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenPacketRef:
    """A portable reference to one immutable test packet."""

    packet_id: str
    path: str
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "packet_id", _require_text(self.packet_id, "packet_id")
        )
        object.__setattr__(
            self, "path", _require_relative_path(self.path, "packet path")
        )
        object.__setattr__(
            self,
            "content_sha256",
            _require_sha256(self.content_sha256, "packet content_sha256"),
        )

    def identity_dict(self) -> dict[str, str]:
        """Return path-independent packet identity used by content addressing."""

        return {
            "packet_id": self.packet_id,
            "content_sha256": self.content_sha256,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "packet_id": self.packet_id,
            "path": self.path,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenPacketRef":
        return cls(
            packet_id=str(value["packet_id"]),
            path=str(value["path"]),
            content_sha256=str(value["content_sha256"]),
        )


@dataclass(frozen=True)
class LocalCertificate:
    """Immutable mechanical evidence attached to an archived component.

    Scores are stored as observations, while the four booleans record the
    controller's already-made pass/fail judgements.  The certificate does not
    infer thresholds and does not contain a mutable alignment prior.
    """

    frozen_packet_refs: tuple[FrozenPacketRef, ...]
    family_keys: tuple[str, ...]
    public_passed: bool
    hidden_passed: bool
    bridge_passed: bool
    regression_passed: bool
    public_score: float = 0.0
    hidden_score: float = 0.0
    bridge_score: float = 0.0
    regression_loss: float = 0.0
    case_results_path: str = ""
    case_results_sha256: str = ""

    def __post_init__(self) -> None:
        refs = tuple(self.frozen_packet_refs)
        if not refs:
            raise ArchiveValidationError(
                "frozen_packet_refs must contain at least one packet"
            )

        by_packet_id: dict[str, FrozenPacketRef] = {}
        by_content: dict[str, FrozenPacketRef] = {}
        for ref in refs:
            if not isinstance(ref, FrozenPacketRef):
                raise ArchiveValidationError(
                    "frozen_packet_refs must contain FrozenPacketRef values"
                )
            prior = by_packet_id.get(ref.packet_id)
            if prior is not None and prior.content_sha256 != ref.content_sha256:
                raise ArchiveValidationError(
                    f"packet_id {ref.packet_id!r} refers to multiple content hashes"
                )
            by_packet_id[ref.packet_id] = ref
            by_content.setdefault(ref.content_sha256, ref)

        normalized_refs = tuple(
            sorted(
                by_content.values(),
                key=lambda item: (item.packet_id, item.content_sha256, item.path),
            )
        )
        object.__setattr__(self, "frozen_packet_refs", normalized_refs)

        families = _unique_strings(self.family_keys, field_name="family_key")
        if not families:
            raise ArchiveValidationError("family_keys must contain at least one family")
        object.__setattr__(self, "family_keys", families)

        for name in (
            "public_score",
            "hidden_score",
            "bridge_score",
            "regression_loss",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))

        result_path, result_hash = _optional_artifact(
            self.case_results_path,
            self.case_results_sha256,
            field_name="case_results",
        )
        object.__setattr__(self, "case_results_path", result_path)
        object.__setattr__(self, "case_results_sha256", result_hash)

    @property
    def mechanically_valid(self) -> bool:
        return all(
            (
                self.public_passed,
                self.hidden_passed,
                self.bridge_passed,
                self.regression_passed,
            )
        )

    @property
    def certificate_sha256(self) -> str:
        return _fingerprint(self.identity_dict())

    def identity_dict(self) -> dict[str, Any]:
        """Return the path-independent facts covered by the certificate hash."""

        return {
            "frozen_packet_refs": [
                item.identity_dict() for item in self.frozen_packet_refs
            ],
            "family_keys": list(self.family_keys),
            "public_passed": self.public_passed,
            "hidden_passed": self.hidden_passed,
            "bridge_passed": self.bridge_passed,
            "regression_passed": self.regression_passed,
            "public_score": self.public_score,
            "hidden_score": self.hidden_score,
            "bridge_score": self.bridge_score,
            "regression_loss": self.regression_loss,
            "case_results_sha256": self.case_results_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen_packet_refs": [item.to_dict() for item in self.frozen_packet_refs],
            "family_keys": list(self.family_keys),
            "public_passed": self.public_passed,
            "hidden_passed": self.hidden_passed,
            "bridge_passed": self.bridge_passed,
            "regression_passed": self.regression_passed,
            "public_score": self.public_score,
            "hidden_score": self.hidden_score,
            "bridge_score": self.bridge_score,
            "regression_loss": self.regression_loss,
            "case_results_path": self.case_results_path,
            "case_results_sha256": self.case_results_sha256,
            "certificate_sha256": self.certificate_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalCertificate":
        certificate = cls(
            frozen_packet_refs=tuple(
                FrozenPacketRef.from_dict(item)
                for item in value.get("frozen_packet_refs") or []
                if isinstance(item, Mapping)
            ),
            family_keys=tuple(str(item) for item in value.get("family_keys") or []),
            public_passed=bool(value.get("public_passed")),
            hidden_passed=bool(value.get("hidden_passed")),
            bridge_passed=bool(value.get("bridge_passed")),
            regression_passed=bool(value.get("regression_passed")),
            public_score=float(value.get("public_score") or 0.0),
            hidden_score=float(value.get("hidden_score") or 0.0),
            bridge_score=float(value.get("bridge_score") or 0.0),
            regression_loss=float(value.get("regression_loss") or 0.0),
            case_results_path=str(value.get("case_results_path") or ""),
            case_results_sha256=str(value.get("case_results_sha256") or ""),
        )
        declared = str(value.get("certificate_sha256") or "")
        if declared and declared != certificate.certificate_sha256:
            raise ArchiveValidationError(
                "local certificate content hash does not match"
            )
        return certificate


@dataclass(frozen=True)
class ArchiveManifest:
    """Content-addressed description of one atomic or composite edit."""

    kind: ArchiveKind
    parent_source_sha256: str
    candidate_source_sha256: str
    patch_path: str
    patch_sha256: str
    certificate: LocalCertificate
    mechanism: str
    target_boundary: str
    dependencies: tuple[str, ...] = ()
    constituents: tuple[str, ...] = ()
    integration_residual_path: str = ""
    integration_residual_sha256: str = ""
    trace_signature: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()
    schema_version: int = 1
    archive_id: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ArchiveKind(str(self.kind)))
        object.__setattr__(
            self,
            "parent_source_sha256",
            _require_sha256(self.parent_source_sha256, "parent_source_sha256"),
        )
        object.__setattr__(
            self,
            "candidate_source_sha256",
            _require_sha256(self.candidate_source_sha256, "candidate_source_sha256"),
        )
        if self.parent_source_sha256 == self.candidate_source_sha256:
            raise ArchiveValidationError(
                "parent and candidate source hashes must be different"
            )

        object.__setattr__(
            self, "patch_path", _require_relative_path(self.patch_path, "patch_path")
        )
        object.__setattr__(
            self, "patch_sha256", _require_sha256(self.patch_sha256, "patch_sha256")
        )
        if not isinstance(self.certificate, LocalCertificate):
            raise ArchiveValidationError("certificate must be a LocalCertificate")
        object.__setattr__(
            self, "mechanism", _require_text(self.mechanism, "mechanism")
        )
        object.__setattr__(
            self,
            "target_boundary",
            _require_text(self.target_boundary, "target_boundary"),
        )

        dependencies = _sha256_strings(self.dependencies, field_name="dependency")
        constituents = _sha256_strings(self.constituents, field_name="constituent")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "constituents", constituents)

        if self.kind is ArchiveKind.ATOMIC and constituents:
            raise ArchiveValidationError("atomic manifests cannot have constituents")
        if self.kind is ArchiveKind.COMPOSITE and not constituents:
            raise ArchiveValidationError(
                "composite manifests must name at least one constituent"
            )

        residual_path, residual_hash = _optional_artifact(
            self.integration_residual_path,
            self.integration_residual_sha256,
            field_name="integration_residual",
        )
        if self.kind is ArchiveKind.ATOMIC and residual_path:
            raise ArchiveValidationError(
                "atomic manifests cannot have an integration residual"
            )
        object.__setattr__(self, "integration_residual_path", residual_path)
        object.__setattr__(self, "integration_residual_sha256", residual_hash)

        object.__setattr__(
            self,
            "trace_signature",
            _unique_strings(self.trace_signature, field_name="trace_signature"),
        )
        object.__setattr__(
            self,
            "applicability",
            _unique_strings(self.applicability, field_name="applicability"),
        )
        if int(self.schema_version) != 1:
            raise ArchiveValidationError("unsupported archive schema_version")
        object.__setattr__(self, "schema_version", 1)

        computed = _fingerprint(self.identity_dict())
        declared = str(self.archive_id).strip().lower()
        if declared and _require_sha256(declared, "archive_id") != computed:
            raise ArchiveValidationError("archive_id does not match manifest content")
        object.__setattr__(self, "archive_id", computed)

        if self.archive_id in set(self.dependencies) | set(self.constituents):
            raise ArchiveValidationError("a manifest cannot reference itself")

    @property
    def content_sha256(self) -> str:
        return self.archive_id

    @property
    def references(self) -> tuple[str, ...]:
        """All dependency and provenance references, deterministically deduplicated."""

        return tuple(sorted(set(self.dependencies) | set(self.constituents)))

    def identity_dict(self) -> dict[str, Any]:
        """Return semantic content; portable artifact paths are intentionally absent."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "parent_source_sha256": self.parent_source_sha256,
            "candidate_source_sha256": self.candidate_source_sha256,
            "patch_sha256": self.patch_sha256,
            "certificate": self.certificate.identity_dict(),
            "mechanism": self.mechanism,
            "target_boundary": self.target_boundary,
            "dependencies": list(self.dependencies),
            "constituents": list(self.constituents),
            "integration_residual_sha256": self.integration_residual_sha256,
            "trace_signature": list(self.trace_signature),
            "applicability": list(self.applicability),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "archive_id": self.archive_id,
            "kind": self.kind.value,
            "parent_source_sha256": self.parent_source_sha256,
            "candidate_source_sha256": self.candidate_source_sha256,
            "patch_path": self.patch_path,
            "patch_sha256": self.patch_sha256,
            "certificate": self.certificate.to_dict(),
            "mechanism": self.mechanism,
            "target_boundary": self.target_boundary,
            "dependencies": list(self.dependencies),
            "constituents": list(self.constituents),
            "integration_residual_path": self.integration_residual_path,
            "integration_residual_sha256": self.integration_residual_sha256,
            "trace_signature": list(self.trace_signature),
            "applicability": list(self.applicability),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArchiveManifest":
        certificate_raw = value.get("certificate")
        if not isinstance(certificate_raw, Mapping):
            raise ArchiveValidationError("manifest certificate must be an object")
        return cls(
            schema_version=int(value.get("schema_version") or 1),
            archive_id=str(value.get("archive_id") or ""),
            kind=ArchiveKind(str(value["kind"])),
            parent_source_sha256=str(value["parent_source_sha256"]),
            candidate_source_sha256=str(value["candidate_source_sha256"]),
            patch_path=str(value["patch_path"]),
            patch_sha256=str(value["patch_sha256"]),
            certificate=LocalCertificate.from_dict(certificate_raw),
            mechanism=str(value["mechanism"]),
            target_boundary=str(value["target_boundary"]),
            dependencies=tuple(str(item) for item in value.get("dependencies") or []),
            constituents=tuple(str(item) for item in value.get("constituents") or []),
            integration_residual_path=str(value.get("integration_residual_path") or ""),
            integration_residual_sha256=str(
                value.get("integration_residual_sha256") or ""
            ),
            trace_signature=tuple(
                str(item) for item in value.get("trace_signature") or []
            ),
            applicability=tuple(str(item) for item in value.get("applicability") or []),
        )


@dataclass(frozen=True)
class ComponentSelection:
    """One component application requested by the search agent."""

    component_id: str
    mode: ApplicationMode = ApplicationMode.EXACT
    semantic_instructions: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _require_sha256(self.component_id, "component_id"),
        )
        object.__setattr__(self, "mode", ApplicationMode(str(self.mode)))
        instructions = str(self.semantic_instructions).strip()
        if self.mode is ApplicationMode.SEMANTIC and not instructions:
            raise ArchiveValidationError(
                "semantic component selections require semantic_instructions"
            )
        object.__setattr__(self, "semantic_instructions", instructions)
        object.__setattr__(self, "rationale", str(self.rationale).strip())

    def attempt_dict(self) -> dict[str, str]:
        """Return fields that change the actual application attempt."""

        return {
            "component_id": self.component_id,
            "mode": self.mode.value,
            "semantic_instructions": self.semantic_instructions,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            **self.attempt_dict(),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComponentSelection":
        return cls(
            component_id=str(value["component_id"]),
            mode=ApplicationMode(str(value.get("mode") or ApplicationMode.EXACT)),
            semantic_instructions=str(value.get("semantic_instructions") or ""),
            rationale=str(value.get("rationale") or ""),
        )


@dataclass(frozen=True)
class CompositionPlan:
    """A deterministic, side-effect-free plan for one composition attempt."""

    base_source_sha256: str
    selections: tuple[ComponentSelection, ...] = ()
    integration_edit_path: str = ""
    integration_edit_sha256: str = ""
    integration_instructions: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_source_sha256",
            _require_sha256(self.base_source_sha256, "base_source_sha256"),
        )
        selections = tuple(self.selections)
        if any(not isinstance(item, ComponentSelection) for item in selections):
            raise ArchiveValidationError(
                "selections must contain ComponentSelection values"
            )
        component_ids = [item.component_id for item in selections]
        if len(component_ids) != len(set(component_ids)):
            raise ArchiveValidationError(
                "a composition plan cannot select the same component twice"
            )
        object.__setattr__(self, "selections", selections)

        edit_path, edit_hash = _optional_artifact(
            self.integration_edit_path,
            self.integration_edit_sha256,
            field_name="integration_edit",
        )
        instructions = str(self.integration_instructions).strip()
        if instructions and not edit_hash:
            # Instructions can describe an edit that has not been materialized yet.
            edit_path = ""
            edit_hash = ""
        object.__setattr__(self, "integration_edit_path", edit_path)
        object.__setattr__(self, "integration_edit_sha256", edit_hash)
        object.__setattr__(self, "integration_instructions", instructions)

        if int(self.schema_version) != 1:
            raise ArchiveValidationError("unsupported composition schema_version")
        object.__setattr__(self, "schema_version", 1)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(item.component_id for item in self.selections)

    @property
    def attempt_fingerprint(self) -> str:
        """Identify equivalent attempts without including explanatory rationale."""

        return _fingerprint(
            {
                "schema_version": self.schema_version,
                "base_source_sha256": self.base_source_sha256,
                "selections": [item.attempt_dict() for item in self.selections],
                "integration_edit_sha256": self.integration_edit_sha256,
                "integration_instructions": self.integration_instructions,
            }
        )

    def frozen_packet_refs(
        self, catalog: "ArchiveCatalog"
    ) -> tuple[FrozenPacketRef, ...]:
        return catalog.frozen_packet_refs(self.component_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_source_sha256": self.base_source_sha256,
            "selections": [item.to_dict() for item in self.selections],
            "integration_edit_path": self.integration_edit_path,
            "integration_edit_sha256": self.integration_edit_sha256,
            "integration_instructions": self.integration_instructions,
            "attempt_fingerprint": self.attempt_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompositionPlan":
        plan = cls(
            schema_version=int(value.get("schema_version") or 1),
            base_source_sha256=str(value["base_source_sha256"]),
            selections=tuple(
                ComponentSelection.from_dict(item)
                for item in value.get("selections") or []
                if isinstance(item, Mapping)
            ),
            integration_edit_path=str(value.get("integration_edit_path") or ""),
            integration_edit_sha256=str(value.get("integration_edit_sha256") or ""),
            integration_instructions=str(value.get("integration_instructions") or ""),
        )
        declared = str(value.get("attempt_fingerprint") or "")
        if declared and declared != plan.attempt_fingerprint:
            raise ArchiveValidationError(
                "composition attempt fingerprint does not match"
            )
        return plan


class ArchiveCatalog:
    """In-memory index of validated content-addressed archive manifests."""

    def __init__(self, manifests: Iterable[ArchiveManifest] = ()) -> None:
        self._manifests: dict[str, ArchiveManifest] = {}
        for manifest in manifests:
            self.register(manifest)

    @property
    def manifests(self) -> Mapping[str, ArchiveManifest]:
        return MappingProxyType(self._manifests)

    def __len__(self) -> int:
        return len(self._manifests)

    def __iter__(self) -> Iterator[ArchiveManifest]:
        for component_id in sorted(self._manifests):
            yield self._manifests[component_id]

    def __contains__(self, component_id: object) -> bool:
        return component_id in self._manifests

    def get(self, component_id: str) -> ArchiveManifest:
        try:
            return self._manifests[component_id]
        except KeyError as exc:
            raise UnknownComponentError(component_id) from exc

    def register(self, manifest: ArchiveManifest) -> str:
        """Register a manifest idempotently and return its content address."""

        if not isinstance(manifest, ArchiveManifest):
            raise TypeError("manifest must be an ArchiveManifest")
        existing = self._manifests.get(manifest.archive_id)
        if existing is not None:
            if existing.identity_dict() != manifest.identity_dict():
                raise ArchiveValidationError(
                    f"content-address collision for {manifest.archive_id}"
                )
            return manifest.archive_id
        self._manifests[manifest.archive_id] = manifest
        return manifest.archive_id

    @classmethod
    def load(cls, path: Path | str) -> "ArchiveCatalog":
        """Load one manifest, a catalog JSON document, or a manifest directory."""

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)

        documents: list[Mapping[str, Any]] = []
        if source.is_dir():
            manifest_paths = sorted(source.rglob("manifest.json"))
            for manifest_path in manifest_paths:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(raw, Mapping):
                    raise ArchiveValidationError(
                        f"manifest must contain an object: {manifest_path}"
                    )
                documents.append(raw)
        else:
            raw = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ArchiveValidationError("archive JSON must contain an object")
            nested = raw.get("manifests")
            if nested is None:
                documents.append(raw)
            else:
                if not isinstance(nested, list):
                    raise ArchiveValidationError("catalog manifests must be a list")
                for item in nested:
                    if not isinstance(item, Mapping):
                        raise ArchiveValidationError(
                            "catalog manifests must contain objects"
                        )
                    documents.append(item)

        catalog = cls(ArchiveManifest.from_dict(item) for item in documents)
        catalog.validate()
        return catalog

    def to_dict(self) -> dict[str, Any]:
        return {"manifests": [manifest.to_dict() for manifest in self]}

    def validate(self) -> None:
        """Reject missing references and cycles in dependency/provenance graphs."""

        self._reference_closure(tuple(self._manifests), include_constituents=True)

    def dependency_closure(
        self, component_ids: Iterable[str]
    ) -> tuple[ArchiveManifest, ...]:
        """Return dependencies before dependants in deterministic order."""

        ids = self._reference_closure(component_ids, include_constituents=False)
        return tuple(self._manifests[item] for item in ids)

    def reference_closure(
        self, component_ids: Iterable[str]
    ) -> tuple[ArchiveManifest, ...]:
        """Return dependency and constituent provenance closure."""

        ids = self._reference_closure(component_ids, include_constituents=True)
        return tuple(self._manifests[item] for item in ids)

    def _reference_closure(
        self,
        component_ids: Iterable[str],
        *,
        include_constituents: bool,
    ) -> tuple[str, ...]:
        roots = tuple(sorted(set(str(item) for item in component_ids)))
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(component_id: str, stack: tuple[str, ...]) -> None:
            status = state.get(component_id, 0)
            if status == 2:
                return
            if status == 1:
                try:
                    start = stack.index(component_id)
                    cycle = stack[start:] + (component_id,)
                except ValueError:
                    cycle = stack + (component_id,)
                raise DependencyCycleError(
                    "archive reference cycle: " + " -> ".join(cycle)
                )

            manifest = self._manifests.get(component_id)
            if manifest is None:
                raise UnknownComponentError(component_id)
            state[component_id] = 1
            references = (
                manifest.references if include_constituents else manifest.dependencies
            )
            for referenced_id in references:
                visit(referenced_id, stack + (component_id,))
            state[component_id] = 2
            order.append(component_id)

        for root in roots:
            visit(root, ())
        return tuple(order)

    def make_plan(
        self,
        *,
        base_source_sha256: str,
        selections: Sequence[ComponentSelection] = (),
        integration_edit_path: str = "",
        integration_edit_sha256: str = "",
        integration_instructions: str = "",
    ) -> CompositionPlan:
        """Expand dependencies, deduplicate selections, and topologically order them.

        Explicit selections control application mode.  Dependencies omitted by
        the caller are inserted as exact replays.  Conflicting duplicate
        selections are rejected instead of silently choosing a mode.
        """

        explicit: dict[str, ComponentSelection] = {}
        for selection in selections:
            if not isinstance(selection, ComponentSelection):
                raise ArchiveValidationError(
                    "selections must contain ComponentSelection values"
                )
            prior = explicit.get(selection.component_id)
            if prior is not None and prior != selection:
                raise ArchiveValidationError(
                    f"conflicting selections for {selection.component_id}"
                )
            explicit[selection.component_id] = selection

        closure = self.dependency_closure(explicit)
        ordered = tuple(
            explicit.get(
                manifest.archive_id,
                ComponentSelection(component_id=manifest.archive_id),
            )
            for manifest in closure
        )
        return CompositionPlan(
            base_source_sha256=base_source_sha256,
            selections=ordered,
            integration_edit_path=integration_edit_path,
            integration_edit_sha256=integration_edit_sha256,
            integration_instructions=integration_instructions,
        )

    def validate_plan(self, plan: CompositionPlan) -> None:
        """Ensure a plan is complete and ordered for this catalog."""

        if not isinstance(plan, CompositionPlan):
            raise TypeError("plan must be a CompositionPlan")
        positions = {
            item.component_id: index for index, item in enumerate(plan.selections)
        }
        for selection in plan.selections:
            manifest = self.get(selection.component_id)
            for dependency in manifest.dependencies:
                dependency_position = positions.get(dependency)
                if dependency_position is None:
                    raise ArchiveValidationError(
                        f"plan omits dependency {dependency} required by "
                        f"{selection.component_id}"
                    )
                if dependency_position >= positions[selection.component_id]:
                    raise ArchiveValidationError(
                        f"dependency {dependency} must precede {selection.component_id}"
                    )

    def frozen_packet_refs(
        self, component_ids: Iterable[str]
    ) -> tuple[FrozenPacketRef, ...]:
        """Return unique frozen packets that must be replayed for components.

        Composite constituents are included because their local certificates
        remain obligations even when the composite's complete patch is replayed
        directly.
        """

        manifests = self.reference_closure(component_ids)
        by_packet_id: dict[str, FrozenPacketRef] = {}
        by_content: dict[str, FrozenPacketRef] = {}
        ordered: list[FrozenPacketRef] = []
        for manifest in manifests:
            for ref in manifest.certificate.frozen_packet_refs:
                prior = by_packet_id.get(ref.packet_id)
                if prior is not None and prior.content_sha256 != ref.content_sha256:
                    raise ArchiveValidationError(
                        f"packet_id {ref.packet_id!r} has conflicting frozen contents"
                    )
                by_packet_id[ref.packet_id] = ref
                if ref.content_sha256 in by_content:
                    continue
                by_content[ref.content_sha256] = ref
                ordered.append(ref)
        return tuple(ordered)


__all__ = [
    "ApplicationMode",
    "ArchiveCatalog",
    "ArchiveKind",
    "ArchiveManifest",
    "ArchiveValidationError",
    "ComponentSelection",
    "CompositionPlan",
    "DependencyCycleError",
    "FrozenPacketRef",
    "LocalCertificate",
    "UnknownComponentError",
]
