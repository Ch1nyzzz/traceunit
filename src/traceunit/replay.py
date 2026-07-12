"""Frozen-packet references and behavioral replay.

A frozen TestPacket is the unit of certified behavior. Replay runs a stored
packet against a candidate source and checks the packet's declared candidate
contract. Only ``preserved`` packets (from promoted candidates) are replayed;
they gate every later candidate.

Integrity failures (missing or modified packet bundles) raise ``ReplayError``
because they indicate a corrupted store, not a property of the candidate.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from traceunit.io import safe_relative_path
from traceunit.tests_runtime import (
    candidate_contract,
    load_test_packet,
    run_test_cases,
    verify_frozen_packet,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReplayError(RuntimeError):
    """Raised when a frozen packet cannot be trusted or located."""


def _require_text(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _require_sha256(value: str, field_name: str) -> str:
    result = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest"
        )
    return result


def _require_relative_path(value: str, field_name: str) -> str:
    raw = _require_text(value, field_name).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{field_name} must be a portable path relative to the packet store"
        )
    return path.as_posix()


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
class PacketReplayResult:
    """Outcome of replaying one frozen packet against one candidate source."""

    packet_id: str
    content_sha256: str
    primary_family: str
    contract_passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PacketReplayer:
    def __init__(
        self,
        *,
        packet_root: Path,
        python: Path | None = None,
        probe_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.packet_root = packet_root.resolve()
        self.python = python
        self.probe_runner = probe_runner

    def replay(
        self,
        *,
        refs: Iterable[FrozenPacketRef],
        candidate_source: Path,
        output_dir: Path,
    ) -> tuple[PacketReplayResult, ...]:
        results: list[PacketReplayResult] = []
        seen: set[str] = set()
        for ref in refs:
            if ref.content_sha256 in seen:
                continue
            seen.add(ref.content_sha256)
            bundle = safe_relative_path(self.packet_root, ref.path)
            if not bundle.is_dir():
                raise ReplayError(f"frozen packet bundle is missing: {bundle}")
            packet = load_test_packet(bundle)
            if packet.content_sha256 != ref.content_sha256:
                raise ReplayError(
                    f"{ref.packet_id}: declared packet hash does not match the bundle"
                )
            if not verify_frozen_packet(bundle, packet):
                raise ReplayError(f"{ref.packet_id}: frozen packet was modified")
            executions = run_test_cases(
                packet=packet,
                bundle=bundle,
                source=candidate_source,
                subject="candidate",
                output_dir=output_dir / ref.content_sha256[:16],
                python=self.python,
                probe_runner=self.probe_runner,
            )
            passed, reasons = candidate_contract(packet, executions)
            results.append(
                PacketReplayResult(
                    packet_id=ref.packet_id,
                    content_sha256=ref.content_sha256,
                    primary_family=(
                        packet.primary_family.value
                        if packet.primary_family is not None
                        else ""
                    ),
                    contract_passed=passed,
                    reasons=tuple(f"{ref.packet_id}: {reason}" for reason in reasons),
                )
            )
        return tuple(results)


def copy_packet_into_store(
    *,
    packet_root: Path,
    packet_bundle: Path,
    content_sha256: str,
) -> str:
    """Copy a frozen packet into the immutable store and return a portable path."""

    relative = Path("packets") / content_sha256
    destination = packet_root / relative
    if destination.exists():
        packet = load_test_packet(destination)
        if not verify_frozen_packet(destination, packet):
            raise ReplayError(f"existing stored packet is corrupt: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(packet_bundle, destination)
    return relative.as_posix()
