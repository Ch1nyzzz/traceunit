"""Persistent capability battery: the unit-test axis of the optimization.

The battery is a run-long collection of capability groups; each group holds
several cross-domain probe instances of one atomic capability. An instance is
a frozen single-case packet bundle (``packet_kind: battery_instance``) reusing
the sandboxed deterministic/probe execution machinery. Scores are per-group
pass rates against a stored incumbent reference, so an edit is judged by
whether it moved a capability - not by whether it satisfied one reproduction
of one failing task.

The calibration ledger records, per candidate, the per-capability battery
deltas next to the paired search delta. It is a mechanical, host-written
record: it informs the Test Author's attention (which capabilities' batteries
predict search, which instances carry no information) and never gates a
decision by itself.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from traceunit.io import append_jsonl, read_json, read_jsonl, write_json
from traceunit.models import TestExecution, UnitFamily
from traceunit.tests_runtime import (
    load_test_packet,
    run_test_cases,
    verify_frozen_packet,
)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class BatteryError(RuntimeError):
    pass


@dataclass(frozen=True)
class BatteryInstance:
    """One frozen probe of one atomic capability."""

    instance_id: str
    capability: str
    family: UnitFamily
    description: str
    expected_incumbent_pass: bool
    content_sha256: str
    status: str = "active"
    created_iteration: int = 0
    retired_iteration: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["family"] = self.family.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatteryInstance":
        return cls(
            instance_id=str(value["instance_id"]),
            capability=str(value["capability"]),
            family=UnitFamily(str(value["family"])),
            description=str(value.get("description") or ""),
            expected_incumbent_pass=bool(value.get("expected_incumbent_pass", False)),
            content_sha256=str(value.get("content_sha256") or ""),
            status=str(value.get("status") or "active"),
            created_iteration=int(value.get("created_iteration") or 0),
            retired_iteration=(
                None
                if value.get("retired_iteration") is None
                else int(value["retired_iteration"])
            ),
        )


@dataclass(frozen=True)
class BatteryResult:
    """One instance executed against one candidate source."""

    instance_id: str
    capability: str
    passed: bool
    timed_out: bool = False
    error: str = ""
    duration_s: float = 0.0
    model_calls: int = 0
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_slug(value: str, field_name: str) -> str:
    slug = str(value).strip()
    if not _SLUG_RE.fullmatch(slug):
        raise BatteryError(
            f"{field_name} must be a short kebab/snake-case slug, got {value!r}"
        )
    return slug


class Battery:
    """Manifest plus frozen instance bundles under <run_dir>/battery/."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.instances_root = root / "instances"
        self.reference_path = root / "incumbent_results.json"
        self.capability_notes_path = root / "capabilities.json"

    def load(self) -> tuple[BatteryInstance, ...]:
        if not self.manifest_path.is_file():
            return ()
        raw = read_json(self.manifest_path)
        return tuple(
            BatteryInstance.from_dict(item) for item in raw.get("instances") or []
        )

    def save(self, instances: Iterable[BatteryInstance]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(
            self.manifest_path,
            {"instances": [item.to_dict() for item in instances]},
        )

    def active(self, capability: str | None = None) -> tuple[BatteryInstance, ...]:
        return tuple(
            item
            for item in self.load()
            if item.status == "active"
            and (capability is None or item.capability == capability)
        )

    def capabilities(self) -> dict[str, tuple[BatteryInstance, ...]]:
        groups: dict[str, list[BatteryInstance]] = {}
        for item in self.active():
            groups.setdefault(item.capability, []).append(item)
        return {key: tuple(value) for key, value in groups.items()}

    def bundle_dir(self, instance_id: str) -> Path:
        return self.instances_root / instance_id

    def add(self, instance: BatteryInstance, source_bundle: Path) -> None:
        """Copy a frozen instance bundle into the battery and record it."""

        existing = {item.instance_id: item for item in self.load()}
        if instance.instance_id in existing:
            raise BatteryError(f"duplicate instance_id: {instance.instance_id}")
        destination = self.bundle_dir(instance.instance_id)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_bundle, destination)
        packet = load_test_packet(destination)
        if (
            not verify_frozen_packet(destination, packet)
            or packet.content_sha256 != instance.content_sha256
        ):
            shutil.rmtree(destination, ignore_errors=True)
            raise BatteryError(
                f"instance bundle failed integrity check: {instance.instance_id}"
            )
        self.save([*existing.values(), instance])

    def retire(self, instance_ids: Iterable[str], *, iteration: int) -> list[str]:
        instances = list(self.load())
        by_id = {item.instance_id: index for index, item in enumerate(instances)}
        retired: list[str] = []
        for instance_id in instance_ids:
            index = by_id.get(str(instance_id))
            if index is None:
                raise BatteryError(f"unknown instance_id to retire: {instance_id}")
            if instances[index].status != "active":
                continue
            instances[index] = replace(
                instances[index], status="retired", retired_iteration=iteration
            )
            retired.append(str(instance_id))
        if retired:
            self.save(instances)
        return retired

    # -- incumbent reference -------------------------------------------------

    def load_reference(self) -> dict[str, bool]:
        if not self.reference_path.is_file():
            return {}
        return {
            str(key): bool(value)
            for key, value in read_json(self.reference_path).items()
        }

    def save_reference(self, reference: Mapping[str, bool]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.reference_path, dict(reference))

    def update_reference(self, updates: Mapping[str, bool]) -> None:
        reference = self.load_reference()
        reference.update({str(key): bool(value) for key, value in updates.items()})
        self.save_reference(reference)

    # -- capability descriptions ----------------------------------------------

    def load_capability_notes(self) -> dict[str, str]:
        if not self.capability_notes_path.is_file():
            return {}
        return {
            str(key): str(value)
            for key, value in read_json(self.capability_notes_path).items()
        }

    def update_capability_notes(self, notes: Mapping[str, str]) -> None:
        merged = self.load_capability_notes()
        merged.update({str(key): str(value) for key, value in notes.items()})
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.capability_notes_path, merged)

    # -- summaries -------------------------------------------------------------

    def state_summary(self, reference: Mapping[str, bool] | None = None) -> dict:
        """Public battery state staged to agents: groups, instances, scores."""

        reference = self.load_reference() if reference is None else dict(reference)
        notes = self.load_capability_notes()
        groups = []
        for capability, instances in sorted(self.capabilities().items()):
            groups.append(
                {
                    "capability": capability,
                    "description": notes.get(capability, ""),
                    "family": instances[0].family.value,
                    "instances": [
                        {
                            "instance_id": item.instance_id,
                            "description": item.description,
                            "incumbent_passed": reference.get(item.instance_id),
                        }
                        for item in instances
                    ],
                    "incumbent_pass_rate": _rate(
                        [reference.get(item.instance_id) for item in instances]
                    ),
                }
            )
        return {"capabilities": groups}


class BatteryRunner:
    """Execute active instances against one source via the frozen sandbox."""

    def __init__(
        self,
        *,
        battery: Battery,
        python: Path | None = None,
        probe_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.battery = battery
        self.python = python
        self.probe_runner = probe_runner

    def run(
        self,
        *,
        source: Path,
        subject: str,
        output_dir: Path,
        capability: str | None = None,
    ) -> tuple[BatteryResult, ...]:
        results: list[BatteryResult] = []
        for instance in self.battery.active(capability):
            bundle = self.battery.bundle_dir(instance.instance_id)
            packet = load_test_packet(bundle)
            if not verify_frozen_packet(bundle, packet):
                raise BatteryError(
                    f"battery instance was modified: {instance.instance_id}"
                )
            executions = run_test_cases(
                packet=packet,
                bundle=bundle,
                source=source,
                subject=subject,
                output_dir=output_dir / instance.instance_id,
                python=self.python,
                probe_runner=self.probe_runner,
            )
            execution: TestExecution = executions[0]
            results.append(
                BatteryResult(
                    instance_id=instance.instance_id,
                    capability=instance.capability,
                    passed=bool(execution.passed),
                    timed_out=execution.timed_out,
                    error=execution.error,
                    duration_s=execution.duration_s,
                    model_calls=execution.model_calls,
                    tokens=execution.tokens,
                )
            )
        return tuple(results)


# -- scoring ---------------------------------------------------------------


def _rate(values: Iterable[bool | None]) -> float | None:
    known = [bool(value) for value in values if value is not None]
    if not known:
        return None
    return sum(known) / len(known)


def capability_scores(
    results: Iterable[BatteryResult],
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[BatteryResult]] = {}
    for result in results:
        groups.setdefault(result.capability, []).append(result)
    return {
        capability: {
            "passed": sum(1 for item in items if item.passed),
            "total": len(items),
            "rate": sum(1 for item in items if item.passed) / len(items),
        }
        for capability, items in groups.items()
    }


def battery_deltas(
    *,
    instances: Iterable[BatteryInstance],
    reference: Mapping[str, bool],
    results: Iterable[BatteryResult],
) -> dict[str, dict[str, float | int]]:
    """Per-capability paired comparison of candidate results vs the reference.

    Only instances with both a reference value and a candidate result are
    paired; the delta is the pass-rate difference over those instances.
    """

    by_result = {item.instance_id: item for item in results}
    groups: dict[str, list[tuple[bool, bool]]] = {}
    for instance in instances:
        if instance.status != "active":
            continue
        result = by_result.get(instance.instance_id)
        if result is None or instance.instance_id not in reference:
            continue
        groups.setdefault(instance.capability, []).append(
            (bool(reference[instance.instance_id]), result.passed)
        )
    deltas: dict[str, dict[str, float | int]] = {}
    for capability, pairs in groups.items():
        incumbent_passed = sum(1 for before, _ in pairs if before)
        candidate_passed = sum(1 for _, after in pairs if after)
        deltas[capability] = {
            "paired": len(pairs),
            "incumbent_passed": incumbent_passed,
            "candidate_passed": candidate_passed,
            "delta": (candidate_passed - incumbent_passed) / len(pairs),
        }
    return deltas


# -- calibration -------------------------------------------------------------


class CalibrationLedger:
    """Mechanical (battery delta, search delta) pairs, one row per candidate.

    Direction agreement per capability and per-instance variance are the two
    statistics our data volume actually supports; nothing here fits weights
    or gates decisions.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self,
        *,
        iteration: int,
        candidate_id: str,
        target_capability: str,
        deltas: Mapping[str, Mapping[str, float | int]],
        instance_results: Mapping[str, bool],
        search_delta: float | None,
        decision: str,
    ) -> None:
        append_jsonl(
            self.path,
            {
                "iteration": iteration,
                "candidate_id": candidate_id,
                "target_capability": target_capability,
                "deltas": {key: dict(value) for key, value in deltas.items()},
                "instance_results": dict(instance_results),
                "search_delta": search_delta,
                "decision": decision,
            },
        )

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in read_jsonl(self.path)]

    def summary(self) -> dict[str, Any]:
        rows = [row for row in self.rows() if row.get("search_delta") is not None]
        capabilities: dict[str, dict[str, int]] = {}
        instance_values: dict[str, list[bool]] = {}
        for row in rows:
            search_up = float(row["search_delta"]) > 0
            for capability, delta in (row.get("deltas") or {}).items():
                stats = capabilities.setdefault(
                    capability,
                    {
                        "pairs": 0,
                        "battery_up": 0,
                        "battery_up_search_up": 0,
                        "battery_up_search_flat_or_down": 0,
                        "targeted": 0,
                    },
                )
                stats["pairs"] += 1
                if capability == row.get("target_capability"):
                    stats["targeted"] += 1
                if float(delta.get("delta") or 0.0) > 0:
                    stats["battery_up"] += 1
                    if search_up:
                        stats["battery_up_search_up"] += 1
                    else:
                        stats["battery_up_search_flat_or_down"] += 1
            for instance_id, passed in (row.get("instance_results") or {}).items():
                instance_values.setdefault(instance_id, []).append(bool(passed))
        uninformative = sorted(
            instance_id
            for instance_id, values in instance_values.items()
            if len(values) >= 5 and len(set(values)) == 1
        )
        return {
            "rows": len(rows),
            "capabilities": capabilities,
            "uninformative_instances": uninformative,
        }

    def markdown(self) -> str:
        summary = self.summary()
        lines = [
            "# Battery calibration (host-computed)",
            "",
            f"Candidates with paired search evidence: {summary['rows']}.",
            "",
            "| capability | targeted | battery-up | ...and search up | ...and search flat/down |",
            "| --- | --- | --- | --- | --- |",
        ]
        for capability, stats in sorted(summary["capabilities"].items()):
            lines.append(
                f"| {capability} | {stats['targeted']} | {stats['battery_up']} "
                f"| {stats['battery_up_search_up']} "
                f"| {stats['battery_up_search_flat_or_down']} |"
            )
        if summary["uninformative_instances"]:
            lines.extend(
                [
                    "",
                    "Instances constant across every candidate (carrying no "
                    "information; retire or replace):",
                ]
            )
            lines.extend(f"- {item}" for item in summary["uninformative_instances"])
        return "\n".join(lines) + "\n"
