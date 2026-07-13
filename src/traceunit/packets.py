"""The Test Author as battery maintainer.

Each iteration the author reads the incumbent's failing search traces (and, in
C3, the world model, the previous iteration's raw outcome, any mismatch
evidence, and the host-computed calibration table), diagnoses the root-cause
atomic capability, and updates the persistent battery: new cross-domain
instances for the target capability, optional retirements of instances the
calibration flags as uninformative. On a cold start (empty battery) the same
call builds the initial battery from the baseline's failure clusters.

Every new instance is admitted against the incumbent: the author declares
whether the incumbent passes it, the host measures, and a mismatch sends the
whole update back with concrete feedback. Admitted instances are frozen
(content-hashed) before joining the battery.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from traceunit.agents.prompts import battery_author_prompt
from traceunit.agents.runner import WorkspaceAgent
from traceunit.candidate import stage_archive_records
from traceunit.battery import (
    Battery,
    BatteryError,
    BatteryInstance,
    CalibrationLedger,
    validate_slug,
)
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.io import copy_source, read_json, safe_relative_path, write_json
from traceunit.models import RunState, TestExecutionMode, UnitFamily
from traceunit.store import RunStore
from traceunit.trace_evidence import (
    TraceEvidenceError,
    stage_search_trace_evidence,
)
from traceunit.tests_runtime import (
    InvalidTestPacket,
    freeze_test_packet,
    load_test_packet,
    run_test_cases,
)
from traceunit.ut_memory import WorldModel


class TestDesignFailure(RuntimeError):
    pass


class BatteryAuthor:
    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        agent: WorkspaceAgent,
        battery: Battery,
        calibration: CalibrationLedger,
        world_model: WorldModel | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.agent = agent
        self.battery = battery
        self.calibration = calibration
        self.world_model = world_model

    def get_or_update(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
    ) -> tuple[str, UnitFamily]:
        """Apply one battery update per iteration; the ref file is resume-only."""

        update_ref = iteration_dir / "battery_update_ref.json"
        if update_ref.is_file():
            ref = read_json(update_ref)
            return str(ref["target_capability"]), UnitFamily(str(ref["target_family"]))

        target_capability, target_family = self._author(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
        )
        write_json(
            update_ref,
            {
                "target_capability": target_capability,
                "target_family": target_family.value,
            },
        )
        return target_capability, target_family

    def _stage_memory_inputs(self, workspace: Path) -> dict[str, Path | None]:
        """Stage the world model, the previous-iteration digest, and any
        mismatch evidence into the author's workspace."""

        staged: dict[str, Path | None] = {
            "world_model": None,
            "last_iteration": None,
            "mismatch": None,
        }
        if self.world_model is None:
            return staged
        staged["world_model"] = self.world_model.stage_into(workspace)
        last_iteration = self.store.memory_root / "last_iteration.json"
        if last_iteration.is_file():
            target = workspace / "last_iteration.json"
            if not target.exists():
                shutil.copy2(last_iteration, target)
            staged["last_iteration"] = target
            info = read_json(last_iteration)
            mismatch_path = info.get("mismatch_path")
            if mismatch_path and Path(str(mismatch_path)).is_dir():
                mismatch_dir = workspace / "mismatch_evidence"
                if not mismatch_dir.exists():
                    shutil.copytree(Path(str(mismatch_path)), mismatch_dir)
                    try:
                        stage_search_trace_evidence(
                            store=self.store,
                            candidate_id=str(info.get("candidate_id") or ""),
                            destination=mismatch_dir / "candidate_traces",
                            max_failure_traces=self.config.loop.max_failure_traces,
                        )
                    except (TraceEvidenceError, FileNotFoundError):
                        pass
                staged["mismatch"] = mismatch_dir
        return staged

    def _stage_previous_attempt(
        self, *, iteration_dir: Path, attempt: int, workspace: Path
    ) -> None:
        """Copy the rejected attempt's own output and the host's admission
        transcripts into the retry workspace.

        Without them the retrying author rebuilds blind: it cannot read its
        own rejected update, and an admission failure ('declared pass=True,
        measured False') is undiagnosable without the incumbent's actual
        reply against each expectation.
        """

        previous = iteration_dir / "test_author" / f"attempt_{attempt - 1}"
        workspace.mkdir(parents=True, exist_ok=True)
        for source_name, target_name in (
            ("workspace/output", "previous_output"),
            ("admission", "previous_admission"),
        ):
            source = previous / source_name
            target = workspace / target_name
            if source.is_dir() and not target.exists():
                shutil.copytree(source, target)

    def _stage_battery_inputs(self, workspace: Path) -> dict[str, Path | None]:
        """Stage the battery state, the calibration table, and the frozen
        instance bundles for the author.

        The author owns the battery, so it gets the probe files themselves -
        without them a predecessor's unfair expectation (an invented exact
        format string, a starved token budget) is indistinguishable from a
        genuine capability gap, and the mismatch channel cannot self-correct.
        """

        workspace.mkdir(parents=True, exist_ok=True)
        state_path = workspace / "battery_state.json"
        if not state_path.exists():
            write_json(state_path, self.battery.state_summary())
        calibration_path: Path | None = None
        if self.calibration.path.is_file():
            calibration_path = workspace / "battery_calibration.md"
            if not calibration_path.exists():
                calibration_path.write_text(
                    self.calibration.markdown(), encoding="utf-8"
                )
        instances_path: Path | None = None
        if self.battery.instances_root.is_dir():
            instances_path = workspace / "battery_instances"
            if not instances_path.exists():
                shutil.copytree(self.battery.instances_root, instances_path)
        return {
            "battery_state": state_path,
            "calibration": calibration_path,
            "instances": instances_path,
        }

    def _commit_world_model(self, workspace: Path, *, iteration: int) -> None:
        """Copy the author's world-model file back; record whether it grew."""

        if self.world_model is None:
            return
        if self.world_model.commit_from(workspace):
            self.store.append_event(
                "world_model_updated",
                iteration=iteration,
                distills=self.world_model.distill_count,
            )
        elif (self.store.memory_root / "last_iteration.json").is_file():
            self.store.append_event(
                "world_model_not_updated",
                iteration=iteration,
            )

    def _author(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
    ) -> tuple[str, UnitFamily]:
        feedback = ""
        for attempt in range(1, self.config.loop.max_attempts_per_packet + 1):
            workspace = (
                iteration_dir / "test_author" / f"attempt_{attempt}" / "workspace"
            )
            if attempt > 1:
                self._stage_previous_attempt(
                    iteration_dir=iteration_dir,
                    attempt=attempt,
                    workspace=workspace,
                )
            output = workspace / "output"
            incumbent_copy = workspace / "incumbent_source"
            if not incumbent_copy.exists():
                copy_source(Path(state.incumbent_source), incumbent_copy)
            trace_manifest = workspace / "trace_evidence" / "manifest.json"
            if not trace_manifest.exists():
                stage_search_trace_evidence(
                    store=self.store,
                    candidate_id=state.incumbent_id,
                    destination=workspace / "trace_evidence",
                    max_failure_traces=self.config.loop.max_failure_traces,
                )
            memory_inputs = self._stage_memory_inputs(workspace)
            battery_inputs = self._stage_battery_inputs(workspace)
            archives_path: Path | None = None
            if not (workspace / "archives.json").is_file():
                archives = stage_archive_records(state, workspace)
                if archives:
                    archives_path = workspace / "archives.json"
                    write_json(archives_path, {"archives": archives})
            else:
                archives_path = workspace / "archives.json"
            prompt = battery_author_prompt(
                benchmark_context=self.benchmark.context(),
                trace_manifest=trace_manifest,
                incumbent_source=incumbent_copy,
                battery_state_path=battery_inputs["battery_state"],
                calibration_path=battery_inputs["calibration"],
                battery_instances_path=battery_inputs["instances"],
                archives_path=archives_path,
                cold_start=not self.battery.active(),
                max_instances_per_capability=(
                    self.config.loop.max_instances_per_capability
                ),
                world_model_path=memory_inputs["world_model"],
                last_iteration_path=memory_inputs["last_iteration"],
                mismatch_path=memory_inputs["mismatch"],
                iteration=iteration,
                probes_supported=self.benchmark.supports_agent_probe,
                target_api_env=self.config.benchmark.api_key_env,
                output_dir=output,
            )
            if feedback:
                prompt += (
                    "\n\nThe previous battery update was rejected. Fix the causes "
                    "instead of weakening the instances:\n" + feedback
                )
                if (workspace / "previous_output").is_dir():
                    prompt += (
                        "\n\nYour previous attempt's files are under "
                        "previous_output/ - edit and resubmit rather than "
                        "rebuilding from scratch."
                    )
                if (workspace / "previous_admission").is_dir():
                    prompt += (
                        " The host's admission runs are under "
                        "previous_admission/<instance_id>/ - each shows what "
                        "the incumbent actually replied and which expectation "
                        "or budget failed."
                    )
            if not (output / "battery_update.json").is_file():
                run = self.agent.run(
                    role="test_author",
                    prompt=prompt,
                    workspace=workspace,
                    log_dir=iteration_dir
                    / "test_author"
                    / f"attempt_{attempt}"
                    / "agent",
                )
                self._commit_world_model(workspace, iteration=iteration)
                if run.returncode != 0 or run.timed_out:
                    feedback = (
                        f"agent failed: returncode={run.returncode}, "
                        f"timed_out={run.timed_out}"
                    )
                    continue
            try:
                target_capability, target_family = self._apply_update(
                    output=output,
                    state=state,
                    iteration=iteration,
                    iteration_dir=iteration_dir,
                    attempt=attempt,
                )
            except (BatteryError, InvalidTestPacket, TraceEvidenceError) as exc:
                feedback = str(exc)
                # The update file must be re-authored, not re-parsed.
                stale = output / "battery_update.json"
                if stale.is_file():
                    stale.rename(output / f"battery_update.rejected_{attempt}.json")
                continue
            return target_capability, target_family
        raise TestDesignFailure(
            feedback or "Test Author produced no admissible battery update"
        )

    def _apply_update(
        self,
        *,
        output: Path,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        attempt: int,
    ) -> tuple[str, UnitFamily]:
        raw = read_json(output / "battery_update.json")
        target_capability = validate_slug(
            str(raw.get("target_capability") or ""), "target_capability"
        )
        try:
            target_family = UnitFamily(str(raw.get("target_family") or ""))
        except ValueError as exc:
            raise BatteryError(
                "target_family must be one of the frozen L0 families"
            ) from exc

        existing = {item.instance_id: item for item in self.battery.load()}
        active_counts: dict[str, int] = {}
        for item in existing.values():
            if item.status == "active":
                active_counts[item.capability] = (
                    active_counts.get(item.capability, 0) + 1
                )

        retire_ids = [str(item) for item in raw.get("retire_instance_ids") or []]
        for instance_id in retire_ids:
            instance = existing.get(instance_id)
            if instance is None:
                raise BatteryError(f"unknown instance_id to retire: {instance_id}")
            if instance.status == "active":
                active_counts[instance.capability] -= 1

        # Validate and admit every new instance before committing anything.
        admitted: list[tuple[BatteryInstance, Path]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw.get("new_instances") or []):
            instance_id = validate_slug(
                str(item.get("instance_id") or ""), f"new_instances[{index}].instance_id"
            )
            capability = validate_slug(
                str(item.get("capability") or ""), f"{instance_id}.capability"
            )
            if instance_id in existing or instance_id in seen_ids:
                raise BatteryError(f"duplicate instance_id: {instance_id}")
            seen_ids.add(instance_id)
            try:
                family = UnitFamily(str(item.get("family") or ""))
            except ValueError as exc:
                raise BatteryError(
                    f"{instance_id}: family must be a frozen L0 family"
                ) from exc
            bundle = safe_relative_path(
                output, str(item.get("bundle") or f"instances/{instance_id}")
            )
            packet = load_test_packet(bundle)
            if packet.metadata.get("packet_kind") != "battery_instance":
                raise BatteryError(
                    f"{instance_id}: bundle must declare packet_kind=battery_instance"
                )
            if packet.primary_family is not family:
                raise BatteryError(
                    f"{instance_id}: packet primary_family does not match the "
                    "declared family"
                )
            if (
                not self.benchmark.supports_agent_probe
                and packet.cases[0].execution_mode
                is TestExecutionMode.MODEL_BACKED_PROBE
            ):
                raise BatteryError(
                    f"{instance_id}: this benchmark does not support "
                    "model-backed probes"
                )
            expected = bool(item.get("expected_incumbent_pass", False))
            executions = run_test_cases(
                packet=packet,
                bundle=bundle,
                source=Path(state.incumbent_source),
                subject="incumbent",
                output_dir=iteration_dir
                / "test_author"
                / f"attempt_{attempt}"
                / "admission"
                / instance_id,
                python=self.config.benchmark.unit_python,
                probe_runner=self.benchmark.run_agent_probe,
            )
            probe_case = packet.cases[0]
            if probe_case.execution_mode is TestExecutionMode.MODEL_BACKED_PROBE:
                used_tokens = int(executions[0].tokens or 0)
                if used_tokens * 2 > probe_case.max_tokens:
                    raise BatteryError(
                        f"{instance_id}: the incumbent used {used_tokens} of the "
                        f"{probe_case.max_tokens}-token budget; a budget below "
                        "2x the incumbent's usage judges candidates on "
                        "verbosity, not behavior - raise max_tokens"
                    )
            measured = bool(executions[0].passed)
            if measured != expected:
                raise BatteryError(
                    f"{instance_id}: declared expected_incumbent_pass={expected} "
                    f"but the incumbent measured passed={measured}; fix the "
                    "declaration or the instance"
                )
            frozen = freeze_test_packet(bundle, packet, admission_passed=True)
            instance = BatteryInstance(
                instance_id=instance_id,
                capability=capability,
                family=family,
                description=str(item.get("description") or ""),
                expected_incumbent_pass=expected,
                content_sha256=frozen.content_sha256,
                created_iteration=iteration,
            )
            active_counts[capability] = active_counts.get(capability, 0) + 1
            admitted.append((instance, bundle))

        cap = self.config.loop.max_instances_per_capability
        oversized = {
            capability: count
            for capability, count in active_counts.items()
            if count > cap
        }
        if oversized:
            raise BatteryError(
                "capability groups exceed the per-capability cap "
                f"({cap}): {sorted(oversized)} - retire instances first"
            )

        # The target group must leave something to improve.
        reference = dict(self.battery.load_reference())
        for instance, _ in admitted:
            reference[instance.instance_id] = instance.expected_incumbent_pass
        target_active_failing = 0
        for item in existing.values():
            if (
                item.status == "active"
                and item.instance_id not in retire_ids
                and item.capability == target_capability
                and reference.get(item.instance_id) is False
            ):
                target_active_failing += 1
        for instance, _ in admitted:
            if (
                instance.capability == target_capability
                and not instance.expected_incumbent_pass
            ):
                target_active_failing += 1
        if target_active_failing == 0:
            raise BatteryError(
                f"target capability {target_capability!r} has no active "
                "incumbent-failing instance: there is nothing for a candidate "
                "to improve"
            )

        # Commit: retirements, new instances, notes, reference updates.
        if retire_ids:
            retired = self.battery.retire(retire_ids, iteration=iteration)
            if retired:
                self.store.append_event(
                    "battery_instances_retired",
                    iteration=iteration,
                    instance_ids=retired,
                )
        already_present = {item.instance_id for item in self.battery.load()}
        for instance, bundle in admitted:
            if instance.instance_id not in already_present:
                self.battery.add(instance, bundle)
            self.battery.update_reference(
                {instance.instance_id: instance.expected_incumbent_pass}
            )
        notes = {
            validate_slug(str(key), "capability"): str(value)
            for key, value in dict(raw.get("capability_descriptions") or {}).items()
        }
        if notes:
            self.battery.update_capability_notes(notes)
        if admitted:
            self.store.append_event(
                "battery_instances_admitted",
                iteration=iteration,
                target_capability=target_capability,
                instance_ids=[instance.instance_id for instance, _ in admitted],
            )
        if not self.battery.active(target_capability):
            raise BatteryError(
                f"target capability {target_capability!r} has no active instances"
            )
        return target_capability, target_family


def author_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """A compact view of a battery update for logs and records."""

    return {
        "target_capability": raw.get("target_capability"),
        "new_instances": [
            item.get("instance_id") for item in raw.get("new_instances") or []
        ],
        "retire_instance_ids": list(raw.get("retire_instance_ids") or []),
    }
