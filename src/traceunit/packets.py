from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from traceunit.agents.prompts import test_author_prompt
from traceunit.agents.runner import WorkspaceAgent
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.io import copy_source, read_json, write_json
from traceunit.models import RunState, TestPacket, TestStatus
from traceunit.store import RunStore
from traceunit.trace_evidence import stage_search_trace_evidence
from traceunit.tests_runtime import (
    InvalidTestPacket,
    admission_contract,
    freeze_test_packet,
    load_test_packet,
    run_test_cases,
    verify_frozen_packet,
)


class TestDesignFailure(RuntimeError):
    pass


class PacketAuthor:
    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        agent: WorkspaceAgent,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.agent = agent

    def get_or_author(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        ut_memory_path: Path | None,
        pending_reflection: Path | None = None,
    ) -> tuple[TestPacket, Path, bool]:
        packet_ref = iteration_dir / "packet_ref.json"
        if packet_ref.is_file():
            ref = read_json(packet_ref)
            path = Path(str(ref["path"]))
            return self.verified(path), path, bool(ref.get("reused"))
        if state.active_packet_path:
            path = Path(state.active_packet_path)
            try:
                packet = self.verified(path)
            except TestDesignFailure:
                packet = None
            if packet is not None and packet.status == TestStatus.ADMITTED:
                write_json(
                    packet_ref,
                    {"path": str(path), "packet_id": packet.packet_id, "reused": True},
                )
                return packet, path, True
            self.retire_active(state)

        packet, path = self._author(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            ut_memory_path=ut_memory_path,
            pending_reflection=pending_reflection,
        )
        write_json(
            packet_ref,
            {"path": str(path), "packet_id": packet.packet_id, "reused": False},
        )
        state.active_packet_id = packet.packet_id
        state.active_packet_path = str(path)
        state.active_packet_uses = 0
        self.store.save_state(state)
        return packet, path, False

    @staticmethod
    def retire_active(state: RunState) -> None:
        state.active_packet_id = ""
        state.active_packet_path = ""
        state.active_packet_uses = 0

    def verified(self, path: Path) -> TestPacket:
        packet = load_test_packet(path)
        if not verify_frozen_packet(path, packet):
            raise TestDesignFailure(f"frozen TestPacket hash mismatch: {path}")
        return packet

    @staticmethod
    def latest_reflection(iteration_dir: Path) -> Mapping[str, Any] | None:
        """Return the newest attempt's reflection.json, if the author wrote one."""

        paths = sorted(
            (iteration_dir / "test_author").glob("attempt_*/workspace/reflection.json")
        )
        for path in reversed(paths):
            try:
                value = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(value, Mapping):
                return value
        return None

    def _author(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        ut_memory_path: Path | None,
        pending_reflection: Path | None,
    ) -> tuple[TestPacket, Path]:
        feedback = ""
        for attempt in range(1, self.config.loop.max_attempts_per_packet + 1):
            workspace = (
                iteration_dir / "test_author" / f"attempt_{attempt}" / "workspace"
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
            memory_copy = workspace / "ut_design_world_model.md"
            if ut_memory_path is not None and not memory_copy.exists():
                shutil.copy2(ut_memory_path, memory_copy)
            outcome_copy = workspace / "previous_outcome.json"
            if pending_reflection is not None and not outcome_copy.exists():
                shutil.copy2(pending_reflection, outcome_copy)
            prompt = test_author_prompt(
                benchmark_context=self.benchmark.context(),
                trace_manifest=trace_manifest,
                incumbent_source=incumbent_copy,
                ut_memory_path=(memory_copy if ut_memory_path is not None else None),
                previous_outcome_path=(
                    outcome_copy if pending_reflection is not None else None
                ),
                reflection_output_path=(
                    workspace / "reflection.json"
                    if pending_reflection is not None
                    else None
                ),
                output_dir=output,
            )
            if feedback:
                prompt += (
                    "\n\nThe previous packet failed mechanical admission. "
                    "Create a new packet rather than weakening expectations:\n"
                    + feedback
                )
            if not (output / "test_packet.json").is_file():
                run = self.agent.run(
                    role="test_author",
                    prompt=prompt,
                    workspace=workspace,
                    log_dir=iteration_dir
                    / "test_author"
                    / f"attempt_{attempt}"
                    / "agent",
                )
                if run.returncode != 0 or run.timed_out:
                    feedback = (
                        f"agent failed: returncode={run.returncode}, "
                        f"timed_out={run.timed_out}"
                    )
                    continue
            try:
                packet = load_test_packet(output)
            except InvalidTestPacket as exc:
                feedback = str(exc)
                continue
            try:
                incumbent_results = run_test_cases(
                    packet=packet,
                    bundle=output,
                    source=Path(state.incumbent_source),
                    subject="incumbent",
                    output_dir=iteration_dir
                    / "test_author"
                    / f"attempt_{attempt}"
                    / "admission",
                    python=self.config.benchmark.unit_python,
                    probe_runner=self.benchmark.run_agent_probe,
                )
            except InvalidTestPacket as exc:
                feedback = str(exc)
                continue
            admitted, reasons = admission_contract(packet, incumbent_results)
            write_json(
                iteration_dir
                / "test_author"
                / f"attempt_{attempt}"
                / "admission_summary.json",
                {"passed": admitted, "reasons": reasons},
            )
            if not admitted:
                feedback = "\n".join(reasons) or "admission contract failed"
                continue
            packet = freeze_test_packet(output, packet, admission_passed=True)
            name = (
                f"{_safe_name(packet.packet_id)}_v{packet.version}_"
                f"{packet.content_sha256[:12]}"
            )
            library_path = self.store.root / "test_library" / name
            if not library_path.exists():
                shutil.copytree(output, library_path)
            frozen = self.verified(library_path)
            self.store.append_event(
                "test_packet_admitted",
                iteration=iteration,
                packet_id=frozen.packet_id,
                admission_passed=frozen.admission_passed,
                path=str(library_path),
            )
            return frozen, library_path
        raise TestDesignFailure(feedback or "Test Author produced no admissible packet")


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
