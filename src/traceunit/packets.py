from __future__ import annotations

import shutil
from pathlib import Path

from traceunit.agents.prompts import test_author_prompt
from traceunit.agents.runner import WorkspaceAgent
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.io import copy_source, read_json, write_json
from traceunit.models import (
    RunState,
    TestExecutionMode,
    TestPacket,
)
from traceunit.store import RunStore
from traceunit.trace_evidence import (
    TraceEvidenceError,
    stage_search_trace_evidence,
)
from traceunit.tests_runtime import (
    InvalidTestPacket,
    admission_contract,
    freeze_test_packet,
    load_test_packet,
    run_test_cases,
    verify_frozen_packet,
)
from traceunit.ut_memory import WorldModel


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
        world_model: WorldModel | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.agent = agent
        self.world_model = world_model

    def get_or_author(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
    ) -> tuple[TestPacket, Path]:
        """Author one fresh packet per iteration; packet_ref.json is resume-only."""

        packet_ref = iteration_dir / "packet_ref.json"
        if packet_ref.is_file():
            ref = read_json(packet_ref)
            path = Path(str(ref["path"]))
            return self.verified(path), path

        packet, path = self._author(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
        )
        write_json(packet_ref, {"path": str(path), "packet_id": packet.packet_id})
        return packet, path

    def verified(self, path: Path) -> TestPacket:
        packet = load_test_packet(path)
        if not verify_frozen_packet(path, packet):
            raise TestDesignFailure(f"frozen TestPacket hash mismatch: {path}")
        return packet

    def _stage_memory_inputs(self, workspace: Path) -> dict[str, Path | None]:
        """Stage the world model, the previous-iteration digest, and any
        mismatch evidence (record, diff, and the mismatch candidate's failed
        search traces) into the author's workspace."""

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

    def _commit_world_model(self, workspace: Path, *, iteration: int) -> None:
        """Copy the author's world-model file back; record whether it grew.

        No fallback and no template: when the author skipped its distill, the
        run records that fact instead of papering over it.
        """

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
            memory_inputs = self._stage_memory_inputs(workspace)
            prompt = test_author_prompt(
                benchmark_context=self.benchmark.context(),
                trace_manifest=trace_manifest,
                incumbent_source=incumbent_copy,
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
                self._commit_world_model(workspace, iteration=iteration)
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
            if not self.benchmark.supports_agent_probe and any(
                case.execution_mode is TestExecutionMode.MODEL_BACKED_PROBE
                for case in packet.cases
            ):
                feedback = (
                    "this benchmark does not support model_backed_probe cases; "
                    "every case must be deterministic"
                )
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
