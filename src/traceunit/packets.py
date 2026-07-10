from __future__ import annotations

import shutil
from pathlib import Path

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
    admission_score,
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
        alignment_cards_path: Path | None,
    ) -> tuple[TestPacket, Path, bool]:
        packet_ref = iteration_dir / "packet_ref.json"
        if packet_ref.is_file():
            path = Path(str(read_json(packet_ref)["path"]))
            packet = self._verified(path)
            return packet, path, bool(read_json(packet_ref).get("reused"))
        if state.active_packet_path:
            path = Path(state.active_packet_path)
            try:
                packet = self._verified(path)
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
            alignment_cards_path=alignment_cards_path,
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

    def _verified(self, path: Path) -> TestPacket:
        packet = load_test_packet(path)
        if not verify_frozen_packet(path, packet):
            raise TestDesignFailure(f"frozen TestPacket hash mismatch: {path}")
        return packet

    def _author(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        alignment_cards_path: Path | None,
    ) -> tuple[TestPacket, Path]:
        feedback = ""
        for attempt in range(1, 3):
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
            cards_copy = workspace / "alignment_cards.json"
            if alignment_cards_path is not None and not cards_copy.exists():
                shutil.copy2(alignment_cards_path, cards_copy)
            prompt = test_author_prompt(
                benchmark_context=self.benchmark.context(),
                trace_manifest=trace_manifest,
                incumbent_source=incumbent_copy,
                alignment_cards_path=(
                    cards_copy if alignment_cards_path is not None else None
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
            )
            score, reasons = admission_score(packet, incumbent_results)
            write_json(
                iteration_dir
                / "test_author"
                / f"attempt_{attempt}"
                / "admission_summary.json",
                {"score": score, "reasons": reasons},
            )
            if score < self.config.decision.min_admission_score:
                feedback = "\n".join(reasons) or f"admission score {score:.3f}"
                continue
            packet = freeze_test_packet(output, packet, admission_score=score)
            name = (
                f"{_safe_name(packet.packet_id)}_v{packet.version}_"
                f"{packet.content_sha256[:12]}"
            )
            library_path = self.store.root / "test_library" / name
            if not library_path.exists():
                shutil.copytree(output, library_path)
            frozen = self._verified(library_path)
            self.store.append_event(
                "test_packet_admitted",
                iteration=iteration,
                packet_id=frozen.packet_id,
                admission_score=frozen.admission_score,
                path=str(library_path),
            )
            return frozen, library_path
        raise TestDesignFailure(feedback or "Test Author produced no admissible packet")


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
