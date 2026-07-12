from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from traceunit.agents.prompts import (
    candidate_edit_prompt,
    candidate_retry_prompt,
    public_packet,
)
from traceunit.agents.runner import WorkspaceAgent
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.evaluation import UnitEvidence, UnitEvidenceRunner
from traceunit.io import copy_source, read_json, source_diff, write_json
from traceunit.models import (
    CandidateProposal,
    RunState,
    TestExecution,
    TestPacket,
    TestTier,
)
from traceunit.store import RunStore
from traceunit.trace_evidence import stage_search_trace_evidence
from traceunit.ut_memory import WorldModel


class CandidateBuildError(RuntimeError):
    pass


class CandidateBuilder:
    """Stage inputs, then run the proposer inside a cheap unit retry loop.

    The frozen packet is the proposer's fast alignment check: propose a patch,
    run the unit tests, and on failure hand the concrete failures back for
    another attempt. Only after the loop (pass or retries exhausted) does the
    expensive paired search evaluation run.
    """

    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        search_agent: WorkspaceAgent,
        world_model: WorldModel | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.search_agent = search_agent
        self.world_model = world_model
        self.unit_runner = UnitEvidenceRunner(
            config=config, store=store, benchmark=benchmark
        )

    def build(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        packet: TestPacket,
        packet_path: Path,
    ) -> tuple[CandidateProposal, Path, str, UnitEvidence]:
        candidate_id = f"iter{iteration:03d}_candidate"
        candidate_dir = self.store.candidate_dir(candidate_id)
        source = candidate_dir / "source"
        proposal_path = candidate_dir / "proposal.json"
        public_path = candidate_dir / "public_packet.json"
        history_path = candidate_dir / "history.json"
        feedback_path = candidate_dir / "unit_feedback.json"
        inner_state_path = candidate_dir / "inner_state.json"

        if not source.is_dir():
            copy_source(Path(state.incumbent_source), source)
        trace_manifest = candidate_dir / "trace_evidence" / "manifest.json"
        if not public_path.is_file():
            write_json(public_path, public_packet(packet))
            for case in packet.cases:
                if case.tier is not TestTier.PUBLIC:
                    continue
                target = candidate_dir / case.path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(packet_path / case.path, target)
            stage_search_trace_evidence(
                store=self.store,
                candidate_id=state.incumbent_id,
                destination=trace_manifest.parent,
                max_failure_traces=self.config.loop.max_failure_traces,
            )
            write_json(history_path, self.public_history())
            archives = self.public_archives(state, candidate_dir)
            if archives:
                write_json(candidate_dir / "archives.json", {"archives": archives})
            if self.world_model is not None:
                self.world_model.stage_into(candidate_dir)
        archives_path = candidate_dir / "archives.json"
        world_model_path = candidate_dir / "ut_design_world_model.md"

        attempts = 0
        if inner_state_path.is_file():
            attempts = int(read_json(inner_state_path).get("attempts") or 0)
        max_attempts = 1 + self.config.loop.max_inner_retries
        incumbent_results: tuple[TestExecution, ...] | None = None
        total_seconds = 0.0
        total_probe_calls = 0
        total_probe_tokens = 0

        while True:
            if attempts == 0 or not proposal_path.is_file():
                prompt = (
                    candidate_retry_prompt(
                        attempt=attempts + 1,
                        max_attempts=max_attempts,
                        source_dir=source,
                        public_packet_path=public_path,
                        feedback_path=feedback_path,
                        proposal_path=proposal_path,
                    )
                    if attempts > 0 and feedback_path.is_file()
                    else candidate_edit_prompt(
                        benchmark_context=self.benchmark.context(),
                        candidate_id=candidate_id,
                        parent_id=state.incumbent_id,
                        source_dir=source,
                        public_packet_path=public_path,
                        trace_manifest=trace_manifest,
                        incumbent_search_score=state.incumbent_search_score,
                        history_path=history_path,
                        archives_path=(
                            archives_path if archives_path.is_file() else None
                        ),
                        world_model_path=(
                            world_model_path
                            if world_model_path.is_file()
                            else None
                        ),
                        proposal_path=proposal_path,
                        target_api_env=self.config.benchmark.api_key_env,
                    )
                )
                self._run_editor(
                    prompt=prompt,
                    candidate_dir=candidate_dir,
                    log_dir=iteration_dir
                    / "candidate_editor"
                    / f"attempt_{attempts + 1:02d}",
                )
                attempts += 1
                write_json(inner_state_path, {"attempts": attempts})
            if not proposal_path.is_file() or not source.is_dir():
                raise CandidateBuildError(
                    "candidate build is incomplete; missing proposal or source"
                )
            diff_text = self._diff(state, source)
            unit = self.unit_runner.run(
                packet=packet,
                packet_path=packet_path,
                incumbent_source=Path(state.incumbent_source),
                candidate_source=source,
                preserved_refs=state.preserved_packet_refs,
                diff_text=diff_text,
                output_dir=iteration_dir / "unit_loop" / f"attempt_{attempts:02d}",
                incumbent_results=incumbent_results,
            )
            incumbent_results = unit.incumbent_results or incumbent_results
            total_seconds += unit.unit_seconds
            total_probe_calls += unit.probe_calls
            total_probe_tokens += unit.probe_tokens
            if (
                unit.unit_ok(self.config.decision.max_regression_loss)
                or attempts >= max_attempts
            ):
                break
            write_json(feedback_path, unit_feedback(unit, packet, attempt=attempts))
            self._run_editor(
                prompt=candidate_retry_prompt(
                    attempt=attempts + 1,
                    max_attempts=max_attempts,
                    source_dir=source,
                    public_packet_path=public_path,
                    feedback_path=feedback_path,
                    proposal_path=proposal_path,
                ),
                candidate_dir=candidate_dir,
                log_dir=iteration_dir
                / "candidate_editor"
                / f"attempt_{attempts + 1:02d}",
            )
            attempts += 1
            write_json(inner_state_path, {"attempts": attempts})

        unit = replace(
            unit,
            attempts=attempts,
            unit_seconds=total_seconds,
            probe_calls=total_probe_calls,
            probe_tokens=total_probe_tokens,
        )
        proposal = self._validated_proposal(
            proposal_path=proposal_path,
            candidate_id=candidate_id,
            state=state,
            packet=packet,
        )
        return proposal, source, diff_text, unit

    def _run_editor(
        self, *, prompt: str, candidate_dir: Path, log_dir: Path
    ) -> None:
        run = self.search_agent.run(
            role="candidate_editor",
            prompt=prompt,
            workspace=candidate_dir,
            log_dir=log_dir,
        )
        if run.returncode != 0 or run.timed_out:
            raise CandidateBuildError(
                f"candidate editor failed: returncode={run.returncode}, "
                f"timed_out={run.timed_out}"
            )

    def _diff(self, state: RunState, source: Path) -> str:
        try:
            return source_diff(Path(state.incumbent_source), source)
        except ValueError as exc:
            raise CandidateBuildError(str(exc)) from exc

    def _validated_proposal(
        self,
        *,
        proposal_path: Path,
        candidate_id: str,
        state: RunState,
        packet: TestPacket,
    ) -> CandidateProposal:
        try:
            proposal = CandidateProposal.from_dict(read_json(proposal_path))
        except (KeyError, ValueError) as exc:
            # Quarantine the malformed file so a resumed run re-runs the editor
            # instead of re-parsing the same bad proposal forever.
            proposal_path.rename(proposal_path.with_suffix(".invalid.json"))
            raise CandidateBuildError(f"invalid proposal.json: {exc}") from exc
        if proposal.candidate_id != candidate_id:
            raise CandidateBuildError(
                f"proposal candidate {proposal.candidate_id!r} does not match "
                f"{candidate_id!r}"
            )
        if proposal.parent_id != state.incumbent_id:
            raise CandidateBuildError(
                f"proposal parent {proposal.parent_id!r} does not match "
                f"{state.incumbent_id!r}"
            )
        if proposal.hypothesis_id != packet.target_hypothesis_id:
            raise CandidateBuildError(
                "proposal does not target the frozen TestPacket hypothesis"
            )
        target_hypothesis = next(
            item
            for item in packet.hypotheses
            if item.hypothesis_id == packet.target_hypothesis_id
        )
        if proposal.intervention_kind is not target_hypothesis.intervention_kind:
            raise CandidateBuildError(
                "proposal intervention_kind does not match the frozen hypothesis"
            )
        return proposal

    def public_history(self) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        for path in sorted(
            (self.store.root / "iterations").glob("iter_*/decision.json")
        ):
            raw = read_json(path)
            evidence = dict(raw.get("evidence") or {})
            decisions.append(
                {
                    "iteration": raw.get("iteration"),
                    "candidate_id": raw.get("candidate_id"),
                    "decision": raw.get("decision"),
                    "aggregate_evidence": {
                        key: evidence.get(key)
                        for key in (
                            "contract_passed",
                            "bridge_contract_passed",
                            "preservation_passed",
                            "regression_loss",
                            "search_delta",
                        )
                        if key in evidence
                    },
                }
            )
        return {"decisions": decisions}

    def public_archives(
        self, state: RunState, workspace: Path
    ) -> list[dict[str, Any]]:
        """Stage archived-candidate records as reference material.

        Each record is an earlier edit worth reading: its contract passed while
        search stayed flat, or its search improved while its contract failed.
        The proposer may rebuild what it judges valuable; nothing is applied
        or replayed automatically.
        """

        archives: list[dict[str, Any]] = []
        for raw_ref in state.archive_refs[-8:]:
            archive_dir = Path(str(raw_ref.get("path") or ""))
            record_path = archive_dir / "record.json"
            diff_path = archive_dir / "candidate.diff"
            if not record_path.is_file():
                continue
            record = read_json(record_path)
            entry = dict(record)
            if diff_path.is_file():
                staged = (
                    workspace
                    / "archive"
                    / str(record["candidate_id"])
                    / "candidate.diff"
                )
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(diff_path, staged)
                entry["diff_path"] = str(staged)
            archives.append(entry)
        return archives


def _output_tail(path: str, limit: int = 2000) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def unit_feedback(
    unit: UnitEvidence, packet: TestPacket, *, attempt: int
) -> dict[str, Any]:
    """Concrete failure feedback for the proposer's next inner attempt.

    Public cases include their captured output; hidden cases expose only
    their declared description and pass/fail state, so the hidden tier keeps
    measuring generalization rather than becoming a second visible target.
    Preserved-contract failures are spelled out: those contracts were public
    when their candidates were promoted.
    """

    by_case = {case.case_id: case for case in packet.cases}
    failed_cases: list[dict[str, Any]] = []
    for result in unit.candidate_results:
        spec = by_case.get(result.case_id)
        expected = spec.expected_candidate_pass if spec else True
        if result.passed == expected:
            continue
        entry: dict[str, Any] = {
            "case_id": result.case_id,
            "tier": result.tier.value,
            "evidence_role": result.evidence_role.value,
            "description": spec.description if spec else "",
            "expected_pass": expected,
            "passed": result.passed,
            "timed_out": result.timed_out,
            "error": result.error,
        }
        if result.tier is TestTier.PUBLIC:
            entry["stdout_tail"] = _output_tail(result.stdout_path)
            entry["stderr_tail"] = _output_tail(result.stderr_path)
        failed_cases.append(entry)
    return {
        "attempt": attempt,
        "violations": list(unit.violations),
        "contract_passed": unit.contract_passed,
        "contract_reasons": list(unit.contract_reasons),
        "failed_cases": failed_cases,
        "preserved_contract_failures": [
            {"packet_id": item.packet_id, "reasons": list(item.reasons)}
            for item in unit.preservation
            if not item.contract_passed
        ],
        "regression_loss": unit.metrics.get("regression_loss", 0.0),
    }
