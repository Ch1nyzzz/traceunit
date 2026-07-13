from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from traceunit.agents.prompts import (
    candidate_edit_prompt,
    candidate_retry_prompt,
)
from traceunit.agents.runner import WorkspaceAgent
from traceunit.battery import Battery
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.evaluation import UnitEvidence, UnitEvidenceRunner
from traceunit.io import copy_source, read_json, source_diff, write_json
from traceunit.models import CandidateProposal, RunState, UnitFamily
from traceunit.store import RunStore
from traceunit.trace_evidence import stage_search_trace_evidence
from traceunit.ut_memory import WorldModel


class CandidateBuildError(RuntimeError):
    pass


class CandidateBuilder:
    """Stage inputs, then run the proposer inside a cheap battery retry loop.

    The capability battery is the proposer's fast alignment check: propose a
    patch, run the battery, and on failure hand the concrete per-capability
    results back for another attempt. Only after the loop (pass or retries
    exhausted) does the expensive paired search evaluation run.
    """

    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        search_agent: WorkspaceAgent,
        battery: Battery,
        world_model: WorldModel | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.search_agent = search_agent
        self.battery = battery
        self.world_model = world_model
        self.unit_runner = UnitEvidenceRunner(
            config=config, store=store, benchmark=benchmark, battery=battery
        )

    def build(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        target_capability: str,
        target_family: UnitFamily,
    ) -> tuple[CandidateProposal, Path, str, UnitEvidence]:
        candidate_id = f"iter{iteration:03d}_candidate"
        candidate_dir = self.store.candidate_dir(candidate_id)
        source = candidate_dir / "source"
        proposal_path = candidate_dir / "proposal.json"
        target_path = candidate_dir / "target_capability.json"
        history_path = candidate_dir / "history.json"
        feedback_path = candidate_dir / "unit_feedback.json"
        inner_state_path = candidate_dir / "inner_state.json"

        if not source.is_dir():
            copy_source(Path(state.incumbent_source), source)
        trace_manifest = candidate_dir / "trace_evidence" / "manifest.json"
        if not target_path.is_file():
            write_json(
                target_path,
                self._target_view(target_capability, target_family),
            )
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
                        target_capability_path=target_path,
                        feedback_path=feedback_path,
                        proposal_path=proposal_path,
                    )
                    if attempts > 0 and feedback_path.is_file()
                    else candidate_edit_prompt(
                        benchmark_context=self.benchmark.context(),
                        candidate_id=candidate_id,
                        parent_id=state.incumbent_id,
                        source_dir=source,
                        target_capability_path=target_path,
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
                target_capability=target_capability,
                candidate_source=source,
                diff_text=diff_text,
                output_dir=iteration_dir / "unit_loop" / f"attempt_{attempts:02d}",
            )
            total_seconds += unit.unit_seconds
            total_probe_calls += unit.probe_calls
            total_probe_tokens += unit.probe_tokens
            if unit.unit_ok() or attempts >= max_attempts:
                break
            write_json(
                feedback_path,
                unit_feedback(unit, battery=self.battery, attempt=attempts),
            )
            self._run_editor(
                prompt=candidate_retry_prompt(
                    attempt=attempts + 1,
                    max_attempts=max_attempts,
                    source_dir=source,
                    target_capability_path=target_path,
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
        )
        return proposal, source, diff_text, unit

    def _target_view(
        self, target_capability: str, target_family: UnitFamily
    ) -> dict[str, Any]:
        """The target capability as staged to the proposer: the group's
        description and its instances' behavior descriptions plus incumbent
        results - the capability spec, never the probe files themselves."""

        summary = self.battery.state_summary()
        group = next(
            (
                item
                for item in summary["capabilities"]
                if item["capability"] == target_capability
            ),
            None,
        )
        return {
            "target_capability": target_capability,
            "family": target_family.value,
            "description": (group or {}).get("description", ""),
            "instances": (group or {}).get("instances", []),
        }

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
                            "target_capability",
                            "target_improved",
                            "collateral_ok",
                            "target_delta",
                            "collateral_delta",
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

        Each record is an earlier edit worth reading: its battery verdict
        passed while search stayed flat, or its search improved while the
        battery did not certify it. The proposer may rebuild what it judges
        valuable; nothing is applied or replayed automatically.
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


def unit_feedback(
    unit: UnitEvidence, *, battery: Battery, attempt: int
) -> dict[str, Any]:
    """Concrete battery feedback for the proposer's next inner attempt.

    Target-capability instances come with their behavior descriptions and
    outcomes; damaged other capabilities are named with the instances that
    flipped, so collateral damage is visible while the patch is still cheap
    to change.
    """

    descriptions = {
        item.instance_id: item.description for item in battery.load()
    }
    reference = battery.load_reference()
    target_instances = []
    collateral: dict[str, list[dict[str, Any]]] = {}
    for result in unit.results:
        entry = {
            "instance_id": result.instance_id,
            "description": descriptions.get(result.instance_id, ""),
            "incumbent_passed": reference.get(result.instance_id),
            "candidate_passed": result.passed,
            "timed_out": result.timed_out,
            "error": result.error,
        }
        if result.capability == unit.target_capability:
            target_instances.append(entry)
        elif reference.get(result.instance_id) and not result.passed:
            collateral.setdefault(result.capability, []).append(entry)
    return {
        "attempt": attempt,
        "violations": list(unit.violations),
        "target_capability": unit.target_capability,
        "target_improved": unit.target_improved,
        "target_instances": target_instances,
        "collateral_ok": unit.collateral_ok,
        "damaged_capabilities": collateral,
        "capability_deltas": unit.deltas,
    }
