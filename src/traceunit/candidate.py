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
            write_json(history_path, self.public_history(candidate_dir))
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
        mechanism description and its instances' incumbent results under
        opaque codes - never instance ids, descriptions, or probe files.

        Instance ids and descriptions carry the probes' fictional surface
        vocabulary; an editor that sees them keyword-matches the probes
        instead of repairing the mechanism (the v4 AppWorld run promoted a
        patch whose regexes were lifted from instance descriptions)."""

        summary = self.battery.state_summary()
        group = next(
            (
                item
                for item in summary["capabilities"]
                if item["capability"] == target_capability
            ),
            None,
        )
        codes = opaque_instance_codes(self.battery, target_capability)
        return {
            "target_capability": target_capability,
            "family": target_family.value,
            "description": (group or {}).get("description", ""),
            "instances": [
                {
                    "instance": codes[item["instance_id"]],
                    "incumbent_passed": item.get("incumbent_passed"),
                }
                for item in (group or {}).get("instances", [])
                if item.get("instance_id") in codes
            ],
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

    def public_history(self, workspace: Path | None = None) -> dict[str, Any]:
        """Every prior candidate's decision, reason, claimed mechanism, and
        (when a workspace is given) its staged diff.

        A later editor that cannot see what was tried and why it failed
        re-proposes the same dead mechanisms; there is no reason to withhold
        any of it - the diffs contain only earlier editors' own work.
        """

        decisions: list[dict[str, Any]] = []
        for path in sorted(
            (self.store.root / "iterations").glob("iter_*/decision.json")
        ):
            raw = read_json(path)
            evidence = dict(raw.get("evidence") or {})
            entry: dict[str, Any] = {
                "iteration": raw.get("iteration"),
                "candidate_id": raw.get("candidate_id"),
                "decision": raw.get("decision"),
                "reason": raw.get("reason"),
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
            proposal_path = path.parent / "candidate_proposal.json"
            if proposal_path.is_file():
                proposal = read_json(proposal_path)
                entry["mechanism_claim"] = proposal.get("mechanism_claim")
                entry["intervention_kind"] = proposal.get("intervention_kind")
            diff_path = path.parent / "candidate.diff"
            if workspace is not None and diff_path.is_file():
                staged = (
                    workspace
                    / "history_diffs"
                    / f"{path.parent.name}.diff"
                )
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(diff_path, staged)
                entry["diff_path"] = str(staged)
            decisions.append(entry)
        return {"decisions": decisions}

    def public_archives(
        self, state: RunState, workspace: Path
    ) -> list[dict[str, Any]]:
        return stage_archive_records(state, workspace)


def stage_archive_records(
    state: RunState, workspace: Path
) -> list[dict[str, Any]]:
    """Stage archived-candidate records as reference material.

    Each record is an earlier edit worth reading: its battery verdict
    passed while search stayed flat, or its search improved while the
    battery did not certify it. Both the Candidate Editor (re-litigation)
    and the Test Author (designing instances sensitive to the mechanisms
    the battery missed) read them; nothing is applied or replayed
    automatically.
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


def opaque_instance_codes(
    battery: Battery, target_capability: str
) -> dict[str, str]:
    """Stable editor-side codes for the target group's active instances.

    The battery is frozen while a candidate is built, so sorting the active
    instance ids gives the same code for the same instance across every inner
    attempt of an iteration - without exposing the id itself."""

    ids = sorted(
        item.instance_id
        for item in battery.active()
        if item.capability == target_capability
    )
    return {
        instance_id: f"instance_{index + 1:02d}"
        for index, instance_id in enumerate(ids)
    }


def unit_feedback(
    unit: UnitEvidence, *, battery: Battery, attempt: int
) -> dict[str, Any]:
    """Concrete battery feedback for the proposer's next inner attempt.

    Target-capability instances report pass/fail, timeouts, and budget
    exhaustion under opaque codes; damaged other capabilities are named at
    the group level with flip counts. The mechanism description in the target
    view is the direction to repair - the probes' surfaces stay hidden so the
    editor cannot keyword-match them.
    """

    codes = opaque_instance_codes(battery, unit.target_capability)
    reference = battery.load_reference()
    target_instances = []
    collateral: dict[str, int] = {}
    for result in unit.results:
        if result.capability == unit.target_capability:
            target_instances.append(
                {
                    "instance": codes.get(result.instance_id, "instance_??"),
                    "incumbent_passed": reference.get(result.instance_id),
                    "candidate_passed": result.passed,
                    "timed_out": result.timed_out,
                    "budget_exhausted": "token budget" in (result.error or ""),
                    "error": result.error,
                }
            )
        elif reference.get(result.instance_id) and not result.passed:
            collateral[result.capability] = collateral.get(result.capability, 0) + 1
    target_instances.sort(key=lambda entry: str(entry["instance"]))
    return {
        "attempt": attempt,
        "violations": list(unit.violations),
        "target_capability": unit.target_capability,
        "target_improved": unit.target_improved,
        "target_instances": target_instances,
        "collateral_ok": unit.collateral_ok,
        "damaged_capabilities": {
            capability: {"instances_flipped": count}
            for capability, count in sorted(collateral.items())
        },
        "capability_deltas": unit.deltas,
    }
