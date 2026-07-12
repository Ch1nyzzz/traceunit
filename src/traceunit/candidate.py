from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from traceunit.agents.prompts import candidate_edit_prompt, public_packet
from traceunit.agents.runner import WorkspaceAgent
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import ProjectConfig
from traceunit.io import copy_source, read_json, source_diff, write_json
from traceunit.models import CandidateProposal, RunState, TestPacket, TestTier
from traceunit.store import RunStore


class CandidateBuildError(RuntimeError):
    pass


class CandidateBuilder:
    """Stage public inputs and archive records, then run one edit agent."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        search_agent: WorkspaceAgent,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.search_agent = search_agent

    def build(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        packet: TestPacket,
        packet_path: Path,
    ) -> tuple[CandidateProposal, Path, str]:
        candidate_id = f"iter{iteration:03d}_candidate"
        candidate_dir = self.store.candidate_dir(candidate_id)
        source = candidate_dir / "source"
        proposal_path = candidate_dir / "proposal.json"
        public_path = candidate_dir / "public_packet.json"
        history_path = candidate_dir / "history.json"

        if not proposal_path.is_file():
            copy_source(Path(state.incumbent_source), source)
            write_json(public_path, public_packet(packet))
            for case in packet.cases:
                if case.tier is not TestTier.PUBLIC:
                    continue
                target = candidate_dir / case.path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(packet_path / case.path, target)
            write_json(history_path, self.public_history())
            archives = self.public_archives(state, candidate_dir)
            archives_path = candidate_dir / "archives.json" if archives else None
            if archives_path is not None:
                write_json(archives_path, {"archives": archives})
            prompt = candidate_edit_prompt(
                benchmark_context=self.benchmark.context(),
                candidate_id=candidate_id,
                parent_id=state.incumbent_id,
                source_dir=source,
                public_packet_path=public_path,
                history_path=history_path,
                archives_path=archives_path,
                proposal_path=proposal_path,
                target_api_env=self.config.benchmark.api_key_env,
            )
            run = self.search_agent.run(
                role="candidate_editor",
                prompt=prompt,
                workspace=candidate_dir,
                log_dir=iteration_dir / "candidate_editor",
            )
            if run.returncode != 0 or run.timed_out:
                raise CandidateBuildError(
                    f"candidate editor failed: returncode={run.returncode}, "
                    f"timed_out={run.timed_out}"
                )

        if not proposal_path.is_file() or not source.is_dir():
            raise CandidateBuildError(
                "candidate build is incomplete; missing proposal or source"
            )
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
        try:
            diff_text = source_diff(Path(state.incumbent_source), source)
        except ValueError as exc:
            raise CandidateBuildError(str(exc)) from exc
        return proposal, source, diff_text

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
