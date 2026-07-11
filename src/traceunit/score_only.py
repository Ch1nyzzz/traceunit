from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from traceunit.agents.prompts import score_only_edit_prompt
from traceunit.agents.runner import WorkspaceAgent
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.candidate import CandidateBuildError
from traceunit.config import ProjectConfig
from traceunit.evaluation import mechanical_violations
from traceunit.io import copy_source, read_json, source_diff, write_json
from traceunit.models import (
    BenchmarkEvaluation,
    BenchmarkPlan,
    Decision,
    RunState,
    ScoreOnlyDecisionRecord,
    ScoreOnlyEvidence,
    ScoreOnlyProposal,
)
from traceunit.paired import paired_task_differences
from traceunit.store import RunStore
from traceunit.trace_evidence import stage_search_trace_evidence


class ScoreOnlyBuildError(CandidateBuildError):
    """Skips the iteration via the shared candidate-build failure path."""


class ScoreOnlyCandidateBuilder:
    """Meta-Harness-style editor with score, trace, history, and no unit proxy."""

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
    ) -> tuple[ScoreOnlyProposal, Path, str]:
        candidate_id = f"iter{iteration:03d}_candidate"
        candidate_dir = self.store.candidate_dir(candidate_id)
        source = candidate_dir / "source"
        proposal_path = candidate_dir / "score_only_proposal.json"
        trace_manifest = candidate_dir / "trace_evidence" / "manifest.json"
        history_path = candidate_dir / "score_history.json"
        if not proposal_path.is_file():
            copy_source(Path(state.incumbent_source), source)
            stage_search_trace_evidence(
                store=self.store,
                candidate_id=state.incumbent_id,
                destination=trace_manifest.parent,
                max_failure_traces=self.config.loop.max_failure_traces,
            )
            write_json(history_path, self._history())
            run = self.search_agent.run(
                role="score_only_editor",
                prompt=score_only_edit_prompt(
                    benchmark_context=self.benchmark.context(),
                    candidate_id=candidate_id,
                    parent_id=state.incumbent_id,
                    incumbent_search_score=state.incumbent_search_score,
                    source_dir=source,
                    trace_manifest=trace_manifest,
                    history_path=history_path,
                    proposal_path=proposal_path,
                    target_api_env=self.config.benchmark.api_key_env,
                ),
                workspace=candidate_dir,
                log_dir=iteration_dir / "score_only_editor",
            )
            if run.returncode != 0 or run.timed_out:
                raise ScoreOnlyBuildError(
                    f"score-only editor failed: returncode={run.returncode}, "
                    f"timed_out={run.timed_out}"
                )
        if not proposal_path.is_file() or not source.is_dir():
            raise ScoreOnlyBuildError(
                "score-only editor produced an incomplete candidate"
            )
        proposal = ScoreOnlyProposal.from_dict(read_json(proposal_path))
        if proposal.candidate_id != candidate_id:
            raise ScoreOnlyBuildError(
                f"proposal candidate {proposal.candidate_id!r} does not match "
                f"{candidate_id!r}"
            )
        if proposal.parent_id != state.incumbent_id:
            raise ScoreOnlyBuildError(
                f"proposal parent {proposal.parent_id!r} does not match "
                f"{state.incumbent_id!r}"
            )
        try:
            diff_text = source_diff(Path(state.incumbent_source), source)
        except ValueError as exc:
            raise ScoreOnlyBuildError(str(exc)) from exc
        return proposal, source, diff_text

    def _history(self) -> dict[str, Any]:
        decisions = []
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
                    "search_delta": evidence.get("search_delta"),
                }
            )
        return {"decisions": decisions}


class ScoreOnlyEvaluator:
    def __init__(
        self,
        *,
        config: ProjectConfig,
        store: RunStore,
        benchmark: BenchmarkAdapter,
        benchmark_plan: BenchmarkPlan,
    ) -> None:
        self.config = config
        self.store = store
        self.benchmark = benchmark
        self.plan = benchmark_plan

    def evaluate(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        proposal: ScoreOnlyProposal,
        candidate_source: Path,
        diff_text: str,
    ) -> tuple[ScoreOnlyEvidence, ScoreOnlyDecisionRecord]:
        violations = mechanical_violations(
            benchmark=self.benchmark,
            candidate_source=candidate_source,
            diff_text=diff_text,
            out_dir=iteration_dir / "smoke",
        )
        if violations:
            evidence = ScoreOnlyEvidence(
                iteration=iteration,
                candidate_id=proposal.candidate_id,
                parent_id=state.incumbent_id,
                search_delta=None,
                metadata={"violations": violations},
            )
            return evidence, ScoreOnlyDecisionRecord(
                iteration=iteration,
                candidate_id=proposal.candidate_id,
                decision=Decision.REJECT,
                reason="; ".join(violations),
                confidence=1.0,
                evidence=evidence,
            )
        candidate = self.benchmark.evaluate(
            source=candidate_source,
            candidate_id=proposal.candidate_id,
            pool=self.plan.search,
            out_dir=self.store.evaluation_dir(
                proposal.candidate_id, self.plan.search.slice_id
            ),
        )
        parent_path = self.store.evaluation_dir(
            state.incumbent_id, self.plan.search.slice_id
        )
        parent = BenchmarkEvaluation.from_dict(
            read_json(parent_path / "evaluation.json")
        )
        differences = paired_task_differences(parent, candidate)
        search_delta = statistics.fmean(differences)
        evidence = ScoreOnlyEvidence(
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            parent_id=state.incumbent_id,
            search_delta=search_delta,
            total_cost=candidate.cost,
            metadata={
                "violations": [],
                "candidate_source": str(candidate_source.resolve()),
                "search": {
                    "candidate_score": candidate.score,
                    "candidate_passrate": candidate.passrate,
                    "paired_task_count": len(differences),
                },
            },
        )
        if search_delta > self.config.decision.min_search_delta:
            decision = Decision.PROMOTE
            reason = "paired search score improved over the incumbent"
        else:
            decision = Decision.REJECT
            reason = "paired search score did not improve over the incumbent"
        return evidence, ScoreOnlyDecisionRecord(
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            decision=decision,
            reason=reason,
            confidence=1.0,
            evidence=evidence,
        )
