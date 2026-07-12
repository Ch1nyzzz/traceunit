from __future__ import annotations
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping
from traceunit.agents.runner import WorkspaceAgent, build_agent
from traceunit.benchmarks import BenchmarkAdapter, build_benchmark
from traceunit.candidate import CandidateBuildError, CandidateBuilder
from traceunit.config import ProjectConfig
from traceunit.decision import DecisionPolicy, archive_kind, is_mismatch, unit_ok
from traceunit.evaluation import CandidateEvaluator
from traceunit.io import copy_source, read_json, write_json
from traceunit.replay import FrozenPacketRef, copy_packet_into_store
from traceunit.models import (
    BenchmarkPlan,
    CandidateProposal,
    Decision,
    DecisionRecord,
    EvidenceRecord,
    RunState,
    TestPacket,
)
from traceunit.ontology import freeze_ontology, ontology_ref
from traceunit.packets import PacketAuthor, TestDesignFailure
from traceunit.score_only import ScoreOnlyCandidateBuilder, ScoreOnlyEvaluator
from traceunit.store import RunStore
from traceunit.ut_memory import WorldModel
from traceunit.trace_evidence import (
    NoFailureTraces,
    TraceEvidenceError,
)


# Consecutive iteration skips (failed packet authoring or candidate build,
# e.g. an exhausted agent quota) halt the run instead of burning the budget;
# the skipped iterations are handed back and the run resumes later.
MAX_CONSECUTIVE_SKIPS = 3


class OptimizationLoop:
    """Orchestrate search; specialized modules own every protocol mechanism."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        benchmark: BenchmarkAdapter | None = None,
        agents: Mapping[str, WorkspaceAgent] | None = None,
    ) -> None:
        self.config = config
        self.store = RunStore(config.loop.run_dir)
        self.benchmark = benchmark or build_benchmark(config.benchmark)
        supplied = dict(agents or {})

        def _with_target_env(agent_config):
            # Role agents may call the frozen target model for live
            # experimentation while they work; frozen tests never can.
            env = {
                "TRACEUNIT_TARGET_MODEL": config.benchmark.model,
                "TRACEUNIT_TARGET_BASE_URL": config.benchmark.base_url,
            }
            key = os.environ.get(config.benchmark.api_key_env, "")
            if key:
                env[config.benchmark.api_key_env] = key
            return replace(
                agent_config, environment={**env, **agent_config.environment}
            )

        self.test_author_agent = (
            supplied.get("test_author")
            or build_agent(_with_target_env(config.agents.test_author))
            if config.capabilities.generated_packets
            else None
        )
        self.search_agent = supplied.get("search") or build_agent(
            _with_target_env(config.agents.search)
        )
        self.regression_author_agent = (
            supplied.get("regression_author")
            or (
                build_agent(_with_target_env(config.agents.regression_author))
                if config.agents.regression_author.enabled
                else None
            )
            if config.capabilities.generated_packets
            else None
        )
        self.policy = DecisionPolicy(config.decision)
        self.world_model = WorldModel(self.store.ut_world_model_path)
        self.benchmark_plan: BenchmarkPlan | None = None
        self.packet_author: PacketAuthor | None = None
        self.candidate_builder: CandidateBuilder | None = None
        self.evaluator: CandidateEvaluator | None = None
        self.score_only_builder: ScoreOnlyCandidateBuilder | None = None
        self.score_only_evaluator: ScoreOnlyEvaluator | None = None

    def run(self) -> dict[str, Any]:
        self.store.initialize(
            config_snapshot=asdict(self.config),
            capabilities=asdict(self.config.capabilities),
        )
        freeze_ontology(self.store.ontology_path)
        self._preflight_agents()
        plan = self.benchmark.prepare(self.store.root)
        self._bind_plan(plan)
        self.benchmark.preflight()
        state = self.store.load_state()
        if state is not None and (
            state.condition != self.config.protocol.condition.value
            or state.capabilities != asdict(self.config.capabilities)
        ):
            raise RuntimeError(
                "run state condition/capabilities do not match the configured protocol"
            )
        if state is not None and not self.config.loop.resume:
            raise RuntimeError(
                f"run state already exists at {self.store.state_path}; "
                "enable resume or choose a new loop.run_dir"
            )
        if state is None:
            state = self._initialize_baseline()
        state.status = "running"
        self.store.save_state(state)
        if self.config.capabilities.online_ut_memory:
            self.world_model.ensure()

        consecutive_skips = 0
        while state.next_iteration <= self.config.loop.iterations:
            iteration = state.next_iteration
            try:
                state = self._run_iteration(state, iteration)
                consecutive_skips = 0
            except NoFailureTraces:
                state.status = "converged"
                self.store.append_event(
                    "search_converged",
                    iteration=iteration,
                    reason="incumbent has no failed search traces",
                )
                break
            except TestDesignFailure as exc:
                state = self._skip_failed_test_design(state, iteration, exc)
                consecutive_skips += 1
            except TraceEvidenceError as exc:
                state = self._skip_failed_trace_evidence(state, iteration, exc)
                consecutive_skips += 1
            except CandidateBuildError as exc:
                state = self._skip_failed_candidate_build(state, iteration, exc)
                consecutive_skips += 1
            except Exception as exc:
                state.status = "error"
                self.store.save_state(state)
                self.store.append_event(
                    "iteration_failed",
                    iteration=iteration,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                # Hand the skipped iterations back and stop: this pattern is
                # an environment failure (e.g. agent quota), not evidence.
                state.next_iteration -= consecutive_skips
                state.status = "halted"
                self.store.append_event(
                    "run_halted_after_consecutive_skips",
                    iteration=iteration,
                    skips=consecutive_skips,
                    resume_iteration=state.next_iteration,
                )
                break

        if state.status == "running":
            state.status = "completed"
        self.store.save_state(state)
        summary = self._summary(state)
        write_json(self.store.root / "summary.json", summary)
        return summary

    def _bind_plan(self, plan: BenchmarkPlan) -> None:
        if plan.ontology != ontology_ref():
            raise RuntimeError(
                "benchmark plan is not bound to the frozen TraceUnit L0 ontology"
            )
        self.benchmark_plan = plan
        write_json(self.store.benchmark_plan_path, plan.to_dict())
        if not self.config.capabilities.generated_packets:
            self.score_only_builder = ScoreOnlyCandidateBuilder(
                config=self.config,
                store=self.store,
                benchmark=self.benchmark,
                search_agent=self.search_agent,
            )
            self.score_only_evaluator = ScoreOnlyEvaluator(
                config=self.config,
                store=self.store,
                benchmark=self.benchmark,
                benchmark_plan=plan,
            )
            return
        assert self.test_author_agent is not None
        self.packet_author = PacketAuthor(
            config=self.config,
            store=self.store,
            benchmark=self.benchmark,
            agent=self.test_author_agent,
            world_model=(
                self.world_model
                if self.config.capabilities.online_ut_memory
                else None
            ),
        )
        self.candidate_builder = CandidateBuilder(
            config=self.config,
            store=self.store,
            benchmark=self.benchmark,
            search_agent=self.search_agent,
            world_model=(
                self.world_model
                if self.config.capabilities.online_ut_memory
                else None
            ),
        )
        self.evaluator = CandidateEvaluator(
            config=self.config,
            store=self.store,
            benchmark=self.benchmark,
            benchmark_plan=plan,
            policy=self.policy,
            regression_author=self.regression_author_agent,
        )

    def _preflight_agents(self) -> None:
        for agent in (
            self.test_author_agent,
            self.search_agent,
            self.regression_author_agent,
        ):
            preflight = getattr(agent, "preflight", None)
            if callable(preflight):
                preflight()

    def _initialize_baseline(self) -> RunState:
        baseline_dir = self.store.candidate_dir("baseline")
        source = baseline_dir / "source"
        if not source.exists():
            copy_source(self.benchmark.baseline_source(), source)
        search = self.benchmark.evaluate(
            source=source,
            candidate_id="baseline",
            pool=self._plan.search,
            out_dir=self.store.evaluation_dir("baseline", self._plan.search.slice_id),
        )
        state = RunState(
            run_id=self.config.loop.run_id or self.store.root.name,
            benchmark=self.benchmark.name,
            status="running",
            next_iteration=1,
            incumbent_id="baseline",
            incumbent_source=str(source.resolve()),
            incumbent_search_score=search.score,
            condition=self.config.protocol.condition.value,
            capabilities=asdict(self.config.capabilities),
            promoted_ids=["baseline"],
            search_cost=search.cost,
            total_cost=search.cost,
        )
        self.store.save_state(state)
        self.store.append_event(
            "baseline_collected",
            candidate_id="baseline",
            search_score=search.score,
            search_trace_path=search.trace_path,
        )
        return state

    def _run_iteration(self, state: RunState, iteration: int) -> RunState:
        if not self.config.capabilities.generated_packets:
            return self._run_score_only_iteration(state, iteration)
        assert self.packet_author is not None
        assert self.candidate_builder is not None
        assert self.evaluator is not None
        iteration_dir = self.store.iteration_dir(iteration)
        decision_path = iteration_dir / "decision.json"
        if decision_path.is_file():
            return self._resume_completed_iteration(
                state=state, iteration=iteration, iteration_dir=iteration_dir
            )
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "running",
                "incumbent": state.incumbent_id,
            },
        )
        packet, packet_path = self.packet_author.get_or_author(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
        )
        proposal, candidate_source, diff_text, unit = self.candidate_builder.build(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            packet=packet,
            packet_path=packet_path,
        )
        write_json(iteration_dir / "candidate_proposal.json", proposal.to_dict())
        (iteration_dir / "candidate.diff").write_text(diff_text, encoding="utf-8")
        evidence, decision = self.evaluator.evaluate_candidate(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            proposal=proposal,
            packet=packet,
            packet_path=packet_path,
            candidate_source=candidate_source,
            diff_text=diff_text,
            unit=unit,
        )
        decision = replace(decision, evidence=evidence)
        write_json(iteration_dir / "evidence.json", evidence.to_dict())
        write_json(decision_path, decision.to_dict())

        state = self._commit_decision(
            state=state,
            iteration=iteration,
            proposal=proposal,
            packet=packet,
            packet_path=packet_path,
            candidate_source=candidate_source,
            evidence=evidence,
            decision=decision,
        )
        self._record_last_iteration(
            iteration=iteration,
            proposal=proposal,
            evidence=evidence,
            decision=decision,
        )
        return self._finish_iteration(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            proposal=proposal,
            packet=packet,
            decision=decision,
        )

    def _resume_completed_iteration(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
    ) -> RunState:
        """Commit an existing decision artifact without recomputing evidence."""

        assert self.packet_author is not None
        decision_path = iteration_dir / "decision.json"
        evidence_path = iteration_dir / "evidence.json"
        proposal_path = iteration_dir / "candidate_proposal.json"
        packet_ref_path = iteration_dir / "packet_ref.json"
        required = (decision_path, evidence_path, proposal_path, packet_ref_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "cannot resume an incomplete decision artifact; missing "
                + ", ".join(missing)
            )
        decision = DecisionRecord.from_dict(read_json(decision_path))
        evidence = EvidenceRecord.from_dict(read_json(evidence_path))
        if decision.evidence.to_dict() != evidence.to_dict():
            raise RuntimeError("decision evidence does not match evidence.json")
        proposal = CandidateProposal.from_dict(read_json(proposal_path))
        packet_path = Path(str(read_json(packet_ref_path)["path"]))
        packet = self.packet_author.verified(packet_path)
        candidate_source = Path(
            str(
                evidence.metadata.get(
                    "candidate_source",
                    self.store.candidate_dir(proposal.candidate_id) / "source",
                )
            )
        )
        if not candidate_source.is_dir():
            raise RuntimeError("cannot resume: candidate source is missing")
        state = self._commit_decision(
            state=state,
            iteration=iteration,
            proposal=proposal,
            packet=packet,
            packet_path=packet_path,
            candidate_source=candidate_source,
            evidence=evidence,
            decision=decision,
        )
        self._record_last_iteration(
            iteration=iteration,
            proposal=proposal,
            evidence=evidence,
            decision=decision,
        )
        self.store.append_event(
            "iteration_resumed_from_decision",
            iteration=iteration,
            candidate_id=proposal.candidate_id,
        )
        return self._finish_iteration(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            proposal=proposal,
            packet=packet,
            decision=decision,
        )

    def _record_last_iteration(
        self,
        *,
        iteration: int,
        proposal: CandidateProposal,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
    ) -> None:
        """Write the previous-iteration digest the next Test Author reads.

        Nothing here is sanitized: the author gets the decision, the paired
        per-task outcomes, and pointers to the mismatch record when the unit
        verdict and search disagreed. What it learns from them is its own job,
        written into the append-only world model.
        """

        if not self.config.capabilities.online_ut_memory:
            return
        mismatch_dir = self.store.mismatch_root / f"iter_{iteration:03d}"
        write_json(
            self.store.memory_root / "last_iteration.json",
            {
                "iteration": iteration,
                "candidate_id": proposal.candidate_id,
                "decision": decision.decision.value,
                "reason": decision.reason,
                "mechanism_claim": proposal.mechanism_claim,
                "packet_id": evidence.packet_id,
                "search_delta": evidence.search_delta,
                "contract_passed": evidence.contract_passed,
                "preservation_passed": evidence.preservation_passed,
                "unit_attempts": evidence.metadata.get("unit_attempts"),
                "unit_failure_reasons": list(
                    evidence.metadata.get("candidate_contract_reasons") or []
                ),
                "task_flips": self._task_flips(
                    parent_id=str(evidence.metadata.get("parent_id") or ""),
                    candidate_id=proposal.candidate_id,
                ),
                "mismatch": is_mismatch(evidence, self.config.decision),
                "mismatch_path": (
                    str(mismatch_dir) if mismatch_dir.is_dir() else None
                ),
            },
        )

    def _finish_iteration(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        proposal: CandidateProposal,
        packet: TestPacket,
        decision: DecisionRecord,
    ) -> RunState:
        state.next_iteration = iteration + 1
        state.status = "running"
        self.store.save_state(state)
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "completed",
                "decision": decision.decision.value,
                "candidate_id": proposal.candidate_id,
                "next_iteration": state.next_iteration,
            },
        )
        self.store.append_event(
            "iteration_completed",
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            packet_id=packet.packet_id,
            decision=decision.decision.value,
            reason=decision.reason,
        )
        return state

    def _run_score_only_iteration(self, state: RunState, iteration: int) -> RunState:
        assert self.score_only_builder is not None
        assert self.score_only_evaluator is not None
        iteration_dir = self.store.iteration_dir(iteration)
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "running",
                "incumbent": state.incumbent_id,
                "condition": self.config.protocol.condition.value,
            },
        )
        proposal, candidate_source, diff_text = self.score_only_builder.build(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
        )
        evidence, decision = self.score_only_evaluator.evaluate(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            proposal=proposal,
            candidate_source=candidate_source,
            diff_text=diff_text,
        )
        write_json(iteration_dir / "candidate_proposal.json", proposal.to_dict())
        (iteration_dir / "candidate.diff").write_text(diff_text, encoding="utf-8")
        write_json(iteration_dir / "evidence.json", evidence.to_dict())
        write_json(iteration_dir / "decision.json", decision.to_dict())
        if iteration not in state.committed_iterations:
            state.total_cost += evidence.total_cost
            state.search_cost += evidence.total_cost
            if decision.decision is Decision.PROMOTE:
                state.incumbent_id = proposal.candidate_id
                state.incumbent_source = str(candidate_source.resolve())
                search = dict(evidence.metadata.get("search") or {})
                if "candidate_score" not in search:
                    raise RuntimeError("promoted candidate is missing its search score")
                state.incumbent_search_score = float(search["candidate_score"])
                if proposal.candidate_id not in state.promoted_ids:
                    state.promoted_ids.append(proposal.candidate_id)
            state.committed_iterations.append(iteration)
            self.store.save_state(state)
        state.next_iteration = iteration + 1
        state.status = "running"
        self.store.save_state(state)
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "completed",
                "decision": decision.decision.value,
                "candidate_id": proposal.candidate_id,
                "next_iteration": state.next_iteration,
            },
        )
        self.store.append_event(
            "iteration_completed",
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            decision=decision.decision.value,
            reason=decision.reason,
        )
        return state

    def _commit_decision(
        self,
        *,
        state: RunState,
        iteration: int,
        proposal: CandidateProposal,
        packet: TestPacket,
        packet_path: Path,
        candidate_source: Path,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
    ) -> RunState:
        if iteration in state.committed_iterations:
            return state
        state.total_cost += evidence.total_cost
        state.search_cost += evidence.total_cost
        if decision.decision is Decision.PROMOTE:
            state.incumbent_id = proposal.candidate_id
            state.incumbent_source = str(candidate_source.resolve())
            search = dict(evidence.metadata.get("search") or {})
            if "candidate_score" not in search:
                raise RuntimeError("promoted candidate is missing its search score")
            state.incumbent_search_score = float(search["candidate_score"])
            if proposal.candidate_id not in state.promoted_ids:
                state.promoted_ids.append(proposal.candidate_id)
            ref = self._store_packet(packet, packet_path)
            if ref.to_dict() not in state.preserved_packet_refs:
                state.preserved_packet_refs.append(ref.to_dict())
        elif decision.decision is Decision.ARCHIVE:
            self._retain_archive(
                state=state,
                iteration=iteration,
                proposal=proposal,
                evidence=evidence,
                decision=decision,
            )
        if is_mismatch(evidence, self.config.decision):
            self._stage_mismatch(
                iteration=iteration,
                proposal=proposal,
                packet=packet,
                packet_path=packet_path,
                evidence=evidence,
                decision=decision,
            )
        state.committed_iterations.append(iteration)
        self.store.save_state(state)
        return state

    def _retain_archive(
        self,
        *,
        state: RunState,
        iteration: int,
        proposal: CandidateProposal,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
    ) -> None:
        """Record an archived candidate for later agents to read and re-litigate.

        An archive is a record, not a protocol capability: nothing replays it
        and nothing migrates it. A later proposer that finds the idea worth
        rebuilding takes it through the normal propose -> unit -> search path.
        """

        if proposal.candidate_id not in state.archived_ids:
            state.archived_ids.append(proposal.candidate_id)
        if not self.config.capabilities.partial_archive:
            return
        if any(
            ref.get("candidate_id") == proposal.candidate_id
            for ref in state.archive_refs
        ):
            return
        archive_dir = self.store.archive_root / proposal.candidate_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        diff_path = self.store.iteration_dir(iteration) / "candidate.diff"
        if diff_path.is_file():
            shutil.copy2(diff_path, archive_dir / "candidate.diff")
        write_json(
            archive_dir / "record.json",
            {
                "candidate_id": proposal.candidate_id,
                "iteration": iteration,
                "kind": archive_kind(evidence, self.config.decision),
                "reason": decision.reason,
                "search_delta": evidence.search_delta,
                "contract_passed": evidence.contract_passed,
                "preservation_passed": evidence.preservation_passed,
                "packet_id": evidence.packet_id,
                "primary_family": (
                    evidence.primary_family.value if evidence.primary_family else ""
                ),
                "mechanism_claim": proposal.mechanism_claim,
                "predicted_effect": proposal.predicted_effect,
                "unit_failure_reasons": list(
                    evidence.metadata.get("candidate_contract_reasons") or []
                ),
            },
        )
        state.archive_refs.append(
            {"candidate_id": proposal.candidate_id, "path": str(archive_dir)}
        )
        self.store.append_event(
            "candidate_archived",
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            kind=archive_kind(evidence, self.config.decision),
            search_delta=evidence.search_delta,
        )

    def _stage_mismatch(
        self,
        *,
        iteration: int,
        proposal: CandidateProposal,
        packet: TestPacket,
        packet_path: Path,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
    ) -> None:
        """Record a unit/search disagreement for the next Test Author to diagnose.

        Cell 3 (unit passed, search regressed) means the UT design deviated
        from the search distribution; cell 4 (search improved, unit failed)
        means the UT design missed the mechanism. The next Test Author reads
        this record, the frozen tests, and both search traces, then distills
        why into the world model before designing its next packet.
        """

        mismatch_dir = self.store.mismatch_root / f"iter_{iteration:03d}"
        mismatch_dir.mkdir(parents=True, exist_ok=True)
        diff_path = self.store.iteration_dir(iteration) / "candidate.diff"
        if diff_path.is_file():
            shutil.copy2(diff_path, mismatch_dir / "candidate.diff")
        packet_copy = mismatch_dir / "packet"
        if packet_path.is_dir() and not packet_copy.exists():
            shutil.copytree(packet_path, packet_copy)
        kind = (
            "search_improved_unit_failed"
            if not unit_ok(evidence, self.config.decision)
            else "unit_passed_search_regressed"
        )
        write_json(
            mismatch_dir / "mismatch.json",
            {
                "iteration": iteration,
                "candidate_id": proposal.candidate_id,
                "kind": kind,
                "reason": decision.reason,
                "decision": decision.decision.value,
                "search_delta": evidence.search_delta,
                "packet_id": packet.packet_id,
                "packet_path": str(packet_path),
                "mechanism_claim": proposal.mechanism_claim,
                "contract_passed": evidence.contract_passed,
                "preservation_passed": evidence.preservation_passed,
                "unit_failure_reasons": list(
                    evidence.metadata.get("candidate_contract_reasons") or []
                ),
                "task_flips": self._task_flips(
                    parent_id=str(evidence.metadata.get("parent_id") or ""),
                    candidate_id=proposal.candidate_id,
                ),
            },
        )
        self.store.append_event(
            "mismatch_staged",
            iteration=iteration,
            candidate_id=proposal.candidate_id,
            kind=kind,
        )

    def _task_flips(
        self, *, parent_id: str, candidate_id: str
    ) -> list[dict[str, Any]]:
        """Per-task paired outcomes between the incumbent and the candidate."""

        flips: list[dict[str, Any]] = []
        try:
            slice_id = self._plan.search.slice_id
            parent = read_json(
                self.store.evaluation_dir(parent_id, slice_id) / "evaluation.json"
            )
            candidate = read_json(
                self.store.evaluation_dir(candidate_id, slice_id) / "evaluation.json"
            )
        except (OSError, ValueError, RuntimeError):
            return flips
        parent_by_task = {
            str(item.get("task_id")): bool(item.get("passed"))
            for item in parent.get("outcomes") or []
        }
        for item in candidate.get("outcomes") or []:
            task_id = str(item.get("task_id"))
            if task_id not in parent_by_task:
                continue
            incumbent_passed = parent_by_task[task_id]
            candidate_passed = bool(item.get("passed"))
            flips.append(
                {
                    "task_id": task_id,
                    "incumbent_passed": incumbent_passed,
                    "candidate_passed": candidate_passed,
                    "flipped": incumbent_passed != candidate_passed,
                }
            )
        return flips

    def _store_packet(self, packet: TestPacket, packet_path: Path) -> FrozenPacketRef:
        relative = copy_packet_into_store(
            packet_root=self.store.packet_store_root,
            packet_bundle=packet_path,
            content_sha256=packet.content_sha256,
        )
        return FrozenPacketRef(
            packet_id=packet.packet_id,
            path=relative,
            content_sha256=packet.content_sha256,
        )

    def _skip_failed_test_design(
        self, state: RunState, iteration: int, exc: Exception
    ) -> RunState:
        self.store.append_event(
            "test_design_failed", iteration=iteration, error=str(exc)
        )
        write_json(
            self.store.iteration_dir(iteration) / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "test_design_failed",
                "error": str(exc),
            },
        )
        state.next_iteration += 1
        self.store.save_state(state)
        return state

    def _skip_failed_trace_evidence(
        self, state: RunState, iteration: int, exc: Exception
    ) -> RunState:
        self.store.append_event(
            "trace_evidence_failed", iteration=iteration, error=str(exc)
        )
        write_json(
            self.store.iteration_dir(iteration) / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "trace_evidence_failed",
                "error": str(exc),
            },
        )
        state.next_iteration += 1
        self.store.save_state(state)
        return state

    def _skip_failed_candidate_build(
        self, state: RunState, iteration: int, exc: Exception
    ) -> RunState:
        self.store.append_event(
            "candidate_build_failed", iteration=iteration, error=str(exc)
        )
        write_json(
            self.store.iteration_dir(iteration) / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "candidate_build_failed",
                "error": str(exc),
            },
        )
        state.next_iteration += 1
        self.store.save_state(state)
        return state

    def _summary(self, state: RunState) -> dict[str, Any]:
        return {
            "protocol": self.config.protocol.condition.value,
            "capabilities": asdict(self.config.capabilities),
            "run_id": state.run_id,
            "benchmark": state.benchmark,
            "benchmark_plan_sha256": self._plan.plan_sha256,
            "status": state.status,
            "iterations_completed": state.next_iteration - 1,
            "incumbent_id": state.incumbent_id,
            "incumbent_search_score": state.incumbent_search_score,
            "promoted_ids": state.promoted_ids,
            "preserved_packets": list(state.preserved_packet_refs),
            "archived_ids": state.archived_ids,
            "archive_refs": list(state.archive_refs),
            "world_model_distills": self.world_model.distill_count,
            "total_cost": state.total_cost,
            "search_cost": state.search_cost,
            "unit_test_wall_seconds": self._unit_test_wall_seconds(),
            "model_probe_calls": int(self._unit_cost_total("model_probe_calls")),
            "model_probe_tokens": int(self._unit_cost_total("model_probe_tokens")),
            "ut_world_model_path": (
                str(self.store.ut_world_model_path)
                if self.config.capabilities.online_ut_memory
                else None
            ),
            "final_evaluation": "not_opened",
        }

    def _unit_test_wall_seconds(self) -> float:
        return self._unit_cost_total("unit_test_wall_seconds")

    def _unit_cost_total(self, key: str) -> float:
        total = 0.0
        for path in (self.store.root / "iterations").glob("iter_*/evidence.json"):
            costs = dict((read_json(path).get("metadata") or {}).get("costs") or {})
            total += float(costs.get(key) or 0.0)
        return total

    @property
    def _plan(self) -> BenchmarkPlan:
        if self.benchmark_plan is None:
            raise RuntimeError("benchmark plan has not been prepared")
        return self.benchmark_plan
