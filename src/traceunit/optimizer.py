from __future__ import annotations
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping
from traceunit.agents.runner import WorkspaceAgent, build_agent
from traceunit.benchmarks import BenchmarkAdapter, build_benchmark
from traceunit.candidate import CandidateBuildError, CandidateBuilder
from traceunit.config import ProjectConfig
from traceunit.decision import DecisionPolicy
from traceunit.evaluation import CandidateEvaluator
from traceunit.io import copy_source, read_json, source_diff, write_json
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
from traceunit.ut_memory import UTMemoryLedger, UTMemoryManager
from traceunit.trace_evidence import (
    NoFailureTraces,
    TraceEvidenceError,
)


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
        self.test_author_agent = (
            supplied.get("test_author") or build_agent(config.agents.test_author)
            if config.capabilities.generated_packets
            else None
        )
        self.search_agent = supplied.get("search") or build_agent(config.agents.search)
        self.regression_author_agent = (
            supplied.get("regression_author")
            or (
                build_agent(config.agents.regression_author)
                if config.agents.regression_author.enabled
                else None
            )
            if config.capabilities.generated_packets
            else None
        )
        self.ut_critic_agent = (
            supplied.get("ut_critic")
            or (
                build_agent(config.agents.ut_critic)
                if config.agents.ut_critic.enabled
                else None
            )
            if config.capabilities.online_ut_memory
            else None
        )
        self.policy = DecisionPolicy(config.decision)
        self.ut_memory_ledger = UTMemoryLedger(self.store.ut_feedback_episodes_path)
        self.ut_memory = UTMemoryManager(
            root=self.store.memory_root,
            ledger=self.ut_memory_ledger,
            critic=self.ut_critic_agent,
            max_lessons=config.memory.max_world_model_lessons,
            world_model_path=self.store.ut_world_model_path,
        )
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
            self.ut_memory.ensure_world_model()

        while state.next_iteration <= self.config.loop.iterations:
            iteration = state.next_iteration
            try:
                state = self._run_iteration(state, iteration)
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
            except TraceEvidenceError as exc:
                state = self._skip_failed_trace_evidence(state, iteration, exc)
            except CandidateBuildError as exc:
                state = self._skip_failed_candidate_build(state, iteration, exc)
            except Exception as exc:
                state.status = "error"
                self.store.save_state(state)
                self.store.append_event(
                    "iteration_failed",
                    iteration=iteration,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

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
        )
        self.candidate_builder = CandidateBuilder(
            config=self.config,
            store=self.store,
            benchmark=self.benchmark,
            search_agent=self.search_agent,
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
            self.ut_critic_agent,
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
        ut_memory = self._freeze_memory_input(iteration_dir)
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "running",
                "incumbent": state.incumbent_id,
                "memory_version": self.ut_memory_ledger.version,
            },
        )
        packet, packet_path, reused = self.packet_author.get_or_author(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            ut_memory_path=ut_memory,
        )
        proposal, candidate_source, diff_text = self.candidate_builder.build(
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
            packet_reused=reused,
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
        self._reflect_iteration(
            iteration=iteration,
            proposal=proposal,
            packet_path=packet_path,
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
        packet = self.packet_author._verified(packet_path)
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
        self._reflect_iteration(
            iteration=iteration,
            proposal=proposal,
            packet_path=packet_path,
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

    def _reflect_iteration(
        self,
        *,
        iteration: int,
        proposal: CandidateProposal,
        packet_path: Path,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
    ) -> None:
        if not self.config.capabilities.online_ut_memory or evidence.metadata.get(
            "violations"
        ):
            return
        episode = self.ut_memory.reflect_iteration(
            iteration=iteration,
            proposal=proposal,
            packet_path=packet_path,
            evidence=evidence,
            decision=decision,
        )
        if episode is not None:
            self.store.append_event(
                "ut_memory_updated",
                iteration=iteration,
                candidate_id=proposal.candidate_id,
                memory_version=self.ut_memory_ledger.version,
                search_outcome=episode.search_outcome.value,
                assessment=episode.assessment.value,
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
            self._promote_realized_latent(state, evidence)
            PacketAuthor.retire_active(state)
        elif decision.decision is Decision.ARCHIVE:
            if not self.config.capabilities.partial_archive:
                raise RuntimeError("archive decision reached with archive disabled")
            ref = self._retain_latent_packet(
                packet=packet,
                packet_path=packet_path,
                parent_source=Path(str(evidence.metadata["parent_source"])),
                candidate_source=candidate_source,
            )
            if ref.to_dict() not in state.latent_packet_refs:
                state.latent_packet_refs.append(ref.to_dict())
            self.store.append_event(
                "latent_packet_retained",
                candidate_id=proposal.candidate_id,
                packet_id=ref.packet_id,
                content_sha256=ref.content_sha256,
            )
            PacketAuthor.retire_active(state)
        elif decision.decision is Decision.PARTIAL_ELIGIBLE:
            if proposal.candidate_id not in state.partial_eligible_ids:
                state.partial_eligible_ids.append(proposal.candidate_id)
            PacketAuthor.retire_active(state)
        elif decision.decision is Decision.QUARANTINE:
            if proposal.candidate_id not in state.quarantined_ids:
                state.quarantined_ids.append(proposal.candidate_id)
            PacketAuthor.retire_active(state)
        else:
            state.active_packet_uses += 1
            if state.active_packet_uses >= self.config.loop.max_attempts_per_packet:
                PacketAuthor.retire_active(state)
        state.committed_iterations.append(iteration)
        self.store.save_state(state)
        return state

    def _retain_latent_packet(
        self,
        *,
        packet: TestPacket,
        packet_path: Path,
        parent_source: Path,
        candidate_source: Path,
    ) -> FrozenPacketRef:
        """Keep the frozen packet as a latent capability with a reference patch."""

        ref = self._store_packet(packet, packet_path)
        patch_path = self.store.latent_root / ref.content_sha256 / "component.patch"
        if not patch_path.is_file():
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(
                source_diff(parent_source, candidate_source), encoding="utf-8"
            )
        return ref

    def _promote_realized_latent(
        self, state: RunState, evidence: EvidenceRecord
    ) -> None:
        """Migrate latent packets the promoted candidate satisfied into preservation."""

        realized = set(evidence.realized_latent)
        still_latent: list[dict[str, str]] = []
        for raw_ref in state.latent_packet_refs:
            if raw_ref.get("content_sha256") in realized:
                if raw_ref not in state.preserved_packet_refs:
                    state.preserved_packet_refs.append(raw_ref)
                self.store.append_event(
                    "latent_packet_realized",
                    candidate_id=state.incumbent_id,
                    packet_id=raw_ref.get("packet_id"),
                    content_sha256=raw_ref.get("content_sha256"),
                )
            else:
                still_latent.append(raw_ref)
        state.latent_packet_refs = still_latent

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

    def _freeze_memory_input(self, iteration_dir: Path) -> Path | None:
        if not self.config.capabilities.online_ut_memory:
            return None
        if not self.store.ut_world_model_path.is_file():
            self.ut_memory.ensure_world_model()
        test_path = iteration_dir / "inputs" / "ut_design_world_model.md"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.store.ut_world_model_path, test_path)
        return test_path

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
        PacketAuthor.retire_active(state)
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
        PacketAuthor.retire_active(state)
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
            "latent_packets": list(state.latent_packet_refs),
            "preserved_packets": list(state.preserved_packet_refs),
            "partial_eligible_ids": state.partial_eligible_ids,
            "quarantined_ids": state.quarantined_ids,
            "ut_memory_version": self.ut_memory_ledger.version,
            "ut_feedback_episodes": len(self.ut_memory_ledger.episodes),
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
