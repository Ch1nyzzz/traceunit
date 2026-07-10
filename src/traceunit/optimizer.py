from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from traceunit.agents.runner import WorkspaceAgent, build_agent
from traceunit.archive import (
    ArchiveCatalog,
    ArchiveKind,
    ArchiveManifest,
    CompositionPlan,
    FrozenPacketRef,
    LocalCertificate,
)
from traceunit.benchmarks import BenchmarkAdapter, build_benchmark
from traceunit.calibration import AlignmentCalibrator
from traceunit.candidate import CandidateBuilder
from traceunit.composition import copy_packet_into_archive
from traceunit.config import ProjectConfig
from traceunit.decision import DecisionPolicy
from traceunit.evaluation import CandidateEvaluator
from traceunit.io import (
    append_jsonl,
    copy_source,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_tree,
    source_diff,
    write_json,
)
from traceunit.models import (
    BenchmarkPlan,
    CandidateProposal,
    Decision,
    DecisionRecord,
    EvidenceRecord,
    RunState,
    TestPacket,
)
from traceunit.packets import PacketAuthor, TestDesignFailure
from traceunit.protocol import (
    AlignmentCheckpointRunner,
    CalibrationCheckpoint,
    CalibrationSubject,
    RotationScheduler,
    decision_file_hash,
)
from traceunit.score_only import ScoreOnlyCandidateBuilder, ScoreOnlyEvaluator
from traceunit.store import RunStore
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
        self.policy = DecisionPolicy(config.decision)
        self.calibrator = AlignmentCalibrator(self.store.calibration_observations_path)
        self.checkpoints = AlignmentCheckpointRunner(
            root=self.store.calibration_root,
            config=config.alignment,
            calibrator=self.calibrator,
        )
        self.scheduler = RotationScheduler(
            config=config.alignment,
            calibrator=self.calibrator,
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
        if self.config.capabilities.delayed_alignment:
            self.checkpoints.write_public_cards(self.store.calibration_cards_path)

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
        test_cards, search_cards = self._freeze_card_inputs(iteration_dir)
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "running",
                "incumbent": state.incumbent_id,
                "card_version": self.calibrator.version,
            },
        )
        packet, packet_path, reused = self.packet_author.get_or_author(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            alignment_cards_path=test_cards,
        )
        proposal, composition, candidate_source, diff_text = (
            self.candidate_builder.build(
                state=state,
                iteration=iteration,
                iteration_dir=iteration_dir,
                packet=packet,
                packet_path=packet_path,
                alignment_cards_path=search_cards,
            )
        )
        write_json(iteration_dir / "candidate_proposal.json", proposal.to_dict())
        (iteration_dir / "candidate.diff").write_text(diff_text, encoding="utf-8")
        catalog = self.candidate_builder.catalog()
        evidence, decision = self.evaluator.evaluate_candidate(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            proposal=proposal,
            composition=composition,
            catalog=catalog,
            packet=packet,
            packet_path=packet_path,
            candidate_source=candidate_source,
            diff_text=diff_text,
            packet_reused=reused,
        )
        decision = replace(decision, evidence=evidence)
        write_json(iteration_dir / "evidence.json", evidence.to_dict())
        decision_path = iteration_dir / "decision.json"
        write_json(decision_path, decision.to_dict())

        state = self._commit_decision(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            proposal=proposal,
            composition=composition,
            packet=packet,
            packet_path=packet_path,
            candidate_source=candidate_source,
            evidence=evidence,
            decision=decision,
        )
        if self.config.capabilities.delayed_alignment and not evidence.metadata.get(
            "violations"
        ):
            self._enqueue_calibration_subject(
                state=state,
                proposal=proposal,
                composition=composition,
                evidence=evidence,
                decision_path=decision_path,
            )
        self._maybe_run_calibration(state=state, iteration=iteration)
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
                state.incumbent_search_score += float(evidence.search_delta or 0.0)
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
        iteration_dir: Path,
        proposal: CandidateProposal,
        composition: CompositionPlan,
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
            state.incumbent_search_score += float(evidence.search_delta or 0.0)
            if proposal.candidate_id not in state.promoted_ids:
                state.promoted_ids.append(proposal.candidate_id)
            ref = self._archive_packet(packet, packet_path)
            if ref.to_dict() not in state.preserved_packet_refs:
                state.preserved_packet_refs.append(ref.to_dict())
            PacketAuthor.retire_active(state)
        elif decision.decision is Decision.ARCHIVE:
            if not self.config.capabilities.partial_archive:
                raise RuntimeError("archive decision reached with archive disabled")
            manifest = self._archive_component(
                proposal=proposal,
                composition=composition,
                packet=packet,
                packet_path=packet_path,
                parent_source=Path(str(evidence.metadata["parent_source"])),
                candidate_source=candidate_source,
                evidence=evidence,
                iteration_dir=iteration_dir,
            )
            if manifest.archive_id not in state.archive_ids:
                state.archive_ids.append(manifest.archive_id)
            PacketAuthor.retire_active(state)
        elif decision.decision is Decision.PARTIAL_ELIGIBLE:
            if proposal.candidate_id not in state.partial_eligible_ids:
                state.partial_eligible_ids.append(proposal.candidate_id)
            PacketAuthor.retire_active(state)
        elif decision.decision is Decision.QUARANTINE:
            if proposal.candidate_id not in state.quarantined_ids:
                state.quarantined_ids.append(proposal.candidate_id)
            PacketAuthor.retire_active(state)
        elif decision.decision is Decision.CHALLENGE_PACKET:
            if packet.packet_id not in state.challenged_packet_ids:
                state.challenged_packet_ids.append(packet.packet_id)
            PacketAuthor.retire_active(state)
        else:
            state.active_packet_uses += 1
            if state.active_packet_uses >= self.config.loop.max_attempts_per_packet:
                PacketAuthor.retire_active(state)
        state.committed_iterations.append(iteration)
        self.store.save_state(state)
        return state

    def _archive_component(
        self,
        *,
        proposal: CandidateProposal,
        composition: CompositionPlan,
        packet: TestPacket,
        packet_path: Path,
        parent_source: Path,
        candidate_source: Path,
        evidence: EvidenceRecord,
        iteration_dir: Path,
    ) -> ArchiveManifest:
        packet_ref = self._archive_packet(packet, packet_path)
        patch_text = source_diff(parent_source, candidate_source)
        staging = self.store.component_archive_root / ".staging" / proposal.candidate_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        patch_path = staging / "component.patch"
        patch_path.write_text(patch_text, encoding="utf-8")
        certificate = LocalCertificate(
            frozen_packet_refs=(packet_ref,),
            family_keys=tuple(str(item) for item in evidence.metadata["family_keys"]),
            public_passed=(
                evidence.public_gain >= self.config.decision.min_public_gain
            ),
            hidden_passed=(
                evidence.hidden_gain >= self.config.decision.min_hidden_gain
            ),
            bridge_passed=(
                evidence.bridge_gain >= self.config.decision.min_bridge_gain
            ),
            regression_passed=(
                evidence.regression_loss <= self.config.decision.max_regression_loss
            ),
            public_score=evidence.public_gain,
            hidden_score=evidence.hidden_gain,
            bridge_score=evidence.bridge_gain,
            regression_loss=evidence.regression_loss,
        )
        kind = (
            ArchiveKind.COMPOSITE if composition.component_ids else ArchiveKind.ATOMIC
        )
        target_hypothesis = next(
            hypothesis
            for hypothesis in packet.hypotheses
            if hypothesis.hypothesis_id == packet.target_hypothesis_id
        )
        manifest = ArchiveManifest(
            kind=kind,
            parent_source_sha256=sha256_tree(parent_source),
            candidate_source_sha256=sha256_tree(candidate_source),
            patch_path=f".staging/{proposal.candidate_id}/component.patch",
            patch_sha256=sha256_file(patch_path),
            certificate=certificate,
            mechanism=proposal.mechanism_claim,
            target_boundary=target_hypothesis.target_boundary,
            constituents=composition.component_ids,
            trace_signature=packet.source_trace_ids,
            applicability=proposal.regression_risks,
        )
        component_dir = self.store.component_archive_root / manifest.archive_id
        manifest = replace(
            manifest,
            patch_path=f"{manifest.archive_id}/component.patch",
        )
        if component_dir.exists():
            existing = ArchiveManifest.from_dict(
                read_json(component_dir / "manifest.json")
            )
            existing_patch = component_dir / "component.patch"
            if (
                existing.identity_dict() != manifest.identity_dict()
                or not existing_patch.is_file()
                or sha256_file(existing_patch) != manifest.patch_sha256
            ):
                raise RuntimeError(
                    f"content-addressed archive collision: {manifest.archive_id}"
                )
            shutil.rmtree(staging)
            manifest = existing
        else:
            write_json(staging / "manifest.json", manifest.to_dict())
            staging.rename(component_dir)
        usage_path = self.store.component_archive_root / "usage.jsonl"
        if not any(
            row.get("candidate_id") == proposal.candidate_id
            and row.get("component_id") == manifest.archive_id
            for row in read_jsonl(usage_path)
        ):
            append_jsonl(
                usage_path,
                {
                    "candidate_id": proposal.candidate_id,
                    "component_id": manifest.archive_id,
                    "attempt_fingerprint": composition.attempt_fingerprint,
                    "status": "archived",
                },
            )
        ArchiveCatalog.load(self.store.component_archive_root)
        self.store.append_event(
            "component_archived",
            candidate_id=proposal.candidate_id,
            component_id=manifest.archive_id,
        )
        return manifest

    def _archive_packet(self, packet: TestPacket, packet_path: Path) -> FrozenPacketRef:
        relative = copy_packet_into_archive(
            archive_root=self.store.packet_store_root,
            packet_bundle=packet_path,
            content_sha256=packet.content_sha256,
        )
        return FrozenPacketRef(
            packet_id=packet.packet_id,
            path=relative,
            content_sha256=packet.content_sha256,
        )

    def _enqueue_calibration_subject(
        self,
        *,
        state: RunState,
        proposal: CandidateProposal,
        composition: CompositionPlan,
        evidence: EvidenceRecord,
        decision_path: Path,
    ) -> None:
        subject = CalibrationSubject(
            candidate_id=proposal.candidate_id,
            parent_id=str(evidence.metadata["parent_id"]),
            lineage_id=state.run_id,
            candidate_source=str(evidence.metadata["candidate_source"]),
            parent_source=str(evidence.metadata["parent_source"]),
            candidate_source_sha256=sha256_tree(
                Path(str(evidence.metadata["candidate_source"]))
            ),
            parent_source_sha256=sha256_tree(
                Path(str(evidence.metadata["parent_source"]))
            ),
            decision_path=str(decision_path.resolve()),
            decision_sha256=decision_file_hash(decision_path),
            family_keys=tuple(
                str(item) for item in evidence.metadata.get("family_keys") or []
            ),
            unit_profile=_unit_profile(evidence, self.config),
            stratum=_calibration_stratum(evidence),
            composition_signature=(
                composition.attempt_fingerprint if composition.component_ids else ""
            ),
        )
        path = self.store.calibration_root / "pending" / f"{proposal.candidate_id}.json"
        if path.is_file() and read_json(path) != subject.to_dict():
            raise RuntimeError(
                f"pending calibration subject changed: {proposal.candidate_id}"
            )
        write_json(path, subject.to_dict())
        if proposal.candidate_id not in state.pending_calibration_ids:
            state.pending_calibration_ids.append(proposal.candidate_id)
        if not any(
            row.get("candidate_id") == proposal.candidate_id
            for row in read_jsonl(self.store.calibration_queue_path)
        ):
            append_jsonl(self.store.calibration_queue_path, subject.to_dict())
        self.store.save_state(state)

    def _maybe_run_calibration(self, *, state: RunState, iteration: int) -> None:
        assert self.evaluator is not None
        if not self.config.capabilities.delayed_alignment:
            return
        self._reconcile_completed_checkpoints(state)
        reserved = self.checkpoints.reserved_checkpoint()
        if reserved is not None:
            checkpoint = reserved
            reasons = ("resume_reserved_checkpoint",)
        else:
            pending = tuple(
                CalibrationSubject.from_dict(
                    read_json(
                        self.store.calibration_root / "pending" / f"{candidate_id}.json"
                    )
                )
                for candidate_id in state.pending_calibration_ids
            )
            available = self.checkpoints.available_shards(self._plan.calibration)
            rotation = self.scheduler.decide(
                pending=pending,
                available_shards=available,
                known_composition_signatures=self._known_composition_signatures(),
            )
            if not rotation.open_checkpoint or rotation.shard is None:
                return
            checkpoint = self.checkpoints.freeze(
                iteration=iteration,
                shard=rotation.shard,
                subjects=pending,
            )
            reasons = rotation.reasons
        self.checkpoints.run(
            checkpoint,
            evaluate=lambda source, candidate_id, pool, tag: (
                self.evaluator.evaluate_pool(
                    source=source,
                    candidate_id=candidate_id,
                    pool=pool,
                    cache_tag=tag,
                )
            ),
            noninferiority_margin=self.config.decision.noninferiority_margin,
            positive_effect=self.config.alignment.positive_margin,
        )
        self._apply_checkpoint_result(state, checkpoint, reasons=reasons)

    def _reconcile_completed_checkpoints(self, state: RunState) -> None:
        for result_path in sorted(
            self.store.calibration_root.glob("checkpoints/*/result.json")
        ):
            checkpoint_id = result_path.parent.name
            if checkpoint_id in state.applied_calibration_checkpoint_ids:
                continue
            checkpoint = CalibrationCheckpoint.from_dict(
                read_json(result_path.parent / "checkpoint.json")
            )
            self._apply_checkpoint_result(
                state,
                checkpoint,
                reasons=("reconcile_completed_checkpoint",),
            )

    def _apply_checkpoint_result(
        self,
        state: RunState,
        checkpoint: CalibrationCheckpoint,
        *,
        reasons: tuple[str, ...],
    ) -> None:
        if checkpoint.checkpoint_id in state.applied_calibration_checkpoint_ids:
            return
        result_path = (
            self.store.calibration_root
            / "checkpoints"
            / checkpoint.checkpoint_id
            / "result.json"
        )
        result = read_json(result_path)
        if result.get("checkpoint_id") != checkpoint.checkpoint_id:
            raise RuntimeError("calibration result does not match its checkpoint")
        consumed = {item.candidate_id for item in checkpoint.subjects}
        state.pending_calibration_ids = [
            item for item in state.pending_calibration_ids if item not in consumed
        ]
        cost = float(result.get("cost") or 0.0)
        state.calibration_cost += cost
        state.total_cost += cost
        state.calibration_epoch = max(
            state.calibration_epoch, int(result["card_version_after"])
        )
        state.next_calibration_shard = max(
            state.next_calibration_shard, checkpoint.shard.ordinal + 1
        )
        state.applied_calibration_checkpoint_ids.append(checkpoint.checkpoint_id)
        self.checkpoints.write_public_cards(self.store.calibration_cards_path)
        self.store.save_state(state)
        self.store.append_event(
            "alignment_checkpoint_completed",
            checkpoint_id=checkpoint.checkpoint_id,
            shard_id=checkpoint.shard.slice_id,
            reasons=reasons,
            card_version=self.calibrator.version,
            cost=cost,
        )

    def _known_composition_signatures(self) -> tuple[str, ...]:
        signatures: set[str] = set()
        for result_path in self.store.calibration_root.glob(
            "checkpoints/*/result.json"
        ):
            checkpoint = CalibrationCheckpoint.from_dict(
                read_json(result_path.parent / "checkpoint.json")
            )
            signatures.update(
                subject.composition_signature
                for subject in checkpoint.subjects
                if subject.composition_signature
            )
        return tuple(sorted(signatures))

    def _freeze_card_inputs(
        self, iteration_dir: Path
    ) -> tuple[Path | None, Path | None]:
        if not self.config.capabilities.delayed_alignment:
            return None, None
        if (
            self.config.capabilities.delayed_alignment
            and not self.store.calibration_cards_path.is_file()
        ):
            self.checkpoints.write_public_cards(self.store.calibration_cards_path)
        payload = read_json(self.store.calibration_cards_path)
        test_path = iteration_dir / "inputs" / "test_author_cards.json"
        search_path = iteration_dir / "inputs" / "search_cards.json"
        write_json(test_path, {**payload, "audience": "test_author"})
        write_json(search_path, {**payload, "audience": "search"})
        return test_path, search_path

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
            "archive_ids": state.archive_ids,
            "partial_eligible_ids": state.partial_eligible_ids,
            "quarantined_ids": state.quarantined_ids,
            "challenged_packet_ids": state.challenged_packet_ids,
            "alignment_version": self.calibrator.version,
            "alignment_observations": len(self.calibrator.observations),
            "pending_calibration_ids": state.pending_calibration_ids,
            "total_cost": state.total_cost,
            "search_cost": state.search_cost,
            "calibration_cost": state.calibration_cost,
            "unit_test_wall_seconds": self._unit_test_wall_seconds(),
            "calibration_cards_path": (
                str(self.store.calibration_cards_path)
                if self.config.capabilities.delayed_alignment
                else None
            ),
            "final_evaluation": "not_opened",
        }

    def _unit_test_wall_seconds(self) -> float:
        total = 0.0
        for path in (self.store.root / "iterations").glob("iter_*/evidence.json"):
            costs = dict((read_json(path).get("metadata") or {}).get("costs") or {})
            total += float(costs.get("unit_test_wall_seconds") or 0.0)
        return total

    @property
    def _plan(self) -> BenchmarkPlan:
        if self.benchmark_plan is None:
            raise RuntimeError("benchmark plan has not been prepared")
        return self.benchmark_plan


def _unit_profile(evidence: EvidenceRecord, config: ProjectConfig) -> str:
    unit = (
        "unit+"
        if evidence.public_gain >= config.decision.min_public_gain
        and evidence.hidden_gain >= config.decision.min_hidden_gain
        else "unit-"
    )
    bridge = (
        "bridge+"
        if evidence.metadata.get("has_bridge")
        and evidence.bridge_gain >= config.decision.min_bridge_gain
        else "bridge0"
    )
    regression = (
        "regression+"
        if evidence.regression_loss <= config.decision.max_regression_loss
        else "regression-"
    )
    return f"{unit}|{bridge}|{regression}"


def _calibration_stratum(evidence: EvidenceRecord) -> str:
    if evidence.search_delta is None:
        search = "search?"
    elif evidence.search_delta > 0:
        search = "search+"
    elif evidence.search_delta == 0:
        search = "search0"
    else:
        search = "search-"
    composition = (
        "composition+" if evidence.metadata.get("composition_ids") else "composition0"
    )
    return f"{search}|{composition}"
