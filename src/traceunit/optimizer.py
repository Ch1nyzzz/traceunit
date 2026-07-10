from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from traceunit.agents.prompts import (
    auditor_prompt,
    experimentalist_prompt,
    optimizer_prompt,
    public_packet,
)
from traceunit.agents.runner import WorkspaceAgent, build_agent
from traceunit.benchmarks import BenchmarkAdapter, build_benchmark
from traceunit.calibration import CrossLayerCalibrator
from traceunit.config import ProjectConfig
from traceunit.decision import DecisionPolicy
from traceunit.io import (
    copy_source,
    read_json,
    read_jsonl,
    source_diff,
    write_json,
)
from traceunit.models import (
    BenchmarkEvaluation,
    CandidateProposal,
    Decision,
    DecisionRecord,
    EvidenceRecord,
    RunState,
    TestPacket,
    TestStatus,
    TestTier,
)
from traceunit.store import RunStore
from traceunit.tests_runtime import (
    InvalidTestPacket,
    admission_score,
    freeze_test_packet,
    load_test_packet,
    paired_test_metrics,
    run_test_cases,
    verify_frozen_packet,
)


class TestDesignFailure(RuntimeError):
    pass


class NoFailureTraces(RuntimeError):
    pass


class OptimizationLoop:
    """Trace -> frozen tests -> candidate -> paired evidence -> four-way decision."""

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
        self.experimentalist = supplied.get("experimentalist") or build_agent(
            config.agents.experimentalist
        )
        self.optimizer_agent = supplied.get("optimizer") or build_agent(
            config.agents.optimizer
        )
        self.auditor_agent = supplied.get("auditor") or (
            build_agent(config.agents.auditor)
            if config.agents.auditor.enabled
            else None
        )
        self.policy = DecisionPolicy(config.decision)
        self.calibrator = CrossLayerCalibrator(self.store.calibration_path)

    def run(self) -> dict[str, Any]:
        self.store.initialize(config_snapshot=asdict(self.config))
        for agent in (
            self.experimentalist,
            self.optimizer_agent,
            self.auditor_agent,
        ):
            preflight = getattr(agent, "preflight", None)
            if callable(preflight):
                preflight()
        self.benchmark.prepare(self.store.root)
        self.benchmark.preflight()
        state = self.store.load_state()
        if state is not None and not self.config.loop.resume:
            raise RuntimeError(
                f"run state already exists at {self.store.state_path}; enable resume "
                "or choose a new loop.run_dir"
            )
        if state is None:
            state = self._initialize_baseline()
        state.status = "running"
        self.store.save_state(state)

        while state.next_iteration <= self.config.loop.iterations:
            iteration = state.next_iteration
            try:
                state = self._run_iteration(state, iteration)
            except NoFailureTraces:
                state.status = "converged"
                self.store.append_event(
                    "run_converged",
                    iteration=iteration,
                    reason="incumbent has no failed diagnostic traces",
                )
                break
            except TestDesignFailure as exc:
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
                state.active_packet_id = ""
                state.active_packet_path = ""
                state.active_packet_uses = 0
                state.next_iteration += 1
                self.store.save_state(state)
                continue
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
        posthoc = {}
        if (
            self.config.loop.posthoc_audit
            and not self.config.decision.require_audit_for_promotion
        ):
            posthoc = self._run_posthoc_audit(state)
        summary = {
            "run_id": state.run_id,
            "benchmark": state.benchmark,
            "status": state.status,
            "iterations_completed": state.next_iteration - 1,
            "incumbent_id": state.incumbent_id,
            "incumbent_diagnostic_score": state.incumbent_diagnostic_score,
            "incumbent_canary_score": state.incumbent_canary_score,
            "promoted_ids": state.promoted_ids,
            "partial_archive_ids": state.partial_archive_ids,
            "challenged_packet_ids": state.challenged_packet_ids,
            "total_cost": state.total_cost,
            "search_natural_task_tokens": state.total_cost,
            "unit_test_wall_seconds": self._unit_test_wall_seconds(),
            "calibration_path": str(self.store.calibration_path),
            **posthoc,
        }
        write_json(self.store.root / "summary.json", summary)
        return summary

    def _initialize_baseline(self) -> RunState:
        seed_id = "seed"
        seed_dir = self.store.candidate_dir(seed_id)
        source = seed_dir / "source"
        if not source.exists():
            copy_source(self.benchmark.seed_source(), source)
        diagnostic = self._evaluate(
            source=source, candidate_id=seed_id, split="diagnostic"
        )
        canary = self._evaluate(source=source, candidate_id=seed_id, split="canary")
        run_id = self.config.loop.run_id or self.store.root.name
        state = RunState(
            run_id=run_id,
            benchmark=self.benchmark.name,
            status="running",
            next_iteration=1,
            incumbent_id=seed_id,
            incumbent_source=str(source.resolve()),
            incumbent_diagnostic_score=diagnostic.score,
            incumbent_canary_score=canary.score,
            promoted_ids=[seed_id],
            total_cost=diagnostic.cost + canary.cost,
        )
        self.store.save_state(state)
        self.store.append_event(
            "baseline_collected",
            candidate_id=seed_id,
            diagnostic_score=diagnostic.score,
            canary_score=canary.score,
            diagnostic_trace_path=diagnostic.trace_path,
        )
        return state

    def _evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        split: str,
        limit_override: int | None = None,
        cache_tag: str = "",
    ) -> BenchmarkEvaluation:
        storage_id = candidate_id if not cache_tag else f"{candidate_id}__{cache_tag}"
        return self.benchmark.evaluate(
            source=source,
            candidate_id=storage_id,
            split=split,
            out_dir=self.store.evaluation_dir(storage_id, split),
            limit_override=limit_override,
        )

    def _get_packet(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
    ) -> tuple[TestPacket, Path, bool]:
        packet_ref = iteration_dir / "packet_ref.json"
        if packet_ref.is_file():
            path = Path(str(read_json(packet_ref)["path"]))
            packet = load_test_packet(path)
            if not verify_frozen_packet(path, packet):
                raise TestDesignFailure(f"frozen TestPacket hash mismatch: {path}")
            return packet, path, bool(read_json(packet_ref).get("reused"))
        if state.active_packet_path:
            path = Path(state.active_packet_path)
            packet = load_test_packet(path)
            if packet.status == TestStatus.ADMITTED and verify_frozen_packet(
                path, packet
            ):
                write_json(
                    packet_ref,
                    {"path": str(path), "packet_id": packet.packet_id, "reused": True},
                )
                return packet, path, True
            state.active_packet_id = ""
            state.active_packet_path = ""
            state.active_packet_uses = 0

        packet, path = self._author_test_packet(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
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

    def _author_test_packet(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
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
                self._stage_trace_evidence(
                    state=state,
                    destination=workspace / "trace_evidence",
                )
            prompt = experimentalist_prompt(
                benchmark_context=self.benchmark.context(),
                trace_manifest=trace_manifest,
                incumbent_source=incumbent_copy,
                output_dir=output,
            )
            if feedback:
                prompt += (
                    "\n\nThe previous packet was rejected by mechanical admission. "
                    "Create a new packet rather than weakening expectations. Reasons:\n"
                    + feedback
                )
            if not (output / "test_packet.json").is_file():
                run = self.experimentalist.run(
                    role="experimentalist",
                    prompt=prompt,
                    workspace=workspace,
                    log_dir=iteration_dir
                    / "test_author"
                    / f"attempt_{attempt}"
                    / "agent",
                )
                if run.returncode != 0 or run.timed_out:
                    feedback = (
                        f"agent process failed: returncode={run.returncode}, "
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
            library_name = (
                f"{_safe_name(packet.packet_id)}_v{packet.version}_"
                f"{packet.content_sha256[:12]}"
            )
            library_path = self.store.root / "test_library" / library_name
            if not library_path.exists():
                shutil.copytree(output, library_path)
            frozen = load_test_packet(library_path)
            if not verify_frozen_packet(library_path, frozen):
                raise TestDesignFailure("TestPacket changed while entering the library")
            self.store.append_event(
                "test_packet_admitted",
                iteration=iteration,
                packet_id=frozen.packet_id,
                admission_score=frozen.admission_score,
                path=str(library_path),
            )
            return frozen, library_path
        raise TestDesignFailure(feedback or "Test Author produced no admissible packet")

    def _stage_trace_evidence(self, *, state: RunState, destination: Path) -> None:
        evaluation = self._load_evaluation(state.incumbent_id, "diagnostic")
        rows = read_jsonl(Path(evaluation.trace_path))
        failed = [
            row
            for row in rows
            if not bool(row.get("passed"))
            and str(row.get("status") or "ok") in {"ok", "unresolved"}
        ]
        failed.sort(
            key=lambda row: (float(row.get("score") or 0.0), str(row.get("task_id")))
        )
        if not failed:
            invalid = [row for row in rows if not bool(row.get("passed"))]
            if invalid:
                statuses = sorted(
                    {str(row.get("status") or "unknown") for row in invalid}
                )
                raise TestDesignFailure(
                    "diagnostic pool has failures but none are valid behavioral traces; "
                    f"statuses={statuses}"
                )
            raise NoFailureTraces
        successful = [row for row in rows if bool(row.get("passed"))][:2]
        selected = failed[: self.config.loop.max_failure_traces] + successful
        destination.mkdir(parents=True, exist_ok=True)
        staged: list[dict[str, Any]] = []
        for row in selected:
            copied = dict(row)
            digest = hashlib.sha256(str(row.get("trace_id")).encode()).hexdigest()[:16]
            trace_dir = destination / "artifacts" / digest
            trace_dir.mkdir(parents=True, exist_ok=True)
            staged_paths: list[str] = []
            for index, raw_path in enumerate(row.get("artifact_paths") or []):
                source = Path(str(raw_path))
                if not source.is_file():
                    continue
                target = trace_dir / f"{index:02d}_{source.name}"
                shutil.copy2(source, target)
                staged_paths.append(str(target.relative_to(destination)))
            copied["artifact_paths"] = staged_paths
            staged_events: list[dict[str, Any]] = []
            artifact_index = 0
            for event in copied.get("events") or []:
                if not isinstance(event, Mapping):
                    continue
                staged_event = dict(event)
                if event.get("kind") == "artifact":
                    staged_event["input"] = (
                        {"staged_artifact": staged_paths[artifact_index]}
                        if artifact_index < len(staged_paths)
                        else {"staged_artifact": None}
                    )
                    artifact_index += 1
                staged_events.append(staged_event)
            copied["events"] = staged_events
            metrics = dict(copied.get("metrics") or {})
            metrics.pop("task_dump", None)
            copied["metrics"] = metrics
            staged.append(copied)
        write_json(destination / "manifest.json", {"traces": staged})

    def _load_evaluation(self, candidate_id: str, split: str) -> BenchmarkEvaluation:
        path = self.store.evaluation_dir(candidate_id, split) / "evaluation.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing cached evaluation: {path}")
        return BenchmarkEvaluation.from_dict(read_json(path))

    def _run_iteration(self, state: RunState, iteration: int) -> RunState:
        iteration_dir = self.store.iteration_dir(iteration)
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "running",
                "incumbent": state.incumbent_id,
            },
        )
        packet, packet_path, reused = self._get_packet(
            state=state, iteration=iteration, iteration_dir=iteration_dir
        )
        proposal, candidate_source, diff_text = self._propose_candidate(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            packet=packet,
            packet_path=packet_path,
        )
        write_json(iteration_dir / "candidate_proposal.json", proposal.to_dict())
        (iteration_dir / "candidate.diff").write_text(diff_text, encoding="utf-8")

        candidate_id = proposal.candidate_id
        smoke_ok, smoke_message = self.benchmark.smoke_test(
            candidate_source, iteration_dir / "smoke"
        )
        violations = [
            *self.benchmark.policy_violations(candidate_source, diff_text),
            *_external_symlink_violations(candidate_source),
        ]
        if not diff_text.strip():
            violations.append("candidate source is identical to the incumbent")
        if not smoke_ok:
            violations.append(f"candidate smoke check failed: {smoke_message[-1000:]}")
        if violations:
            evidence = EvidenceRecord(
                iteration=iteration,
                candidate_id=candidate_id,
                packet_id=packet.packet_id,
                public_gain=0.0,
                hidden_gain=0.0,
                bridge_gain=0.0,
                regression_loss=1.0,
                admission_score=packet.admission_score,
                metadata={
                    "violations": violations,
                    "parent_id": state.incumbent_id,
                    "packet_reused": reused,
                    "has_bridge": any(
                        case.tier == TestTier.BRIDGE for case in packet.cases
                    ),
                },
            )
            decision = DecisionRecord(
                iteration=iteration,
                candidate_id=candidate_id,
                decision=Decision.REJECT,
                reason="; ".join(violations),
                confidence=1.0,
                evidence=evidence,
            )
            return self._finalize_iteration(
                state=state,
                iteration=iteration,
                iteration_dir=iteration_dir,
                packet=packet,
                candidate_source=candidate_source,
                evidence=evidence,
                decision=decision,
            )

        pair_dir = iteration_dir / "paired_tests"
        incumbent_results = run_test_cases(
            packet=packet,
            bundle=packet_path,
            source=Path(state.incumbent_source),
            subject="incumbent",
            output_dir=pair_dir / "incumbent",
            python=self.config.benchmark.unit_python,
        )
        candidate_results = run_test_cases(
            packet=packet,
            bundle=packet_path,
            source=candidate_source,
            subject="candidate",
            output_dir=pair_dir / "candidate",
            python=self.config.benchmark.unit_python,
        )
        metrics = paired_test_metrics(packet, incumbent_results, candidate_results)
        auditor_regression = self._run_auditor(
            iteration=iteration,
            iteration_dir=iteration_dir,
            state=state,
            proposal=proposal,
            candidate_source=candidate_source,
            diff_text=diff_text,
        )
        metrics["regression_loss"] = max(metrics["regression_loss"], auditor_regression)
        unit_cost = sum(
            result.duration_s for result in [*incumbent_results, *candidate_results]
        )
        evidence = EvidenceRecord(
            iteration=iteration,
            candidate_id=candidate_id,
            packet_id=packet.packet_id,
            public_gain=metrics["public_gain"],
            hidden_gain=metrics["hidden_gain"],
            bridge_gain=metrics["bridge_gain"],
            regression_loss=metrics["regression_loss"],
            admission_score=packet.admission_score,
            total_cost=0.0,
            metadata={
                "parent_id": state.incumbent_id,
                "costs": {
                    "unit_test_wall_seconds": unit_cost,
                    "natural_task_tokens": 0.0,
                },
                "packet_reused": reused,
                "has_bridge": any(
                    case.tier == TestTier.BRIDGE for case in packet.cases
                ),
                "incumbent_test_results": [
                    result.to_dict() for result in incumbent_results
                ],
                "candidate_test_results": [
                    result.to_dict() for result in candidate_results
                ],
            },
        )
        decision = self.policy.decide(evidence)

        unit_failed = (
            evidence.public_gain < self.config.decision.min_public_gain
            or evidence.hidden_gain < self.config.decision.min_hidden_gain
        )
        if (
            decision.decision == Decision.REJECT
            and unit_failed
            and evidence.regression_loss <= self.config.decision.max_regression_loss
            and self._should_challenge_rejection(iteration)
        ):
            challenge = self._evaluate(
                source=candidate_source,
                candidate_id=candidate_id,
                split="diagnostic",
                limit_override=2,
                cache_tag="test_challenge",
            )
            incumbent = self._load_evaluation(state.incumbent_id, "diagnostic")
            challenge_delta = _matched_delta(incumbent, challenge)
            evidence = replace(
                evidence,
                diagnostic_delta=challenge_delta,
                total_cost=evidence.total_cost + challenge.cost,
                metadata=_with_natural_cost(
                    evidence.metadata,
                    challenge.cost,
                    rejected_candidate_probe=True,
                ),
            )
            if challenge_delta > self.config.decision.min_diagnostic_delta:
                decision = DecisionRecord(
                    iteration=iteration,
                    candidate_id=candidate_id,
                    decision=Decision.TEST_CHALLENGE,
                    reason=(
                        "candidate failed the frozen unit packet but improved a paired "
                        "natural-task probe; the packet must be re-diagnosed"
                    ),
                    confidence=min(1.0, 0.5 + challenge_delta),
                    evidence=evidence,
                )
            else:
                decision = self.policy.decide(evidence)

        while decision.decision == Decision.ESCALATE:
            if evidence.diagnostic_delta is None:
                diagnostic = self._evaluate(
                    source=candidate_source,
                    candidate_id=candidate_id,
                    split="diagnostic",
                )
                evidence = replace(
                    evidence,
                    diagnostic_delta=diagnostic.score
                    - state.incumbent_diagnostic_score,
                    total_cost=evidence.total_cost + diagnostic.cost,
                    metadata=_with_natural_cost(evidence.metadata, diagnostic.cost),
                )
            elif evidence.canary_delta is None:
                canary = self._evaluate(
                    source=candidate_source,
                    candidate_id=candidate_id,
                    split="canary",
                )
                evidence = replace(
                    evidence,
                    canary_delta=canary.score
                    - float(state.incumbent_canary_score or 0.0),
                    total_cost=evidence.total_cost + canary.cost,
                    metadata=_with_natural_cost(evidence.metadata, canary.cost),
                )
            elif (
                self.config.decision.require_audit_for_promotion
                and evidence.audit_delta is None
            ):
                incumbent_audit = self._evaluate(
                    source=Path(state.incumbent_source),
                    candidate_id=state.incumbent_id,
                    split="audit",
                )
                candidate_audit = self._evaluate(
                    source=candidate_source,
                    candidate_id=candidate_id,
                    split="audit",
                )
                evidence = replace(
                    evidence,
                    audit_delta=candidate_audit.score - incumbent_audit.score,
                    total_cost=(
                        evidence.total_cost
                        + incumbent_audit.cost
                        + candidate_audit.cost
                    ),
                    metadata=_with_natural_cost(
                        evidence.metadata,
                        incumbent_audit.cost + candidate_audit.cost,
                    ),
                )
            else:
                raise RuntimeError(
                    "decision policy requested an unavailable escalation"
                )
            decision = self.policy.decide(evidence)

        return self._finalize_iteration(
            state=state,
            iteration=iteration,
            iteration_dir=iteration_dir,
            packet=packet,
            candidate_source=candidate_source,
            evidence=evidence,
            decision=decision,
        )

    def _run_posthoc_audit(self, state: RunState) -> dict[str, Any]:
        """Label candidates after search, without feeding held-out results back."""

        cached: dict[str, BenchmarkEvaluation] = {}

        def audit(candidate_id: str) -> BenchmarkEvaluation:
            if candidate_id not in cached:
                source = self.store.root / "candidates" / candidate_id / "source"
                if not source.is_dir():
                    raise FileNotFoundError(
                        f"missing candidate source for post-hoc audit: {source}"
                    )
                cached[candidate_id] = self._evaluate(
                    source=source,
                    candidate_id=candidate_id,
                    split="audit",
                    cache_tag="posthoc_audit",
                )
            return cached[candidate_id]

        seed_audit = audit("seed")
        records: list[dict[str, Any]] = []
        for decision_path in sorted(
            (self.store.root / "iterations").glob("iter_*/decision.json")
        ):
            raw = read_json(decision_path)
            decision = DecisionRecord.from_dict(raw)
            evidence = decision.evidence
            if evidence.metadata.get("violations"):
                records.append(
                    {
                        "iteration": decision.iteration,
                        "candidate_id": decision.candidate_id,
                        "status": "skipped_invalid_candidate",
                    }
                )
                continue
            parent_id = str(evidence.metadata.get("parent_id") or "")
            if not parent_id:
                proposal_path = decision_path.parent / "candidate_proposal.json"
                if proposal_path.is_file():
                    parent_id = str(read_json(proposal_path).get("parent_id") or "")
            if not parent_id:
                raise RuntimeError(
                    f"post-hoc audit lacks parent lineage for {decision.candidate_id}"
                )
            parent_audit = audit(parent_id)
            candidate_audit = audit(decision.candidate_id)
            audit_delta = _matched_delta(parent_audit, candidate_audit)
            audited_evidence = replace(
                evidence,
                audit_delta=audit_delta,
                metadata={
                    **evidence.metadata,
                    "audit_protocol": "posthoc_nonadaptive",
                },
            )
            audited_decision = replace(decision, evidence=audited_evidence)
            packet_ref = decision_path.parent / "packet_ref.json"
            family_ids: set[str] = set()
            if packet_ref.is_file():
                packet_path = Path(str(read_json(packet_ref).get("path") or ""))
                if packet_path.is_dir():
                    packet = load_test_packet(packet_path)
                    family_ids = {case.family_id for case in packet.cases}
            self.calibrator.update(
                evidence=audited_evidence,
                decision=audited_decision,
                family_ids=family_ids,
            )
            records.append(
                {
                    "iteration": decision.iteration,
                    "candidate_id": decision.candidate_id,
                    "parent_id": parent_id,
                    "search_decision": decision.decision.value,
                    "parent_audit_score": parent_audit.score,
                    "candidate_audit_score": candidate_audit.score,
                    "audit_delta": audit_delta,
                }
            )

        final_audit = audit(state.incumbent_id)
        payload = {
            "protocol": "posthoc_nonadaptive",
            "seed_audit_score": seed_audit.score,
            "final_incumbent_id": state.incumbent_id,
            "final_audit_score": final_audit.score,
            "final_audit_delta": _matched_delta(seed_audit, final_audit),
            "research_audit_cost": sum(item.cost for item in cached.values()),
            "records": records,
        }
        path = self.store.root / "sealed" / "posthoc_audit.json"
        write_json(path, payload)
        return {
            key: value for key, value in payload.items() if key not in {"records"}
        } | {"posthoc_audit_path": str(path)}

    def _unit_test_wall_seconds(self) -> float:
        total = 0.0
        for path in (self.store.root / "iterations").glob("iter_*/evidence.json"):
            costs = dict((read_json(path).get("metadata") or {}).get("costs") or {})
            total += float(costs.get("unit_test_wall_seconds") or 0.0)
        return total

    def _propose_candidate(
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
        if not proposal_path.is_file():
            copy_source(Path(state.incumbent_source), source)
            public_path = candidate_dir / "public_packet.json"
            write_json(public_path, public_packet(packet))
            for case in packet.cases:
                if case.tier != TestTier.PUBLIC:
                    continue
                source_test = packet_path / case.path
                target_test = candidate_dir / case.path
                target_test.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_test, target_test)
            history_path = candidate_dir / "history.json"
            write_json(history_path, self._public_history())
            prompt = optimizer_prompt(
                benchmark_context=self.benchmark.context(),
                candidate_id=candidate_id,
                parent_id=state.incumbent_id,
                source_dir=source,
                public_packet_path=public_path,
                history_path=history_path,
                proposal_path=proposal_path,
            )
            run = self.optimizer_agent.run(
                role="optimizer",
                prompt=prompt,
                workspace=candidate_dir,
                log_dir=iteration_dir / "optimizer_agent",
            )
            if run.returncode != 0 or run.timed_out:
                raise RuntimeError(
                    f"optimizer agent failed: returncode={run.returncode}, "
                    f"timed_out={run.timed_out}"
                )
        if not proposal_path.is_file():
            raise RuntimeError("optimizer did not write proposal.json")
        proposal = CandidateProposal.from_dict(read_json(proposal_path))
        if proposal.candidate_id != candidate_id:
            proposal = replace(proposal, candidate_id=candidate_id)
        if proposal.parent_id != state.incumbent_id:
            raise RuntimeError(
                f"proposal parent {proposal.parent_id!r} does not match incumbent "
                f"{state.incumbent_id!r}"
            )
        if proposal.hypothesis_id != packet.target_hypothesis_id:
            raise RuntimeError(
                "proposal must target the frozen TestPacket hypothesis "
                f"{packet.target_hypothesis_id!r}"
            )
        diff_text = source_diff(Path(state.incumbent_source), source)
        return proposal, source, diff_text

    def _public_history(self) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        for path in sorted(
            (self.store.root / "iterations").glob("iter_*/decision.json")
        ):
            raw = read_json(path)
            evidence = _public_evidence(raw.get("evidence") or {})
            decisions.append(
                {
                    "iteration": raw.get("iteration"),
                    "candidate_id": raw.get("candidate_id"),
                    "decision": raw.get("decision"),
                    "aggregate_evidence": evidence,
                }
            )
        archives: list[dict[str, Any]] = []
        for path in sorted(
            (self.store.root / "partial_archive").glob("*/manifest.json")
        ):
            raw = read_json(path)
            archives.append(
                {
                    "candidate_id": raw.get("candidate_id"),
                    "packet_id": raw.get("packet_id"),
                    "hypothesis_id": raw.get("hypothesis_id"),
                    "aggregate_evidence": _public_evidence(raw.get("evidence") or {}),
                }
            )
        return {"decisions": decisions, "partial_archives": archives}

    def _run_auditor(
        self,
        *,
        iteration: int,
        iteration_dir: Path,
        state: RunState,
        proposal: CandidateProposal,
        candidate_source: Path,
        diff_text: str,
    ) -> float:
        if self.auditor_agent is None:
            return 0.0
        workspace = iteration_dir / "auditor" / "workspace"
        output = workspace / "output"
        incumbent_copy = workspace / "incumbent_source"
        candidate_copy = workspace / "candidate_source"
        if not (output / "test_packet.json").is_file():
            copy_source(Path(state.incumbent_source), incumbent_copy)
            copy_source(candidate_source, candidate_copy)
            diff_path = workspace / "candidate.diff"
            diff_path.write_text(diff_text, encoding="utf-8")
            proposal_path = workspace / "proposal.json"
            write_json(proposal_path, proposal.to_dict())
            run = self.auditor_agent.run(
                role="auditor",
                prompt=auditor_prompt(
                    benchmark_context=self.benchmark.context(),
                    incumbent_source=incumbent_copy,
                    candidate_source=candidate_copy,
                    diff_path=diff_path,
                    proposal_path=proposal_path,
                    output_dir=output,
                ),
                workspace=workspace,
                log_dir=iteration_dir / "auditor" / "agent",
            )
            if run.returncode != 0 or run.timed_out:
                self.store.append_event(
                    "auditor_failed",
                    iteration=iteration,
                    returncode=run.returncode,
                    timed_out=run.timed_out,
                )
                return 0.0
        try:
            packet = load_test_packet(output)
        except InvalidTestPacket as exc:
            self.store.append_event(
                "auditor_packet_rejected", iteration=iteration, error=str(exc)
            )
            return 0.0
        incumbent = run_test_cases(
            packet=packet,
            bundle=output,
            source=Path(state.incumbent_source),
            subject="incumbent",
            output_dir=iteration_dir / "auditor" / "incumbent",
            python=self.config.benchmark.unit_python,
        )
        score, reasons = admission_score(packet, incumbent)
        if score < self.config.decision.min_admission_score:
            self.store.append_event(
                "auditor_packet_rejected",
                iteration=iteration,
                score=score,
                reasons=reasons,
            )
            return 0.0
        candidate = run_test_cases(
            packet=packet,
            bundle=output,
            source=candidate_source,
            subject="candidate",
            output_dir=iteration_dir / "auditor" / "candidate",
            python=self.config.benchmark.unit_python,
        )
        return paired_test_metrics(packet, incumbent, candidate)["regression_loss"]

    def _should_challenge_rejection(self, iteration: int) -> bool:
        rate = max(0.0, min(1.0, self.config.decision.rejected_probe_rate))
        return random.Random(self.config.loop.seed + iteration).random() < rate

    def _finalize_iteration(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        packet: TestPacket,
        candidate_source: Path,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
    ) -> RunState:
        # Ensure a decision created before the last escalation carries final evidence.
        if decision.evidence != evidence:
            decision = replace(decision, evidence=evidence)
        write_json(iteration_dir / "evidence.json", evidence.to_dict())
        write_json(iteration_dir / "decision.json", decision.to_dict())
        families = {case.family_id for case in packet.cases}
        self.calibrator.update(
            evidence=evidence, decision=decision, family_ids=families
        )
        state.total_cost += evidence.total_cost

        if decision.decision == Decision.PROMOTE:
            state.incumbent_id = evidence.candidate_id
            state.incumbent_source = str(candidate_source.resolve())
            state.incumbent_diagnostic_score += float(evidence.diagnostic_delta or 0.0)
            state.incumbent_canary_score = float(
                state.incumbent_canary_score or 0.0
            ) + float(evidence.canary_delta or 0.0)
            state.promoted_ids.append(evidence.candidate_id)
            state.active_packet_id = ""
            state.active_packet_path = ""
            state.active_packet_uses = 0
        elif decision.decision == Decision.PARTIAL_ARCHIVE:
            archive_dir = self.store.root / "partial_archive" / evidence.candidate_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                archive_dir / "manifest.json",
                {
                    "candidate_id": evidence.candidate_id,
                    "source": str(candidate_source.resolve()),
                    "packet_id": packet.packet_id,
                    "hypothesis_id": packet.target_hypothesis_id,
                    "evidence": evidence.to_dict(),
                    "decision_reason": decision.reason,
                },
            )
            state.partial_archive_ids.append(evidence.candidate_id)
            state.active_packet_id = ""
            state.active_packet_path = ""
            state.active_packet_uses = 0
        elif decision.decision == Decision.TEST_CHALLENGE:
            state.challenged_packet_ids.append(packet.packet_id)
            state.active_packet_id = ""
            state.active_packet_path = ""
            state.active_packet_uses = 0
        else:
            state.active_packet_uses += 1
            if state.active_packet_uses >= self.config.loop.max_candidates_per_packet:
                state.active_packet_id = ""
                state.active_packet_path = ""
                state.active_packet_uses = 0

        state.next_iteration = iteration + 1
        state.status = "running"
        self.store.save_state(state)
        write_json(
            iteration_dir / "iteration_status.json",
            {
                "iteration": iteration,
                "status": "completed",
                "decision": decision.decision.value,
                "candidate_id": evidence.candidate_id,
                "next_iteration": state.next_iteration,
            },
        )
        self.store.append_event(
            "iteration_completed",
            iteration=iteration,
            candidate_id=evidence.candidate_id,
            packet_id=packet.packet_id,
            decision=decision.decision.value,
            reason=decision.reason,
        )
        return state


def _matched_delta(
    incumbent: BenchmarkEvaluation, candidate: BenchmarkEvaluation
) -> float:
    base = {item.task_id: item.score for item in incumbent.outcomes}
    pairs = [
        item.score - base[item.task_id]
        for item in candidate.outcomes
        if item.task_id in base
    ]
    return sum(pairs) / len(pairs) if pairs else 0.0


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _public_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "admission_score",
        "bridge_gain",
        "canary_delta",
        "diagnostic_delta",
        "hidden_gain",
        "packet_id",
        "public_gain",
        "regression_loss",
    }
    return {key: value.get(key) for key in sorted(allowed) if key in value}


def _with_natural_cost(
    metadata: Mapping[str, Any],
    cost: float,
    **updates: Any,
) -> dict[str, Any]:
    result = dict(metadata)
    costs = dict(result.get("costs") or {})
    costs["natural_task_tokens"] = float(
        costs.get("natural_task_tokens") or 0.0
    ) + float(cost)
    result["costs"] = costs
    result.update(updates)
    return result


def _external_symlink_violations(source: Path) -> list[str]:
    root = source.resolve()
    violations: list[str] = []
    for path in source.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            target = path.resolve(strict=False)
        except OSError:
            violations.append(
                f"unresolvable source symlink: {path.relative_to(source)}"
            )
            continue
        if target != root and root not in target.parents:
            violations.append(
                f"source symlink escapes candidate snapshot: "
                f"{path.relative_to(source)} -> {target}"
            )
    return violations
