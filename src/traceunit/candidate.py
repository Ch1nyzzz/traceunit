from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from traceunit.agents.prompts import (
    candidate_edit_prompt,
    public_packet,
    search_plan_prompt,
)
from traceunit.agents.runner import WorkspaceAgent
from traceunit.archive import (
    ArchiveCatalog,
    ComponentSelection,
    CompositionPlan,
)
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.composition import CompositionExecutor
from traceunit.config import ProjectConfig
from traceunit.io import (
    append_jsonl,
    copy_source,
    read_json,
    read_jsonl,
    safe_relative_path,
    sha256_file,
    sha256_tree,
    source_diff,
    write_json,
)
from traceunit.models import CandidateProposal, RunState, TestPacket, TestTier
from traceunit.store import RunStore


class CandidateBuildError(RuntimeError):
    pass


class CandidateBuilder:
    """Two-stage search: freeze retrieval plan, then edit materialized source."""

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
        alignment_cards_path: Path | None,
    ) -> tuple[CandidateProposal, CompositionPlan, Path, str]:
        candidate_id = f"iter{iteration:03d}_candidate"
        candidate_dir = self.store.candidate_dir(candidate_id)
        source = candidate_dir / "source"
        proposal_path = candidate_dir / "proposal.json"
        plan_path = candidate_dir / "composition_plan.json"
        receipt_path = candidate_dir / "materialization_receipt.json"
        public_path = candidate_dir / "public_packet.json"
        history_path = candidate_dir / "history.json"
        catalog_path = (
            candidate_dir / "archive_catalog.json"
            if self.config.capabilities.partial_archive
            else None
        )
        cards_path = (
            candidate_dir / "alignment_cards.json"
            if alignment_cards_path is not None
            else None
        )
        parent_copy = candidate_dir / "parent_source"
        lock_path = iteration_dir / "composition_lock.json"

        if not proposal_path.is_file():
            self._stage_public_inputs(
                state=state,
                packet=packet,
                packet_path=packet_path,
                public_path=public_path,
                history_path=history_path,
                catalog_path=catalog_path,
                cards_source=alignment_cards_path,
                cards_path=cards_path,
                parent_copy=parent_copy,
                candidate_dir=candidate_dir,
            )
            catalog = self.catalog()
            plan = self._plan(
                state=state,
                iteration=iteration,
                iteration_dir=iteration_dir,
                candidate_dir=candidate_dir,
                public_path=public_path,
                history_path=history_path,
                catalog_path=catalog_path,
                cards_path=cards_path,
                parent_copy=parent_copy,
                plan_path=plan_path,
                catalog=catalog,
                packet_id=packet.packet_id,
            )
            self._freeze_plan_lock(
                lock_path=lock_path,
                plan_path=plan_path,
                plan=plan,
                packet=packet,
            )
            receipt = CompositionExecutor(
                self.store.component_archive_root, catalog
            ).materialize(
                plan=plan,
                parent_source=Path(state.incumbent_source),
                destination=source,
            )
            write_json(receipt_path, receipt.to_dict())
            prompt = candidate_edit_prompt(
                benchmark_context=self.benchmark.context(),
                candidate_id=candidate_id,
                parent_id=state.incumbent_id,
                source_dir=source,
                public_packet_path=public_path,
                plan_path=plan_path,
                materialization_receipt_path=receipt_path,
                proposal_path=proposal_path,
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
            self._verify_plan_lock(
                lock_path=lock_path,
                plan_path=plan_path,
                plan=plan,
                packet=packet,
            )

        required = (proposal_path, plan_path, receipt_path, lock_path, source)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise CandidateBuildError(
                "candidate build is incomplete; missing " + ", ".join(missing)
            )
        catalog = self.catalog()
        plan = CompositionPlan.from_dict(read_json(plan_path))
        catalog.validate_plan(plan)
        self._verify_plan_lock(
            lock_path=lock_path,
            plan_path=plan_path,
            plan=plan,
            packet=packet,
        )
        if plan.base_source_sha256 != sha256_tree(Path(state.incumbent_source)):
            raise CandidateBuildError(
                "frozen plan no longer matches the incumbent source"
            )
        proposal = CandidateProposal.from_dict(read_json(proposal_path))
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
        if proposal.plan_id != plan.attempt_fingerprint:
            raise CandidateBuildError("proposal plan_id does not match frozen plan")
        if proposal.selected_archive_ids != plan.component_ids:
            raise CandidateBuildError(
                "proposal selected_archive_ids do not match frozen plan order"
            )
        return (
            proposal,
            plan,
            source,
            source_diff(Path(state.incumbent_source), source),
        )

    def _plan(
        self,
        *,
        state: RunState,
        iteration: int,
        iteration_dir: Path,
        candidate_dir: Path,
        public_path: Path,
        history_path: Path,
        catalog_path: Path | None,
        cards_path: Path | None,
        parent_copy: Path,
        plan_path: Path,
        catalog: ArchiveCatalog,
        packet_id: str,
    ) -> CompositionPlan:
        if not plan_path.is_file():
            parent_hash = sha256_tree(parent_copy)
            prompt = search_plan_prompt(
                benchmark_context=self.benchmark.context(),
                parent_id=state.incumbent_id,
                parent_source_sha256=sha256_tree(Path(state.incumbent_source)),
                parent_source_path=parent_copy,
                public_packet_path=public_path,
                history_path=history_path,
                archive_catalog_path=catalog_path,
                alignment_cards_path=cards_path,
                plan_path=plan_path,
            )
            run = self.search_agent.run(
                role="search_planner",
                prompt=prompt,
                workspace=candidate_dir,
                log_dir=iteration_dir / "search_planner",
            )
            if run.returncode != 0 or run.timed_out:
                raise CandidateBuildError(
                    f"search planner failed: returncode={run.returncode}, "
                    f"timed_out={run.timed_out}"
                )
            if sha256_tree(parent_copy) != parent_hash:
                raise CandidateBuildError(
                    "search planner modified the read-only parent copy"
                )
        if not plan_path.is_file():
            raise CandidateBuildError(
                "search planner did not write composition_plan.json"
            )
        raw = read_json(plan_path)
        requested = tuple(
            ComponentSelection.from_dict(item) for item in raw.get("selections") or []
        )
        plan = catalog.make_plan(
            base_source_sha256=sha256_tree(Path(state.incumbent_source)),
            selections=requested,
            integration_instructions=str(raw.get("integration_instructions") or ""),
        )
        if not self.config.capabilities.partial_archive and plan.component_ids:
            raise CandidateBuildError("archive composition is disabled for this run")
        if not self.config.archive.allow_semantic_port and any(
            selection.mode.value == "semantic" for selection in plan.selections
        ):
            raise CandidateBuildError(
                "semantic archive ports are disabled for this run"
            )
        self._reject_duplicate_attempt(
            candidate_id=f"iter{iteration:03d}_candidate",
            parent_id=state.incumbent_id,
            packet_id=packet_id,
            plan=plan,
        )
        write_json(plan_path, plan.to_dict())
        if not any(
            row.get("candidate_id") == f"iter{iteration:03d}_candidate"
            and row.get("attempt_fingerprint") == plan.attempt_fingerprint
            for row in read_jsonl(self.store.search_attempts_path)
        ):
            append_jsonl(
                self.store.search_attempts_path,
                {
                    "candidate_id": f"iter{iteration:03d}_candidate",
                    "parent_id": state.incumbent_id,
                    "packet_id": packet_id,
                    "attempt_fingerprint": plan.attempt_fingerprint,
                    "component_ids": plan.component_ids,
                    "status": "planned",
                },
            )
        return plan

    def _stage_public_inputs(
        self,
        *,
        state: RunState,
        packet: TestPacket,
        packet_path: Path,
        public_path: Path,
        history_path: Path,
        catalog_path: Path | None,
        cards_source: Path | None,
        cards_path: Path | None,
        parent_copy: Path,
        candidate_dir: Path,
    ) -> None:
        copy_source(Path(state.incumbent_source), parent_copy)
        write_json(public_path, public_packet(packet))
        for case in packet.cases:
            if case.tier is not TestTier.PUBLIC:
                continue
            target = candidate_dir / case.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(packet_path / case.path, target)
        write_json(history_path, self.public_history())
        if catalog_path is not None:
            write_json(catalog_path, self.public_archive_catalog(candidate_dir))
        if cards_source is not None and cards_path is not None:
            shutil.copy2(cards_source, cards_path)

    def catalog(self) -> ArchiveCatalog:
        if not self.config.capabilities.partial_archive:
            return ArchiveCatalog()
        manifests = list(self.store.component_archive_root.glob("*/manifest.json"))
        if not manifests:
            return ArchiveCatalog()
        return ArchiveCatalog.load(self.store.component_archive_root)

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
                            "admission_score",
                            "archive_replay_passed",
                            "bridge_gain",
                            "hidden_gain",
                            "preservation_passed",
                            "public_gain",
                            "regression_loss",
                            "search_delta",
                        )
                        if key in evidence
                    },
                }
            )
        return {"decisions": decisions}

    def public_archive_catalog(self, workspace: Path) -> dict[str, Any]:
        catalog = self.catalog()
        components = []
        for manifest in catalog:
            source_patch = safe_relative_path(
                self.store.component_archive_root, manifest.patch_path
            )
            if (
                not source_patch.is_file()
                or sha256_file(source_patch) != manifest.patch_sha256
            ):
                raise CandidateBuildError(
                    f"archive component is corrupt: {manifest.archive_id}"
                )
            public_patch = (
                workspace
                / "archive_components"
                / manifest.archive_id
                / "component.patch"
            )
            public_patch.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_patch, public_patch)
            components.append(
                {
                    "component_id": manifest.archive_id,
                    "kind": manifest.kind.value,
                    "mechanism": manifest.mechanism,
                    "target_boundary": manifest.target_boundary,
                    "family_keys": manifest.certificate.family_keys,
                    "dependencies": manifest.dependencies,
                    "constituents": manifest.constituents,
                    "applicability": manifest.applicability,
                    "patch_path": str(public_patch.relative_to(workspace)),
                    "local_certificate": {
                        "public_passed": manifest.certificate.public_passed,
                        "hidden_passed": manifest.certificate.hidden_passed,
                        "bridge_passed": manifest.certificate.bridge_passed,
                        "regression_passed": manifest.certificate.regression_passed,
                    },
                }
            )
        return {"components": components}

    @staticmethod
    def _freeze_plan_lock(
        *,
        lock_path: Path,
        plan_path: Path,
        plan: CompositionPlan,
        packet: TestPacket,
    ) -> None:
        payload = {
            "plan_sha256": sha256_file(plan_path),
            "attempt_fingerprint": plan.attempt_fingerprint,
            "base_source_sha256": plan.base_source_sha256,
            "packet_content_sha256": packet.content_sha256,
        }
        if lock_path.is_file() and read_json(lock_path) != payload:
            raise CandidateBuildError("composition lock conflicts with the frozen plan")
        write_json(lock_path, payload)

    @staticmethod
    def _verify_plan_lock(
        *,
        lock_path: Path,
        plan_path: Path,
        plan: CompositionPlan,
        packet: TestPacket,
    ) -> None:
        if not lock_path.is_file():
            raise CandidateBuildError("host composition lock is missing")
        lock = read_json(lock_path)
        expected = {
            "plan_sha256": sha256_file(plan_path),
            "attempt_fingerprint": plan.attempt_fingerprint,
            "base_source_sha256": plan.base_source_sha256,
            "packet_content_sha256": packet.content_sha256,
        }
        if lock != expected:
            raise CandidateBuildError(
                "frozen composition plan was modified after planning"
            )

    def _reject_duplicate_attempt(
        self,
        *,
        candidate_id: str,
        parent_id: str,
        packet_id: str,
        plan: CompositionPlan,
    ) -> None:
        for row in read_jsonl(self.store.search_attempts_path):
            if row.get("candidate_id") == candidate_id:
                continue
            if (
                row.get("parent_id") == parent_id
                and row.get("packet_id") == packet_id
                and row.get("attempt_fingerprint") == plan.attempt_fingerprint
            ):
                raise CandidateBuildError(
                    "search planner repeated an existing composition attempt"
                )
