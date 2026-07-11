from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from traceunit.agents.prompts import ut_critic_prompt
from traceunit.agents.runner import WorkspaceAgent
from traceunit.io import append_jsonl, read_json, read_jsonl, write_json
from traceunit.models import (
    AttributionScope,
    CandidateProposal,
    DecisionRecord,
    EvidenceRecord,
    InterventionKind,
    UnitFamily,
)
from traceunit.tests_runtime import load_test_packet


class SearchOutcome(StrEnum):
    IMPROVED = "improved"
    NONINFERIOR = "noninferior"
    REGRESSED = "regressed"


class ReflectionAssessment(StrEnum):
    LIKELY_TEST_GAP = "likely_test_gap"
    LIKELY_EDIT_OVERFIT = "likely_edit_overfit"
    TRAJECTORY_INTERACTION = "trajectory_interaction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class UTFeedbackEpisode:
    """A sanitized online lesson from one frozen packet and one search comparison."""

    candidate_id: str
    iteration: int
    primary_family: UnitFamily
    intervention_kind: InterventionKind
    attribution_scope: AttributionScope
    component_families: tuple[UnitFamily, ...]
    local_contract_passed: bool
    bridge_contract_passed: bool
    search_outcome: SearchOutcome
    assessment: ReflectionAssessment
    suspected_gap: str
    recommendation: str
    alternative_explanation: str
    confidence: str

    @property
    def identity(self) -> tuple[str, int]:
        return self.candidate_id, self.iteration

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["primary_family"] = self.primary_family.value
        value["intervention_kind"] = self.intervention_kind.value
        value["attribution_scope"] = self.attribution_scope.value
        value["component_families"] = [item.value for item in self.component_families]
        value["search_outcome"] = self.search_outcome.value
        value["assessment"] = self.assessment.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UTFeedbackEpisode":
        return cls(
            candidate_id=str(value["candidate_id"]),
            iteration=int(value["iteration"]),
            primary_family=UnitFamily(str(value["primary_family"])),
            intervention_kind=InterventionKind(str(value["intervention_kind"])),
            attribution_scope=AttributionScope(str(value["attribution_scope"])),
            component_families=tuple(
                UnitFamily(str(item)) for item in value.get("component_families") or []
            ),
            local_contract_passed=bool(value.get("local_contract_passed", False)),
            bridge_contract_passed=bool(value.get("bridge_contract_passed", False)),
            search_outcome=SearchOutcome(str(value["search_outcome"])),
            assessment=ReflectionAssessment(str(value["assessment"])),
            suspected_gap=str(value.get("suspected_gap") or ""),
            recommendation=str(value.get("recommendation") or ""),
            alternative_explanation=str(value.get("alternative_explanation") or ""),
            confidence=str(value.get("confidence") or "low"),
        )


class UTMemoryLedger:
    """Append-only episodes plus a compact, chronological UT-design world model."""

    def __init__(self, episode_path: Path) -> None:
        self.episode_path = episode_path
        self._episodes = tuple(
            UTFeedbackEpisode.from_dict(item) for item in read_jsonl(episode_path)
        )
        if len({item.identity for item in self._episodes}) != len(self._episodes):
            raise ValueError("duplicate UT feedback episodes")

    @property
    def episodes(self) -> tuple[UTFeedbackEpisode, ...]:
        return self._episodes

    @property
    def version(self) -> int:
        return max((item.iteration for item in self._episodes), default=0)

    def append_episode(self, episode: UTFeedbackEpisode) -> None:
        if episode.identity in {item.identity for item in self._episodes}:
            return
        append_jsonl(self.episode_path, episode.to_dict())
        self._episodes = (*self._episodes, episode)

    def write_world_model(self, path: Path, *, max_lessons: int = 64) -> None:
        """Write only recent, de-duplicated guidance; never a family scorecard."""

        latest: dict[tuple[str, str], UTFeedbackEpisode] = {}
        for episode in self._episodes:
            if (
                episode.assessment is ReflectionAssessment.INSUFFICIENT_EVIDENCE
                or not episode.recommendation.strip()
            ):
                continue
            section = (
                "__composition__"
                if episode.attribution_scope is AttributionScope.COMPOSITION
                else episode.primary_family.value
            )
            latest[(section, episode.recommendation.strip())] = episode
        selected = sorted(
            latest.values(),
            key=lambda item: (item.iteration, item.candidate_id),
        )[-max_lessons:]
        by_section: dict[str, list[UTFeedbackEpisode]] = {}
        for episode in selected:
            section = (
                "__composition__"
                if episode.attribution_scope is AttributionScope.COMPOSITION
                else episode.primary_family.value
            )
            by_section.setdefault(section, []).append(episode)

        lines = [
            "# TraceUnit UT-design world model",
            "",
            f"Version: {self.version}",
            "",
            "This document contains sanitized lessons from earlier frozen TestPackets",
            "and their later paired search-pool outcomes. It supports online optimization,",
            "not independent held-out validation, and never ranks L0 directions.",
        ]
        for family in UnitFamily:
            lessons = by_section.get(family.value, [])
            if not lessons:
                continue
            lines.extend(["", f"## {family.value}", ""])
            lines.extend(f"- {item.recommendation.strip()}" for item in lessons)
        composition_lessons = by_section.get("__composition__", [])
        if composition_lessons:
            lines.extend(["", "## composition interactions", ""])
            lines.extend(
                f"- {item.recommendation.strip()}" for item in composition_lessons
            )
        content = "\n".join(lines).rstrip() + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        write_json(
            path.with_suffix(".meta.json"),
            {
                "version": self.version,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "lesson_count": len(selected),
            },
        )


class UTMemoryManager:
    """Reflect on each completed search comparison and update future UT design."""

    def __init__(
        self,
        *,
        root: Path,
        ledger: UTMemoryLedger,
        critic: WorkspaceAgent | None,
        max_lessons: int,
        world_model_path: Path,
    ) -> None:
        self.root = root
        self.ledger = ledger
        self.critic = critic
        self.max_lessons = max_lessons
        self.world_model_path = world_model_path

    def reflect_iteration(
        self,
        *,
        iteration: int,
        proposal: CandidateProposal,
        packet_path: Path,
        evidence: EvidenceRecord,
        decision: DecisionRecord,
    ) -> UTFeedbackEpisode | None:
        if evidence.primary_family is None or evidence.search_delta is None:
            return None
        if (proposal.candidate_id, iteration) in {
            item.identity for item in self.ledger.episodes
        }:
            return None

        packet = load_test_packet(packet_path)
        outcome = _search_outcome(evidence.search_delta)
        workspace = (
            self.root / "reflections" / f"iter_{iteration:03d}" / proposal.candidate_id
        )
        input_path = workspace / "reflection_input.json"
        output_path = workspace / "reflection.json"
        write_json(
            input_path,
            {
                "primary_family": evidence.primary_family.value,
                "intervention_kind": evidence.intervention_kind.value,
                "attribution_scope": evidence.attribution_scope.value,
                "component_families": [
                    item.value for item in evidence.component_families
                ],
                "mechanism_claim": proposal.mechanism_claim,
                "public_contract": packet.public_contract,
                "hidden_variant_strategy": packet.hidden_variant_strategy,
                "cases": [
                    {
                        "tier": case.tier.value,
                        "evidence_role": case.evidence_role.value,
                        "execution_mode": case.execution_mode.value,
                        "description": case.description,
                    }
                    for case in packet.cases
                ],
                "local_contract_passed": evidence.contract_passed,
                "bridge_contract_passed": evidence.bridge_contract_passed,
                "aggregate_unit_evidence": {
                    "public_gain": evidence.public_gain,
                    "hidden_gain": evidence.hidden_gain,
                    "bridge_gain": evidence.bridge_gain,
                    "regression_loss": evidence.regression_loss,
                    "realized_latent_count": len(evidence.realized_latent),
                    "preservation_passed": evidence.preservation_passed,
                },
                "search_outcome": outcome.value,
                "committed_decision": decision.decision.value,
            },
        )
        raw = self._run_critic(
            workspace=workspace, input_path=input_path, output_path=output_path
        )
        if not raw:
            raw = _fallback_reflection(
                outcome=outcome,
                local_contract_passed=evidence.contract_passed,
                attribution_scope=evidence.attribution_scope,
            )
            write_json(output_path, raw)

        try:
            assessment = ReflectionAssessment(str(raw.get("assessment") or ""))
        except ValueError:
            assessment = ReflectionAssessment.INSUFFICIENT_EVIDENCE
        confidence = str(raw.get("confidence") or "low").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        episode = UTFeedbackEpisode(
            candidate_id=proposal.candidate_id,
            iteration=iteration,
            primary_family=evidence.primary_family,
            intervention_kind=evidence.intervention_kind,
            attribution_scope=evidence.attribution_scope,
            component_families=evidence.component_families,
            local_contract_passed=evidence.contract_passed,
            bridge_contract_passed=evidence.bridge_contract_passed,
            search_outcome=outcome,
            assessment=assessment,
            suspected_gap=str(raw.get("suspected_gap") or ""),
            recommendation=str(raw.get("recommendation") or ""),
            alternative_explanation=str(raw.get("alternative_explanation") or ""),
            confidence=confidence,
        )
        self.ledger.append_episode(episode)
        self.ledger.write_world_model(
            self.world_model_path,
            max_lessons=self.max_lessons,
        )
        return episode

    def ensure_world_model(self) -> None:
        self.ledger.write_world_model(
            self.world_model_path,
            max_lessons=self.max_lessons,
        )

    def _run_critic(
        self,
        *,
        workspace: Path,
        input_path: Path,
        output_path: Path,
    ) -> Mapping[str, object]:
        if self.critic is None:
            return {}
        run = self.critic.run(
            role="ut_critic",
            prompt=ut_critic_prompt(
                reflection_input=input_path,
                output_path=output_path,
            ),
            workspace=workspace,
            log_dir=workspace / "agent",
        )
        if run.returncode != 0 or run.timed_out or not output_path.is_file():
            return {}
        value = read_json(output_path)
        return value if isinstance(value, Mapping) else {}


def _search_outcome(delta: float) -> SearchOutcome:
    if delta > 0:
        return SearchOutcome.IMPROVED
    if delta < 0:
        return SearchOutcome.REGRESSED
    return SearchOutcome.NONINFERIOR


def _fallback_reflection(
    *,
    outcome: SearchOutcome,
    local_contract_passed: bool,
    attribution_scope: AttributionScope,
) -> dict[str, str]:
    if attribution_scope is AttributionScope.COMPOSITION:
        return {
            "assessment": ReflectionAssessment.TRAJECTORY_INTERACTION.value,
            "suspected_gap": "the local packet may not cover interactions among composed components",
            "recommendation": (
                "For similar compositions, add an integration bridge that exercises data and "
                "control flow across component boundaries; do not infer one component's effect."
            ),
            "alternative_explanation": "the search outcome belongs to the complete composition",
            "confidence": "low",
        }
    if local_contract_passed and outcome is not SearchOutcome.IMPROVED:
        return {
            "assessment": ReflectionAssessment.LIKELY_TEST_GAP.value,
            "suspected_gap": "the frozen local contract did not predict paired search improvement",
            "recommendation": (
                "For similar traces, strengthen structural hidden siblings and a downstream "
                "bridge; test behavior adoption rather than component presence."
            ),
            "alternative_explanation": "the edit may have overfit the local contract",
            "confidence": "low",
        }
    if not local_contract_passed and outcome is SearchOutcome.IMPROVED:
        return {
            "assessment": ReflectionAssessment.LIKELY_TEST_GAP.value,
            "suspected_gap": "search improved despite a failed local contract",
            "recommendation": (
                "For similar traces, check whether the contract targets the actual trajectory "
                "failure and avoid requiring an incidental implementation detail."
            ),
            "alternative_explanation": "the search gain may be unrelated to the target mechanism",
            "confidence": "low",
        }
    if not local_contract_passed:
        return {
            "assessment": ReflectionAssessment.LIKELY_EDIT_OVERFIT.value,
            "suspected_gap": "the candidate did not realize the stated local mechanism",
            "recommendation": (
                "Keep the target reproducer concrete and add a positive witness that distinguishes "
                "the intended intervention from an inert component addition."
            ),
            "alternative_explanation": "the trace diagnosis may be incomplete",
            "confidence": "low",
        }
    return {
        "assessment": ReflectionAssessment.INSUFFICIENT_EVIDENCE.value,
        "suspected_gap": "",
        "recommendation": "",
        "alternative_explanation": "the completed comparison does not isolate a UT design lesson",
        "confidence": "low",
    }
