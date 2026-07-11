from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceunit.models import TestPacket
from traceunit.ontology import prompt_definitions


def test_author_prompt(
    *,
    benchmark_context: str,
    trace_manifest: Path,
    incumbent_source: Path,
    ut_memory_path: Path | None,
    output_dir: Path,
) -> str:
    example = {
        "packet_id": "iter001_verification_contract",
        "version": 1,
        "source_trace_ids": ["trace-id"],
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "family": "verification",
                "intervention_kind": "capability_augmentation",
                "mechanism": "draft accepted without adversarial edge-case generation",
                "target_boundary": "callable or trajectory-prefix boundary",
                "claim": "falsifiable causal claim",
                "evidence_trace_ids": ["trace-id"],
                "alternatives": ["h2"],
                "confidence": 0.7,
            },
            {
                "hypothesis_id": "h2",
                "family": "context",
                "intervention_kind": "orchestration_change",
                "mechanism": "distinct alternative explanation",
                "target_boundary": "different callable or trajectory boundary",
                "claim": "competing falsifiable causal claim",
                "evidence_trace_ids": ["trace-id"],
                "alternatives": ["h1"],
                "confidence": 0.3,
            },
        ],
        "target_hypothesis_id": "h1",
        "primary_family": "verification",
        "public_contract": "implementation-independent repaired behavior",
        "hidden_variant_strategy": "structural variations that preserve the mechanism",
        "cases": [
            {
                "case_id": "public_reproducer",
                "tier": "public",
                "evidence_role": "target_reproducer",
                "execution_mode": "deterministic",
                "path": "tests/public/test_reproducer.py",
                "driver": "python",
                "expected_incumbent_pass": False,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "hidden_sibling",
                "tier": "hidden",
                "evidence_role": "structural_sibling",
                "execution_mode": "deterministic",
                "path": "tests/hidden/test_sibling.py",
                "driver": "python",
                "expected_incumbent_pass": False,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "existing_behavior",
                "tier": "regression",
                "evidence_role": "off_target_control",
                "execution_mode": "deterministic",
                "path": "tests/hidden/test_regression.py",
                "driver": "python",
                "expected_incumbent_pass": True,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "positive_witness",
                "tier": "admission",
                "evidence_role": "positive_witness",
                "execution_mode": "deterministic",
                "path": "tests/hidden/test_positive_witness.py",
                "driver": "python",
                "expected_incumbent_pass": True,
                "expected_candidate_pass": True,
            },
        ],
        "status": "proposed",
        "admission_passed": False,
        "content_sha256": "",
        "metadata": {},
    }
    memory_input = (
        f"- online UT-design world model: {ut_memory_path}"
        if ut_memory_path is not None
        else "- online UT-design memory: disabled for this experiment condition"
    )
    memory_guidance = (
        "The world model contains sanitized lessons from earlier TestPackets. Use it only to "
        "improve reproducer, hidden-sibling, intervention, bridge, and regression design. It "
        "does not rank directions and must never override the current trace evidence."
        if ut_memory_path is not None
        else "No online UT-design memory is available in this condition."
    )
    return f"""You are the Test Author in a trace-conditioned optimization protocol.

Author a causal TestPacket before any candidate edit or composition plan exists. Diagnose at
least two trace-supported hypotheses, choose one, and distinguish it from the alternatives.
The tests measure agent policy behavior; they must not solve or grade benchmark tasks.

Choose family only from the frozen L0 registry:
{prompt_definitions()}

These are coarse diagnostic directions, never transfer scores. Choose intervention_kind from:
local_repair, capability_augmentation, orchestration_change. Multi-agent, debate, red-team,
self-critique, retrieval modules, and similar scaffolds are capability augmentations or
orchestration changes, not new families.

Benchmark contract:
{benchmark_context}

Inputs:
- normalized trace evidence: {trace_manifest} (failing traces worst-first and passing
  traces best-first, flagged by "passed"; skim the manifest and choose which traces to
  read in depth, using passing traces as behavioral contrast)
- incumbent source: {incumbent_source}
{memory_input}

{memory_guidance} Set packet.primary_family to the selected target hypothesis family. Keep the
specific mechanism in mechanism, claim, target_boundary, and the tests. Do not put family labels
on individual cases.

Write under {output_dir}:
- test_packet.json
- tests/public/* for exactly one visible reproducer
- tests/hidden/* for structural siblings, bridge probes, admission checks, and regressions

Deterministic tests receive TRACEUNIT_SOURCE, TRACEUNIT_TEST_BUNDLE, and TRACEUNIT_SUBJECT. Use
'python' or 'pytest'; do not use network, evaluator APIs, gold data, held-out artifacts, or task
ids. Include a public target, a structurally varied hidden target, a positive-witness
intervention, and an off-target regression. Add a downstream bridge whenever it can be represented
without a grader. For model-backed capability behavior, write a declarative JSON probe, set
execution_mode='model_backed_probe' and driver='agent_probe', and declare strict max_model_calls
and max_tokens. Generated code must never call a model API or access credentials.
Prefer mutation-based contracts that test counterexample discovery, critique adoption, and final
correction rather than merely checking that a critic/debate component exists. Run every supported
test against the incumbent before finishing. Do not edit the incumbent.

Required JSON shape:
{json.dumps(example, indent=2, ensure_ascii=False)}
"""


def candidate_edit_prompt(
    *,
    benchmark_context: str,
    candidate_id: str,
    parent_id: str,
    source_dir: Path,
    public_packet_path: Path,
    latent_capabilities_path: Path | None,
    proposal_path: Path,
) -> str:
    proposal = {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "hypothesis_id": "copy from public packet",
        "intervention_kind": "copy from target hypothesis",
        "mechanism_claim": "falsifiable mechanism-level change",
        "predicted_effect": "expected unit/search effect",
        "regression_risks": ["behavior that could regress"],
        "metadata": {"notes": ""},
    }
    latent_input = (
        f"Latent capabilities: {latent_capabilities_path}"
        if latent_capabilities_path is not None
        else "Latent capabilities: none available in this condition"
    )
    latent_guidance = (
        "Each latent capability is a previously certified but not yet promoted behavior "
        "contract, with a reference patch from the source tree it was written against. "
        "You may realize any of them alongside the new mechanism edit when they genuinely "
        "fit; treat the patch as reference material, never as something to apply blindly. "
        "Realization is measured afterwards by replaying each latent capability's frozen "
        "tests — there is nothing to declare."
        if latent_capabilities_path is not None
        else ""
    )
    return f"""You are the Candidate Editor.

Benchmark contract:
{benchmark_context}

Editable source: {source_dir}
Public frozen TestPacket: {public_packet_path}
{latent_input}

Implement one general mechanism-level edit that repairs the frozen public contract.
{latent_guidance}
Generalize beyond the visible reproducer. Do not inspect hidden tests, search-pool tasks,
final tasks, evaluators, gold data, or task ids. Run the public test and a syntax/import
smoke check.

Write {proposal_path}:
{json.dumps(proposal, indent=2, ensure_ascii=False)}
"""


def regression_author_prompt(
    *,
    benchmark_context: str,
    incumbent_source: Path,
    candidate_source: Path,
    diff_path: Path,
    proposal_path: Path,
    output_dir: Path,
) -> str:
    return f"""You are the post-edit Regression Author. You do not decide promotion or repair code.

Benchmark contract:
{benchmark_context}

Incumbent source: {incumbent_source}
Candidate source: {candidate_source}
Source diff: {diff_path}
Candidate claim: {proposal_path}

Author implementation-independent regression/admission tests under {output_dir}. Every target case
must pass the incumbent and use tier 'regression' or 'admission'. Set
metadata.packet_kind='regression'. Do not access benchmark graders, gold data, search/final
tasks, or network resources. Run all tests against the incumbent before finishing.
"""


def ut_critic_prompt(*, reflection_input: Path, output_path: Path) -> str:
    example = {
        "assessment": "likely_test_gap",
        "suspected_gap": "the packet checked critic invocation but not critique adoption",
        "recommendation": (
            "For similar traces, test counterexample discovery, delivery to the solver, "
            "and a resulting correction; vary the hidden edge case structurally."
        ),
        "alternative_explanation": "the candidate may have overfit the visible contract",
        "confidence": "low",
    }
    return f"""You are the controller-side UT Critic. Diagnose how an earlier frozen TestPacket
could be improved; do not repair source and do not rank L0 directions.

Sanitized online search-feedback summary: {reflection_input}

The input contains only a coarse natural label, frozen test-design metadata, and aggregate unit
evidence. It deliberately contains no search-pool task content, task-level outcome, exact natural
delta, final artifact, or family score. A unit/natural mismatch does not prove the UT was wrong.
Choose exactly one assessment: likely_test_gap, likely_edit_overfit, trajectory_interaction, or
insufficient_evidence. Composition outcomes have low attribution: derive only interaction/bridge
test lessons and never blame or credit one L0 direction. Recommendations must be general test-design
rules, not candidate-, task-, repository-, or benchmark-answer-specific facts.

Write {output_path} as JSON:
{json.dumps(example, indent=2, ensure_ascii=False)}
"""


def score_only_edit_prompt(
    *,
    benchmark_context: str,
    candidate_id: str,
    parent_id: str,
    incumbent_search_score: float,
    source_dir: Path,
    trace_manifest: Path,
    history_path: Path,
    proposal_path: Path,
) -> str:
    proposal = {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "mechanism_claim": "trace-supported mechanism-level change",
        "predicted_effect": "expected search-score effect",
        "regression_risks": ["behavior that could regress"],
        "metadata": {"notes": ""},
    }
    return f"""You are the editor in the score-only Meta-Harness baseline.

Benchmark contract:
{benchmark_context}

Editable source: {source_dir}
Current aggregate search score: {incumbent_search_score}
Current search traces: {trace_manifest}
Prior score-only decisions: {history_path}

Diagnose the failed search trajectories and make one general mechanism-level improvement. This
condition has no generated TestPacket, unit-test feedback, component archive, online UT memory,
or final-task feedback. Do not access benchmark evaluators, gold data, held-out pools, final tasks,
or task-specific answers. Run only a syntax/import smoke check before finishing.

Write {proposal_path}:
{json.dumps(proposal, indent=2, ensure_ascii=False)}
"""


def public_packet(packet: TestPacket) -> dict[str, Any]:
    return {
        "packet_id": packet.packet_id,
        "version": packet.version,
        "target_hypothesis_id": packet.target_hypothesis_id,
        "primary_family": (
            packet.primary_family.value if packet.primary_family is not None else None
        ),
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "family": item.family.value,
                "intervention_kind": item.intervention_kind.value,
                "mechanism": item.mechanism,
                "target_boundary": item.target_boundary,
                "claim": item.claim,
                "confidence": item.confidence,
            }
            for item in packet.hypotheses
        ],
        "public_contract": packet.public_contract,
        "cases": [
            case.__dict__ for case in packet.cases if case.tier.value == "public"
        ],
        "content_sha256": packet.content_sha256,
    }
