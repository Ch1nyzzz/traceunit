from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceunit.models import TestPacket


def test_author_prompt(
    *,
    benchmark_context: str,
    trace_manifest: Path,
    incumbent_source: Path,
    alignment_cards_path: Path | None,
    output_dir: Path,
) -> str:
    example = {
        "packet_id": "iter001_retrieval_contract",
        "version": 1,
        "source_trace_ids": ["trace-id"],
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "mechanism": "specific behavior",
                "target_boundary": "callable or trajectory-prefix boundary",
                "claim": "falsifiable causal claim",
                "evidence_trace_ids": ["trace-id"],
                "alternatives": ["h2"],
                "confidence": 0.7,
            },
            {
                "hypothesis_id": "h2",
                "mechanism": "distinct alternative explanation",
                "target_boundary": "different callable or trajectory boundary",
                "claim": "competing falsifiable causal claim",
                "evidence_trace_ids": ["trace-id"],
                "alternatives": ["h1"],
                "confidence": 0.3,
            },
        ],
        "target_hypothesis_id": "h1",
        "public_contract": "implementation-independent repaired behavior",
        "hidden_variant_strategy": "structural variations that preserve the mechanism",
        "cases": [
            {
                "case_id": "public_reproducer",
                "family_id": "retrieval.verify.empty_result",
                "tier": "public",
                "path": "tests/public/test_reproducer.py",
                "driver": "python",
                "expected_incumbent_pass": False,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "hidden_sibling",
                "family_id": "retrieval.verify.empty_result",
                "tier": "hidden",
                "path": "tests/hidden/test_sibling.py",
                "driver": "python",
                "expected_incumbent_pass": False,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "existing_behavior",
                "family_id": "retrieval.neighbor.preservation",
                "tier": "regression",
                "path": "tests/hidden/test_regression.py",
                "driver": "python",
                "expected_incumbent_pass": True,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "positive_witness",
                "family_id": "retrieval.verify.empty_result",
                "tier": "admission",
                "path": "tests/hidden/test_positive_witness.py",
                "driver": "python",
                "expected_incumbent_pass": True,
                "expected_candidate_pass": True,
                "admission_role": "positive_witness",
            },
        ],
        "status": "proposed",
        "admission_score": 0.0,
        "content_sha256": "",
        "metadata": {
            "mechanism_class": "retrieval",
            "intervention_kind": "controlled_boundary_value",
        },
    }
    alignment_input = (
        f"- delayed family-level alignment cards: {alignment_cards_path}"
        if alignment_cards_path is not None
        else "- delayed alignment feedback: disabled for this experiment condition"
    )
    alignment_guidance = (
        "Alignment cards summarize prior candidate cohorts. They never describe the current "
        "candidate and contain no task-level outcomes. Use them to strengthen unreliable test "
        "designs, not to imitate a score."
        if alignment_cards_path is not None
        else "No calibration-derived feedback is available in this condition."
    )
    return f"""You are the Test Author in a trace-conditioned optimization protocol.

Author a causal TestPacket before any candidate edit or composition plan exists. Diagnose at
least two trace-supported hypotheses, choose one, and distinguish it from the alternatives.
The tests measure agent policy behavior; they must not solve or grade benchmark tasks.

Benchmark contract:
{benchmark_context}

Inputs:
- normalized trace evidence: {trace_manifest}
- incumbent source: {incumbent_source}
{alignment_input}

{alignment_guidance} A family_id must be a stable mechanism key such as
'planner.tool_selection.invalid_candidate'; never include task ids, repository names, or free-form
hypothesis text in it.

Write under {output_dir}:
- test_packet.json
- tests/public/* for exactly one visible reproducer
- tests/hidden/* for structural siblings, bridge probes, admission checks, and regressions

Tests receive TRACEUNIT_SOURCE, TRACEUNIT_TEST_BUNDLE, and TRACEUNIT_SUBJECT. Use only 'python' or
'pytest'; do not use network, evaluator APIs, gold data, held-out artifacts, or task ids. Include a
public target, a structurally varied hidden target, a positive-witness intervention, and an
off-target regression. Add a downstream bridge whenever it can be represented without a grader.
Run every test against the incumbent before finishing. Do not edit the incumbent.

Required JSON shape:
{json.dumps(example, indent=2, ensure_ascii=False)}
"""


def search_plan_prompt(
    *,
    benchmark_context: str,
    parent_id: str,
    parent_source_sha256: str,
    parent_source_path: Path,
    public_packet_path: Path,
    history_path: Path,
    archive_catalog_path: Path | None,
    alignment_cards_path: Path | None,
    plan_path: Path,
) -> str:
    example = {
        "schema_version": 1,
        "base_source_sha256": parent_source_sha256,
        "selections": [
            {
                "component_id": "copy an exact 64-character id from the catalog",
                "mode": "exact",
                "semantic_instructions": "",
                "rationale": "locally certified mechanism applies here",
            }
        ],
        "integration_instructions": "new mechanism edit or integration work",
    }
    archive_input = (
        f"- certified component catalog: {archive_catalog_path}"
        if archive_catalog_path is not None
        else "- certified component catalog: disabled; selections must be empty"
    )
    alignment_input = (
        f"- delayed alignment cards: {alignment_cards_path}"
        if alignment_cards_path is not None
        else "- delayed alignment feedback: disabled"
    )
    return f"""You are the Search Planner. Freeze a composition plan before source editing.

Benchmark contract:
{benchmark_context}

Inputs:
- read-only parent source: {parent_source_path}
- public frozen TestPacket: {public_packet_path}
- prior decisions: {history_path}
{archive_input}
{alignment_input}

Select zero, one, or any number of archive components. There is no top-k component limit. Choose
'exact' only when the component's applicability contract holds; choose 'semantic' explicitly
when the mechanism applies but its patch preimage does not. When present, alignment cards are
uncertain priors, not certificates. Never request hidden tests, private calibration observations, natural-task
details, or final-evaluation artifacts.

Write {plan_path} with this shape:
{json.dumps(example, indent=2, ensure_ascii=False)}

Do not edit source files in this planning phase.
"""


def candidate_edit_prompt(
    *,
    benchmark_context: str,
    candidate_id: str,
    parent_id: str,
    source_dir: Path,
    public_packet_path: Path,
    plan_path: Path,
    materialization_receipt_path: Path,
    proposal_path: Path,
) -> str:
    proposal = {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "hypothesis_id": "copy from public packet",
        "mechanism_claim": "falsifiable mechanism-level change",
        "predicted_effect": "expected unit/search effect",
        "regression_risks": ["behavior that could regress"],
        "plan_id": "copy from frozen plan",
        "selected_archive_ids": ["copy selected component ids in plan order"],
        "metadata": {"notes": ""},
    }
    return f"""You are the Candidate Editor.

Benchmark contract:
{benchmark_context}

Editable source: {source_dir}
Public frozen TestPacket: {public_packet_path}
Frozen composition plan: {plan_path}
Host materialization receipt: {materialization_receipt_path}

The host has already applied every exact component. Implement only the declared semantic ports,
integration work, and new mechanism edit. Do not silently add archive components that are absent
from the frozen plan. Generalize beyond the visible reproducer. Do not inspect hidden tests,
calibration tasks, final tasks, evaluators, gold data, or task ids. Run the public test and a
syntax/import smoke check.

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
metadata.packet_kind='regression'. Do not access benchmark graders, gold data, calibration/final
tasks, or network resources. Run all tests against the incumbent before finishing.
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
condition has no generated TestPacket, unit-test feedback, component archive, calibration cards,
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
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
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
