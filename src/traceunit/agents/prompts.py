from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceunit.models import TestPacket


def experimentalist_prompt(
    *,
    benchmark_context: str,
    trace_manifest: Path,
    incumbent_source: Path,
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
                "mechanism": "a distinct upstream or downstream explanation",
                "target_boundary": "a different callable or trajectory boundary",
                "claim": "a competing falsifiable causal claim",
                "evidence_trace_ids": ["trace-id"],
                "alternatives": ["h1"],
                "confidence": 0.3,
            },
        ],
        "target_hypothesis_id": "h1",
        "public_contract": "implementation-independent behavior expected after repair",
        "hidden_variant_strategy": "how hidden siblings vary surface form while preserving mechanism",
        "cases": [
            {
                "case_id": "public_reproducer",
                "family_id": "mechanism_family",
                "tier": "public",
                "path": "tests/public/test_reproducer.py",
                "driver": "python",
                "arguments": [],
                "environment": {},
                "timeout_s": 60,
                "expected_incumbent_pass": False,
                "expected_candidate_pass": True,
                "description": "what this case isolates",
            },
            {
                "case_id": "hidden_sibling",
                "family_id": "mechanism_family",
                "tier": "hidden",
                "path": "tests/hidden/test_sibling.py",
                "driver": "python",
                "expected_incumbent_pass": False,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "existing_behavior",
                "family_id": "neighbor_regression",
                "tier": "regression",
                "path": "tests/hidden/test_regression.py",
                "driver": "python",
                "expected_incumbent_pass": True,
                "expected_candidate_pass": True,
            },
            {
                "case_id": "positive_witness",
                "family_id": "mechanism_family",
                "tier": "admission",
                "path": "tests/hidden/test_positive_witness.py",
                "driver": "python",
                "expected_incumbent_pass": True,
                "expected_candidate_pass": True,
                "admission_role": "positive_witness",
                "description": "inject a trusted-good boundary value and prove the assertion is satisfiable",
            },
        ],
        "status": "proposed",
        "admission_score": 0.0,
        "content_sha256": "",
        "metadata": {"rationale": ""},
    }
    return f"""You are the Experimentalist in a trace-conditioned optimization loop.

You must diagnose a recurring behavioral failure and author a causal TestPacket BEFORE any
candidate edit exists. You are testing the agent policy, not solving or grading the benchmark
tasks. The benchmark evaluator, gold answers, hidden patches, and held-out state are off limits.
Give at least two distinct trace-supported failure hypotheses. Select one as the target and design
tests that distinguish it from the named alternative rather than merely reproducing the final
symptom.

Benchmark contract:
{benchmark_context}

Inputs:
- normalized failed-trace manifest: {trace_manifest}
- incumbent source snapshot: {incumbent_source}
- full raw evidence paths are referenced by the manifest and may be read

Required outputs under {output_dir}:
- test_packet.json
- tests/public/* for the one public reproducer
- tests/hidden/* for hidden siblings, bridge probes, admission checks, and regressions

Tests execute outside the candidate. Each test receives:
- TRACEUNIT_SOURCE: absolute path of the source snapshot under test
- TRACEUNIT_TEST_BUNDLE: this test bundle
- TRACEUNIT_SUBJECT: incumbent or candidate

Use only driver `python` or `pytest`. `path` must be relative to this output directory.
Tests must not use network access, benchmark evaluators, task ids, gold data, or candidate-specific
hard-coding. Prefer fake dependencies, frozen trajectory prefixes, metamorphic relations, and
behavioral assertions. A public/hidden/bridge target test should fail the incumbent for the
observed mechanism. A regression/admission test should pass the incumbent. Include at least one
public and one hidden case. Include an admission case with admission_role `positive_witness` that
uses a controlled trusted-good intervention to prove the target assertion is satisfiable. Also
include an off-target regression control; a known-bad mutant check is preferred. A bridge test is
strongly preferred when a useful downstream
continuation can be represented without the benchmark grader.

Before finishing, execute every test against the incumbent and correct syntax/import errors.
Do not edit the incumbent source. Do not inspect or anticipate a candidate implementation.

The JSON shape is:
{json.dumps(example, indent=2, ensure_ascii=False)}
"""


def optimizer_prompt(
    *,
    benchmark_context: str,
    candidate_id: str,
    parent_id: str,
    source_dir: Path,
    public_packet_path: Path,
    history_path: Path,
    proposal_path: Path,
) -> str:
    proposal = {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "hypothesis_id": "copy from public packet",
        "mechanism_claim": "one falsifiable mechanism-level change",
        "predicted_effect": "expected public/hidden/natural-task effect and cost direction",
        "regression_risks": ["one class of behavior that could regress"],
        "metadata": {"notes": ""},
    }
    return f"""You are the Optimizer in a trace-conditioned self-improvement loop.

Benchmark contract:
{benchmark_context}

Editable agent source: {source_dir}
Public pre-edit test contract: {public_packet_path}
Prior decisions and certified partial edits: {history_path}

Read the evidence, then implement exactly one mechanism-level change in the editable source.
The candidate must generalize beyond the visible reproducer. Do not inspect hidden tests, audit
tasks, benchmark evaluator internals, gold answers, task ids, or held-out artifacts. Do not change
the solver model/provider/sampling configuration. Run the public test and an appropriate syntax or
import smoke check. It is acceptable for the original end-to-end failure to remain if this edit
repairs the claimed mechanism.

Write {proposal_path} with exactly this shape:
{json.dumps(proposal, indent=2, ensure_ascii=False)}

Do not modify files outside {source_dir} except {proposal_path} and local scratch files.
"""


def auditor_prompt(
    *,
    benchmark_context: str,
    incumbent_source: Path,
    candidate_source: Path,
    diff_path: Path,
    proposal_path: Path,
    output_dir: Path,
) -> str:
    return f"""You are the post-edit Auditor. You do not decide promotion and you do not repair code.

Benchmark contract:
{benchmark_context}

Incumbent source: {incumbent_source}
Candidate source: {candidate_source}
Source diff: {diff_path}
Candidate claim: {proposal_path}

Author hidden regression and anti-hardcoding tests under {output_dir}. The packet format and test
runtime are the same as the Experimentalist packet, but every target case must have tier
`regression` or `admission`, and must pass the incumbent. Tests receive TRACEUNIT_SOURCE. Focus on
side effects implied by the actual diff, preserved entry contracts, and implementation-independent
invariants. Do not access benchmark graders, gold data, held-out tasks, or network resources. Write
{output_dir / "test_packet.json"} with metadata.packet_kind set to `audit`, and run the tests
against the incumbent before finishing.
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
