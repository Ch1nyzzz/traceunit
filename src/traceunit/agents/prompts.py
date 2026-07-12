from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceunit.models import TestPacket
from traceunit.ontology import prompt_definitions


def _live_model_block(target_api_env: str | None) -> str:
    if not target_api_env:
        return ""
    return (
        "\nYour workspace ships python3 and pytest, and the frozen target model is "
        "reachable for live experimentation: an OpenAI-compatible endpoint at "
        "$TRACEUNIT_TARGET_BASE_URL, model $TRACEUNIT_TARGET_MODEL, key in "
        f"${target_api_env} (the openai python package is installed). Call it freely "
        "while you work to probe real model behavior; keep experiments small. Frozen "
        "test cases must stay deterministic or declarative agent_probe JSON - they "
        "never call the model themselves.\n"
    )


def test_author_prompt(
    *,
    benchmark_context: str,
    trace_manifest: Path,
    incumbent_source: Path,
    output_dir: Path,
    iteration: int = 0,
    world_model_path: Path | None = None,
    last_iteration_path: Path | None = None,
    mismatch_path: Path | None = None,
    probes_supported: bool = False,
    target_api_env: str | None = None,
) -> str:
    example = {
        "packet_id": "iter001_verification_contract",
        "version": 1,
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
                "case_id": "downstream_bridge",
                "tier": "bridge",
                "evidence_role": "downstream_bridge",
                "execution_mode": "deterministic",
                "path": "tests/hidden/test_bridge.py",
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
    memory_lines = []
    if world_model_path is not None:
        memory_lines.append(f"- UT-design world model (append-only): {world_model_path}")
    if last_iteration_path is not None:
        memory_lines.append(
            f"- previous iteration outcome (decision, per-task paired search flips, "
            f"unit results): {last_iteration_path}"
        )
    if mismatch_path is not None:
        memory_lines.append(
            f"- unit/search MISMATCH evidence from the previous iteration: "
            f"{mismatch_path} (mismatch.json, the candidate diff, the frozen packet "
            f"under packet/, and the mismatch candidate's failed search traces under "
            f"candidate_traces/)"
        )
    memory_input = (
        "\n".join(memory_lines)
        if memory_lines
        else "- UT-design memory: disabled for this experiment condition"
    )
    if world_model_path is not None:
        distill_step = (
            f"FIRST, before designing anything: read the world model at "
            f"{world_model_path}."
        )
        if last_iteration_path is not None:
            distill_step += (
                f" Then study the previous iteration's outcome and append a new "
                f"`## iter_{iteration:03d} distill` section to that same file: what "
                f"the unit tests predicted, what paired search actually did per task, "
                f"why they agreed or disagreed, and what you will design differently "
                f"now. Never rewrite or delete prior entries."
            )
        if mismatch_path is not None:
            distill_step += (
                " The previous iteration was a MISMATCH: the unit verdict and paired "
                "search disagreed. Diagnose it properly before designing: read the "
                "frozen tests, the candidate diff, and the failed search traces, and "
                "name concretely what the tests measured that the search tasks do "
                "not (or the reverse). Your new packet must not repeat that gap."
            )
        memory_guidance = distill_step + (
            " Apply your own accumulated design rules; they never override the "
            "current trace evidence."
        )
    else:
        memory_guidance = "No UT-design memory is available in this condition."
    probe_guidance = (
        "For capability claims only a live model can witness, add a case with "
        "execution_mode='model_backed_probe', driver='agent_probe', strict "
        "max_model_calls and max_tokens (total prompt+completion budget), and point "
        "path at a declarative JSON file: {\"description\": \"...\", \"messages\": "
        "[{\"role\": \"system\"|\"user\"|\"assistant\", \"content\": \"...\"}, ...], "
        "\"expect\": [{\"kind\": \"regex\"|\"contains\", \"pattern\"/\"value\": "
        "\"...\", \"negate\": false}]}. Message content may inline subject source "
        "files with {{source_file:relative/path}} so the probe measures the edited "
        "scaffold rather than the bare model; scripted assistant turns let one live "
        "completion test multi-turn behavior. The host renders the file, sends one "
        "temperature-0 completion, and requires every expectation to hold on the "
        "reply. The messages must end with a user or system turn."
        if probes_supported
        else "This benchmark does not support model-backed probes: every case must "
        "use execution_mode='deterministic'."
    )
    return f"""You are the Test Author in a trace-conditioned optimization protocol.

Author a causal TestPacket before any candidate edit exists. Diagnose at least two
trace-supported hypotheses, choose one, and distinguish it from the alternatives. The tests
are a cheap proxy for the search distribution: a later patch that satisfies them should be a
patch that actually repairs the diagnosed failure on real tasks. They measure agent policy
behavior; they must not solve or grade benchmark tasks.

{memory_guidance}

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

Set packet.primary_family to the selected target hypothesis family. Keep the
specific mechanism in mechanism, claim, target_boundary, and the tests. Do not put family labels
on individual cases.
{_live_model_block(target_api_env)}

Write under {output_dir}:
- test_packet.json
- tests/public/* for exactly one visible reproducer
- tests/hidden/* for structural siblings, bridge probes, admission checks, and regressions

Directories and tiers are independent: only the public reproducer lives under tests/public/,
every other file lives under tests/hidden/, and each case keeps its own tier. Every tier pairs
with exactly one evidence_role: public=target_reproducer, hidden=structural_sibling,
bridge=downstream_bridge, admission=positive_witness, regression=preservation_control or
off_target_control. No other evidence_role values exist. Keep status="proposed" and
content_sha256="" exactly as in the template; the harness freezes and hashes the packet after
admission.

Deterministic tests receive TRACEUNIT_SOURCE, TRACEUNIT_TEST_BUNDLE, and TRACEUNIT_SUBJECT. Use
'python' or 'pytest'; do not use network, evaluator APIs, gold data, held-out artifacts, or task
ids. Include a public target, a structurally varied hidden target, a positive-witness
intervention, and an off-target regression. Add a downstream bridge whenever it can be represented
without a grader. {probe_guidance} Generated code must never call a model API or access
credentials.
Prefer mutation-based contracts that test counterexample discovery, critique adoption, and final
correction rather than merely checking that a critic/debate component exists. Ground each test in
real model behavior: prefer a model-backed probe or a replay of real trace structure over a
scripted fake client that branches on keywords in the prompt - a keyword-matching stub certifies
string content, not behavior, and its verdicts will not track the search distribution. Run every
supported test against the incumbent before finishing. Do not edit the incumbent.

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
    proposal_path: Path,
    trace_manifest: Path | None = None,
    incumbent_search_score: float | None = None,
    history_path: Path | None = None,
    archives_path: Path | None = None,
    world_model_path: Path | None = None,
    target_api_env: str | None = None,
) -> str:
    proposal = {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "hypothesis_id": "h1",
        "intervention_kind": "capability_augmentation",
        "mechanism_claim": "falsifiable mechanism-level change",
        "predicted_effect": "expected unit/search effect",
        "regression_risks": ["behavior that could regress"],
        "metadata": {"notes": ""},
    }
    input_lines = [
        f"- editable source (a copy of the incumbent): {source_dir}",
        f"- public frozen TestPacket: {public_packet_path}",
    ]
    if trace_manifest is not None:
        input_lines.append(
            f"- failing search traces of the incumbent: {trace_manifest} "
            "(failing traces worst-first and passing traces best-first, flagged "
            "by \"passed\"; read the failures the packet diagnoses in depth)"
        )
    if incumbent_search_score is not None:
        input_lines.append(
            f"- current aggregate search score: {incumbent_search_score}"
        )
    if history_path is not None:
        input_lines.append(
            f"- prior decisions and search deltas: {history_path}"
        )
    if archives_path is not None:
        input_lines.append(
            f"- archived earlier candidates: {archives_path} (each record is an "
            "earlier edit worth reading: its unit contract passed while paired "
            "search stayed flat, or its paired search improved while its unit "
            "contract failed; rebuild what you judge valuable, never apply a "
            "diff blindly)"
        )
    if world_model_path is not None:
        input_lines.append(
            f"- UT-design world model (read-only context): {world_model_path}"
        )
    inputs = "\n".join(input_lines)
    return f"""You are the Candidate Editor in a trace-conditioned optimization protocol.

Benchmark contract:
{benchmark_context}

Inputs:
{inputs}

Diagnose the failing traces and implement one general mechanism-level edit that repairs
the failure the frozen TestPacket pins down. The packet is your cheap alignment check:
after you finish, the harness runs the full frozen unit suite (plus every previously
promoted contract) and, on failure, hands the concrete results back to you for another
attempt - all before the expensive search evaluation. Promotion is decided by paired
search on real tasks, so repair the mechanism; an edit that merely satisfies the tests
without changing real behavior earns nothing.

Generalize beyond the visible reproducer. Do not access hidden test files, benchmark
evaluators, gold data, held-out pools, or final tasks. Run the public test and a
syntax/import smoke check before finishing; do not submit an edit whose public test
still fails.
{_live_model_block(target_api_env)}

Write {proposal_path}:
{json.dumps(proposal, indent=2, ensure_ascii=False)}
Copy hypothesis_id and intervention_kind verbatim from the public packet's target
hypothesis; intervention_kind must be exactly one of local_repair,
capability_augmentation, or orchestration_change, never free text.
"""


def candidate_retry_prompt(
    *,
    attempt: int,
    max_attempts: int,
    source_dir: Path,
    public_packet_path: Path,
    feedback_path: Path,
    proposal_path: Path,
) -> str:
    return f"""You are the Candidate Editor, continuing your own previous attempt.

Your last edit did not satisfy the frozen unit tests. This is attempt {attempt} of
{max_attempts} in the cheap unit loop; the expensive search evaluation only runs after
this loop, so use it.

Concrete failures: {feedback_path}
Editable source (already contains your previous edit): {source_dir}
Public frozen TestPacket: {public_packet_path}

Read the feedback first. Public cases include their captured output; hidden cases show
only their declared behavior description and pass/fail state. A failed preserved
contract means your edit broke behavior an earlier promoted candidate certified -
integrate with it instead of fighting it. Refine your edit, or revert and take a
different approach to the same mechanism; do not weaken or game the tests. Run the
public test and a syntax/import smoke check before finishing.

Update {proposal_path} if your mechanism changed; keep candidate_id, parent_id,
hypothesis_id, and intervention_kind unchanged.
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
    target_api_env: str | None = None,
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
{_live_model_block(target_api_env)}

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
