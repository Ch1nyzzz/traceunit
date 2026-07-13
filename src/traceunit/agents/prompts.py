from __future__ import annotations

import json
from pathlib import Path

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


_INSTANCE_PACKET_EXAMPLE = {
    "packet_id": "evidence-before-mutation-inventory",
    "version": 1,
    "hypotheses": [
        {
            "hypothesis_id": "h1",
            "family": "verification",
            "intervention_kind": "capability_augmentation",
            "mechanism": "irreversible actions proceed without complete evidence",
            "target_boundary": "the decision turn before the first mutation",
            "claim": "the policy mutates before observing every documented source",
            "evidence_trace_ids": ["trace-id"],
            "confidence": 0.7,
        }
    ],
    "target_hypothesis_id": "h1",
    "primary_family": "verification",
    "public_contract": "observe every documented source before mutating",
    "hidden_variant_strategy": "cross-domain sibling instances in the same group",
    "cases": [
        {
            "case_id": "probe",
            "tier": "public",
            "evidence_role": "target_reproducer",
            "execution_mode": "model_backed_probe",
            "driver": "agent_probe",
            "path": "tests/public/probe.json",
            "max_model_calls": 1,
            "max_tokens": 4096,
            "expected_incumbent_pass": False,
            "expected_candidate_pass": True,
        }
    ],
    "status": "proposed",
    "admission_passed": False,
    "content_sha256": "",
    "metadata": {"packet_kind": "battery_instance"},
}

_BATTERY_UPDATE_EXAMPLE = {
    "target_capability": "evidence-before-mutation",
    "target_family": "verification",
    "capability_descriptions": {
        "evidence-before-mutation": (
            "before any irreversible action, the policy observes every "
            "documented source the instruction implicates"
        )
    },
    "new_instances": [
        {
            "instance_id": "ebm-inventory-delete",
            "capability": "evidence-before-mutation",
            "family": "verification",
            "description": (
                "fictional inventory app: two of three record categories "
                "listed; a delete-all request must fetch the third first"
            ),
            "expected_incumbent_pass": False,
            "bundle": "instances/ebm-inventory-delete",
        }
    ],
    "retire_instance_ids": [],
}


def battery_author_prompt(
    *,
    benchmark_context: str,
    trace_manifest: Path,
    incumbent_source: Path,
    battery_state_path: Path | None,
    calibration_path: Path | None,
    cold_start: bool,
    max_instances_per_capability: int,
    output_dir: Path,
    iteration: int = 0,
    world_model_path: Path | None = None,
    last_iteration_path: Path | None = None,
    mismatch_path: Path | None = None,
    probes_supported: bool = False,
    target_api_env: str | None = None,
) -> str:
    memory_lines = []
    if world_model_path is not None:
        memory_lines.append(
            f"- UT-design world model (append-only): {world_model_path}"
        )
    if last_iteration_path is not None:
        memory_lines.append(
            f"- previous iteration outcome (decision, per-task paired search "
            f"flips, battery results): {last_iteration_path}"
        )
    if mismatch_path is not None:
        memory_lines.append(
            f"- battery/search MISMATCH evidence from the previous iteration: "
            f"{mismatch_path} (mismatch.json, the candidate diff, and the "
            f"mismatch candidate's failed search traces under candidate_traces/)"
        )
    if battery_state_path is not None:
        memory_lines.append(
            f"- current battery state (groups, instances, incumbent scores): "
            f"{battery_state_path}"
        )
    if calibration_path is not None:
        memory_lines.append(
            f"- host-computed calibration (which capabilities' battery movement "
            f"predicted search, which instances carry no information): "
            f"{calibration_path}"
        )
    memory_input = "\n".join(memory_lines)
    if world_model_path is not None:
        distill_step = (
            f"FIRST, before designing anything: read the world model at "
            f"{world_model_path}."
        )
        if last_iteration_path is not None:
            distill_step += (
                f" Then study the previous iteration's outcome and append a new "
                f"`## iter_{iteration:03d} distill` section to that same file: what "
                f"the battery predicted, what paired search actually did per task, "
                f"why they agreed or disagreed, and what you will design differently "
                f"now. Never rewrite or delete prior entries."
            )
        if mismatch_path is not None:
            distill_step += (
                " The previous iteration was a MISMATCH: the battery verdict and "
                "paired search disagreed. Diagnose it properly before designing: "
                "read the failing instances, the candidate diff, and the failed "
                "search traces, and name concretely what the battery measured "
                "that the search tasks do not (or the reverse). Your update must "
                "not repeat that gap."
            )
        memory_guidance = distill_step + (
            " Apply your own accumulated design rules; they never override the "
            "current trace evidence."
        )
    else:
        memory_guidance = "No UT-design memory is available in this condition."
    probe_guidance = (
        "Prefer execution_mode='model_backed_probe' with driver='agent_probe' and a "
        "declarative JSON file: {\"description\": \"...\", \"messages\": [{\"role\": "
        "\"system\"|\"user\"|\"assistant\", \"content\": \"...\"}, ...], \"expect\": "
        "[{\"kind\": \"regex\"|\"contains\", \"pattern\"/\"value\": \"...\", "
        "\"negate\": false}]}. Message content may inline subject source files with "
        "{{source_file:relative/path}}; the host sends one temperature-0 completion "
        "and requires every expectation to hold. Expectations must demand computed "
        "output over the injected observations (exact identifiers, quantities, "
        "exclusions) - never a bare API-name regex that a verbal prompt reminder "
        "could elicit."
        if probes_supported
        else "This benchmark does not support model-backed probes: every case must "
        "use execution_mode='deterministic' with driver 'python' or 'pytest'."
    )
    mission = (
        (
            "The battery is empty: this is the cold start. Cluster the incumbent's "
            "failing traces into 4-6 root-cause atomic capabilities and build the "
            "initial battery: 3-4 instances per capability, every instance a "
            "different surface (different fictional domain, entities, and API "
            "names) of the same mechanism. Then pick the target_capability whose "
            "repair the next candidate should attempt."
        )
        if cold_start
        else (
            "Diagnose the root-cause atomic capability behind the current failing "
            "traces - the first-principles deficit, not the surface of one task. "
            "Choose it as target_capability (an existing group or a new one), and "
            "update the battery: add cross-domain instances where coverage is "
            "thin, retire instances the calibration flags as uninformative or "
            "misleading. After your update the target group must contain at least "
            "one active instance the incumbent fails."
        )
    )
    return f"""You are the Test Author. You maintain the capability battery: a persistent,
cheap proxy for the capabilities this benchmark demands. A battery instance is NOT a
reproduction of one failing task - it probes one atomic capability at one decision
boundary in a domain of its own. A capability group holds several such instances with
different surfaces, so a patch moves the group only by genuinely acquiring the
capability, never by naming one task's API in a prompt.

{memory_guidance}

{mission}

Hard rules for instances:
- One instance = one frozen single-case packet bundle under
  {output_dir}/instances/<instance_id>/ with metadata.packet_kind="battery_instance",
  exactly one public case, and primary_family set to the capability's L0 family.
  The public case always carries tier="public" and evidence_role="target_reproducer"
  verbatim - these are fixed schema values, never invent alternatives.
- Cross-domain: never reuse the search tasks' app names, API names, entities, or
  literal values. Re-skin the mechanism into a fictional domain or a structurally
  different real one. Sibling instances in a group must differ in surface, not in
  mechanism.
- {probe_guidance}
- Declare expected_incumbent_pass honestly per instance; the host runs every new
  instance against the incumbent and rejects the whole update on a mismatch.
- Groups are capped at {max_instances_per_capability} active instances; retire
  before adding beyond the cap. Do not edit or delete existing instances on disk.

Choose family only from the frozen L0 registry:
{prompt_definitions()}

Benchmark contract:
{benchmark_context}

Inputs:
- normalized trace evidence: {trace_manifest} (failing traces worst-first and passing
  traces best-first, flagged by "passed"; skim the manifest and choose which traces to
  read in depth, using passing traces as behavioral contrast)
- incumbent source: {incumbent_source}
{memory_input}
{_live_model_block(target_api_env)}

Write under {output_dir}:
- battery_update.json
- instances/<instance_id>/test_packet.json and instances/<instance_id>/tests/public/*

battery_update.json shape:
{json.dumps(_BATTERY_UPDATE_EXAMPLE, indent=2, ensure_ascii=False)}

Instance packet shape (keep status="proposed" and content_sha256=""; the harness
freezes and hashes after admission):
{json.dumps(_INSTANCE_PACKET_EXAMPLE, indent=2, ensure_ascii=False)}

Deterministic tests receive TRACEUNIT_SOURCE, TRACEUNIT_TEST_BUNDLE, and
TRACEUNIT_SUBJECT; do not use network, evaluator APIs, gold data, held-out artifacts,
or task ids. Generated code must never call a model API or access credentials. Run
every supported instance against the incumbent before finishing. Do not edit the
incumbent.
"""


def candidate_edit_prompt(
    *,
    benchmark_context: str,
    candidate_id: str,
    parent_id: str,
    source_dir: Path,
    target_capability_path: Path,
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
        "hypothesis_id": "target",
        "intervention_kind": "capability_augmentation",
        "mechanism_claim": "falsifiable mechanism-level change",
        "predicted_effect": "expected battery/search effect",
        "regression_risks": ["behavior that could regress"],
        "metadata": {"notes": ""},
    }
    input_lines = [
        f"- editable source (a copy of the incumbent): {source_dir}",
        f"- target capability (the diagnosed deficit and its battery group): "
        f"{target_capability_path}",
    ]
    if trace_manifest is not None:
        input_lines.append(
            f"- failing search traces of the incumbent: {trace_manifest} "
            "(failing traces worst-first and passing traces best-first, flagged "
            "by \"passed\"; read the failures behind the diagnosis in depth)"
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
            "earlier edit worth reading; rebuild what you judge valuable, never "
            "apply a diff blindly)"
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

Diagnose the failing traces and implement one general mechanism-level edit that
repairs the diagnosed capability deficit. The capability battery is your cheap
alignment check: after you finish, the harness runs every battery instance - the
target capability's group plus every other capability - and hands failures back to
you for another attempt, all before the expensive search evaluation. The battery
instances live in domains other than the benchmark's, so a prompt sentence naming
one task's API moves nothing: repair the policy itself. A rule that fires outside
its intended context damages the other capability groups and fails the verdict, so
prefer scoped, minimal mechanisms over broad prompt additions. Promotion is decided
by paired search on real tasks.

Do not access the battery's probe files, benchmark evaluators, gold data, held-out
pools, or final tasks. Run a syntax/import smoke check before finishing.
{_live_model_block(target_api_env)}

Write {proposal_path}:
{json.dumps(proposal, indent=2, ensure_ascii=False)}
"""


def candidate_retry_prompt(
    *,
    attempt: int,
    max_attempts: int,
    source_dir: Path,
    target_capability_path: Path,
    feedback_path: Path,
    proposal_path: Path,
) -> str:
    return f"""You are the Candidate Editor, continuing your own previous attempt.

Your last edit did not satisfy the capability battery. This is attempt {attempt} of
{max_attempts} in the cheap battery loop; the expensive search evaluation only runs
after this loop, so use it.

Concrete results: {feedback_path}
Editable source (already contains your previous edit): {source_dir}
Target capability: {target_capability_path}

Read the feedback first. target_instances shows how each variant of the diagnosed
capability behaved; damaged_capabilities lists instances of OTHER capabilities your
edit broke - that is collateral damage from a rule firing outside its context, and
scoping your mechanism is usually the fix. Refine your edit, or revert and take a
different approach to the same capability; do not weaken or game the battery. Run a
syntax/import smoke check before finishing.

Update {proposal_path} if your mechanism changed; keep candidate_id and parent_id
unchanged.
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
