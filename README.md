# TraceUnit

TraceUnit optimizes an agent harness with trace-conditioned unit tests. A unit
test is not a benchmark answer or a generic capability score: it is a frozen,
cheap behavioral check that a proposed patch actually repairs one
trace-diagnosed atomic problem, before the expensive search evaluation runs.

## Core loop

For each iteration:

1. Stage the incumbent's failed traces from the immutable search pool.
2. The Test Author reads the traces (and, in C3, the world model plus the
   previous iteration's raw outcome), diagnoses trace-supported hypotheses,
   and freezes a TestPacket before any edit exists.
3. The Candidate Editor - who also sees the failing traces, the current
   search score, the decision history, and the archived-candidate records -
   proposes one mechanism-level patch.
4. The inner unit loop: the controller runs the frozen suite plus every
   preserved contract host-side; on failure the concrete results go back to
   the editor for another attempt (up to `loop.max_inner_retries`). This is
   the cheap alignment step - seconds instead of a search run.
5. After the loop, the candidate is evaluated on the search pool and the
   five-cell decision applies (see below).
6. In C3, the next Test Author reads the previous iteration's raw evidence -
   especially any unit/search mismatch - and appends its own distill to an
   append-only world model before designing the next packet.

The final pool is sealed during search. It is opened only by
`traceunit final-evaluate`.

## The five-cell decision

| unit \ search | improved | flat | regressed |
| --- | --- | --- | --- |
| passed | promote | archive (credit-assignment gap?) | reject + mismatch |
| failed | archive + mismatch | reject | reject |

Promotion requires both a passed unit verdict (frozen contract, preserved
contracts, regressions) and a positive paired search delta. Archives are
records - diff plus record.json - staged to later editors as reference
material; re-litigation goes through the normal propose -> unit -> search
path. A mismatch means the unit tests and the search distribution disagreed;
the mismatch record (frozen tests, diff, per-task flips, traces) is handed to
the next Test Author to diagnose.

## Fixed L0 directions

`instruction`, `context`, `planning`, `retrieval`, `tool`, `state`,
`verification`, `recovery`, `termination`, `other`, and `uncertain`.

A packet has exactly one `primary_family`: a coarse diagnosis direction, not a
transfer claim. Each case has an `evidence_role`: `target_reproducer`,
`structural_sibling`, `downstream_bridge`, `positive_witness`,
`preservation_control`, or `off_target_control`. Capability scaffolds such as
debate or multi-agent review are `intervention_kind` values
(`capability_augmentation`, `orchestration_change`), not new families.

Tests should be grounded in real model behavior - a model-backed probe or a
replay of real trace structure - never a scripted fake client that branches on
prompt keywords.

## UT-design world model (C3)

One append-only markdown file (`ut_memory/world_model.md`), written by the
Test Author itself, WorldCalib style. The harness stages the file and the raw
previous-iteration evidence (decision, per-task paired flips, unit results,
mismatch records with traces) into the author's workspace and copies the file
back verbatim - no schema, no sanitization, no fallback text. A skipped
distill is recorded as an event, not papered over.

## Conditions

| Condition | Generated packets | Archive records staged | World model |
| --- | --- | --- | --- |
| C0 score-only | no | no | no |
| C1 raw TraceUnit | yes | no | no |
| C2 archive | yes | yes | no |
| C3 full | yes | yes | yes |

## Commands

```bash
traceunit validate-config --config configs/swebench_verified.yaml
traceunit prepare --config configs/swebench_verified.yaml
traceunit optimize --config configs/swebench_verified.yaml
traceunit final-evaluate --config configs/swebench_verified.yaml
```

See [docs/PROTOCOL.md](docs/PROTOCOL.md) for the normative behavior and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the experimental claims and
ablations.

## Memory benchmarks

TraceUnit supports `locomo` and `longmemeval` (`lme` is accepted as a
configuration alias) through WorldCalib's MemGPT-style memory scaffold and
evaluator:

```bash
traceunit validate-config --config configs/locomo.yaml
traceunit prepare --config configs/locomo.yaml
traceunit optimize --config configs/locomo.yaml

traceunit validate-config --config configs/longmemeval.yaml
traceunit prepare --config configs/longmemeval.yaml
traceunit optimize --config configs/longmemeval.yaml
```

The adapters reuse WorldCalib's canonical loaders, candidate scaffold, LoCoMo
token-F1 scorer, and LongMemEval LLM judge. TraceUnit still owns pool freezing:
LoCoMo pools are disjoint by complete conversation/sample, and LongMemEval pools
are content-hashed before search begins. Candidate code receives a redacted task
view (no answer, answer-session evidence, task id, or sample id), while gold
answers remain host-side for scoring. Retrieval hits are preserved as bounded
trace evidence for the Test Author. The sample configs point at the existing
`../Optimizer1/data/{locomo,longmemeval}` cache; set `benchmark.data_path` if
your data lives elsewhere.
