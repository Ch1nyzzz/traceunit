# TraceUnit

TraceUnit optimizes an agent harness with a trace-conditioned **capability
battery**. The battery is a persistent, cheap proxy for the capabilities the
benchmark demands: each atomic capability is probed by several cross-domain
instances, so a patch moves a capability's score only by genuinely acquiring
the capability - never by naming one task's API in a prompt. Unit tests are
not reproductions of individual failing tasks.

## Core loop

For each iteration:

1. Stage the incumbent's failed traces from the immutable search pool.
2. The Test Author diagnoses the **root-cause atomic capability** behind the
   failures (first principles, not the surface of one task) and updates the
   battery: new cross-domain instances where coverage is thin, retirements
   where the calibration flags dead weight. On a cold start it clusters the
   baseline's failures into 4-6 capabilities and builds the initial battery.
   Every new instance is admitted against the incumbent.
3. The Candidate Editor - who sees the failing traces, the current search
   score, the decision history, the archived-candidate records, and the
   target capability's spec - proposes one mechanism-level patch.
4. The inner battery loop: the controller runs **every** battery instance;
   on failure (target not improved, or collateral damage to another
   capability) the concrete per-instance results go back to the editor for
   another attempt (up to `loop.max_inner_retries`). Seconds, not a search
   run.
5. After the loop, the candidate is evaluated on the search pool and the
   five-cell decision applies (see below).
6. The host appends a calibration row - per-capability battery deltas next
   to the paired search delta - and, in C3, the next Test Author reads the
   raw outcome (especially any battery/search mismatch) and appends its own
   distill to an append-only world model.

The final pool is sealed during search. `traceunit optimize` runs the sealed
final evaluation automatically once search completes (`--no-final` restores
the two-step flow).

## The five-cell decision

| unit \ search | improved | flat | regressed |
| --- | --- | --- | --- |
| passed | promote | archive (credit-assignment gap?) | reject + mismatch |
| failed | archive + mismatch | reject | reject |

The unit verdict is the battery: the diagnosed capability's pass rate
improved **and** no other capability dropped beyond
`decision.max_battery_regression`. On promote, the candidate's full-battery
results become the new incumbent reference. Archives are records - diff plus
record.json - staged to later editors as reference material; re-litigation
goes through the normal propose -> battery -> search path. A mismatch means
the battery and the search distribution disagreed; the mismatch record is
handed to the next Test Author to diagnose.

## Battery instances

An instance is a frozen single-case packet (`packet_kind: battery_instance`)
probing one capability at one decision boundary, executed in the same
sandbox as before (deterministic or live model probe). Hard rules:

- **Cross-domain**: never the search tasks' app names, APIs, entities, or
  literal values - fictional or re-skinned domains only. Sibling instances in
  a group differ in surface, not in mechanism.
- **Computed expectations**: probe patterns demand computed output over the
  injected observations (exact identifiers, quantities, exclusions), never a
  bare API-name regex that a verbal prompt reminder could elicit.
- Groups are capped (`loop.max_instances_per_capability`); the Test Author
  retires instances before adding beyond the cap.

Capabilities map to the frozen L0 registry (`instruction`, `context`,
`planning`, `retrieval`, `tool`, `state`, `verification`, `recovery`,
`termination`, `other`, `uncertain`) as their coarse family, with a freeform
capability slug per group.

## Calibration

`battery/calibration.jsonl` records, per search-evaluated candidate, the
per-capability battery deltas and the paired search delta. The host derives
the two statistics this data volume supports - per-capability direction
agreement ("when this group moved, did search move?") and constant,
information-free instances - and stages the table to the Test Author. It
informs attention and retirement; it never gates a decision.

## UT-design world model (C3)

One append-only markdown file (`ut_memory/world_model.md`), written by the
Test Author itself, WorldCalib style. The harness stages the file and the raw
previous-iteration evidence (decision, per-task paired flips, battery
results, mismatch records with traces) into the author's workspace and copies
the file back verbatim - no schema, no sanitization, no fallback text. A
skipped distill is recorded as an event, not papered over.

## Conditions

| Condition | Battery | Archive records staged | World model |
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
