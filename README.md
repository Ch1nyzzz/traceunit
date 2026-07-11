# TraceUnit
TraceUnit optimizes an agent harness with trace-conditioned unit tests. A unit
test is not a benchmark answer or a generic capability score: it is a frozen,
local behavioral contract for one trace-supported diagnosis.
The current implementation deliberately uses a small, fixed L0 vocabulary and
an online UT-design memory. It does **not** maintain family transfer cards,
calibration shards, or a proxy model in the main optimization loop.

## Core loop

For each iteration:

1. Stage failed traces from the current incumbent on the immutable search pool.
2. The Test Author selects only trace-supported L0 directions, chooses one
   primary direction, and freezes a TestPacket before any edit exists.
3. The Candidate Editor makes one local repair, capability augmentation, or
   orchestration change against that packet.
4. The controller runs the packet on incumbent and candidate, including public,
   hidden, bridge, witness, and regression cases.
5. Every mechanically valid candidate is evaluated on the search pool, even if
   its local contract fails. This produces the feedback needed to improve later
   UT design rather than selectively observing only already-good packets.
6. The controller promotes, archives, quarantines, or rejects the candidate.
7. In C3, the next Test Author reflects on a sanitized digest of the frozen
   packet, aggregate local evidence, and a categorical search outcome before
   designing its own packet; the harness commits the reflection into the world
   model it consumes.

The final pool is sealed during search. It is opened only by
`traceunit final-evaluate`.

## Fixed L0 directions

`instruction`, `context`, `planning`, `retrieval`, `tool`, `state`,
`verification`, `recovery`, `termination`, `other`, and `uncertain`.

A packet has exactly one `primary_family`. It is a coarse diagnosis direction,
not a claim that the direction itself transfers better than another direction.
Specific behavior remains in the trace-grounded hypothesis, mechanism claim,
target boundary, and frozen test implementation.

Each case instead has an `evidence_role`:

- `target_reproducer`
- `structural_sibling`
- `downstream_bridge`
- `positive_witness`
- `preservation_control`
- `off_target_control`

Capability scaffolds such as debate, a red-team test writer, multi-agent
review, or retrieval helpers are represented by `intervention_kind`
(`capability_augmentation` or `orchestration_change`), not by inventing a
new family.

## Frozen contracts and decisions

Admission is boolean: every case must match its declared incumbent outcome
before the packet is frozen.

Candidate evaluation keeps contract failure separate from regression loss:

- A failed target or hidden test means the frozen local contract was not met.
  It does not become a fake “regression loss.”
- A real regression is an incumbent-passing preservation/admission case that
  the candidate breaks.
- A mechanically valid candidate always receives a paired search evaluation.
- Promotion requires a passed frozen contract and positive paired search delta.
- Archival requires a passed contract, a passed bridge, and non-inferior search.
- A bridge-certified candidate whose search result regresses is quarantined.

This separation is important: a candidate that improves search while failing
its UT is useful evidence for improving future UT design, but is not promoted
as a certified repair.

## Latent capabilities (C2/C3)

There is a single frozen-packet store with two packet states. Packets from
promoted candidates are **preserved**: every later candidate must keep
satisfying them. Packets from archived candidates are **latent**: certified
but not yet promoted behavior contracts, kept with a reference patch.

Each later Candidate Editor sees the latent contracts and reference patches as
optional material. Every candidate evaluation replays all latent packets
observationally; a latent contract the candidate satisfies is *realized*, and
when that candidate is promoted the realized packets migrate into the
preserved set. Reuse is therefore measured by replaying frozen contracts, not
declared by the editor, and a candidate that realizes latent capabilities is
recorded with `attribution_scope=composition`; its search outcome never
updates an atomic L0 direction score.

## Online UT-design memory

C3 writes:

```
ut_memory/
  episodes.jsonl
  reflections/
  world_model.md
```

The world model contains recent, de-duplicated, sanitized recommendations. It
does not include search task content, task IDs, exact deltas, family rankings,
or final-pool data. It supports online optimization only. A prospective
held-out-UT alignment study is a future ablation, not a main-loop dependency.

## Conditions

| Condition | Generated packets | Archive | Online UT memory |
| --- | --- | --- | --- |
| C0 score-only | no | no | no |
| C1 raw TraceUnit | yes | no | no |
| C2 archive | yes | yes | no |
| C3 full | yes | yes | yes |

The sample configs use 12 iterations: below the original 20-iteration budget,
but long enough for C3 to receive repeated per-iteration design feedback.

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

TraceUnit now supports `locomo` and `longmemeval` (`lme` is accepted as a
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
