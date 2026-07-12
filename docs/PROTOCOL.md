# TraceUnit protocol

This document is normative for the current main experiment. It describes online
search-pool optimization, not an independent transfer-calibration experiment.

## 1. Immutable inputs

Before a run:

- freeze the packaged L0 ontology into the run directory;
- freeze one search manifest and one sealed final manifest, each with a content
  hash and cluster-disjoint membership;
- freeze the baseline source and run configuration;
- reject resume if configuration, capability flags, ontology, or plan hash
  differs.

The final manifest is never exposed to the Test Author, the Candidate Editor,
or ordinary search code.

## 2. L0 diagnosis

The canonical L0 values are:

~~~
instruction context planning retrieval tool state
verification recovery termination other uncertain
~~~

The Test Author may mention only trace-supported alternatives and must select
one primary_family for the packet. other and uncertain are valid honest
outcomes; neither creates a family score or forces a finer taxonomy.

A packet also records a free-text mechanism, target boundary, failure claim,
and intervention kind. These fields carry detail that must not be encoded by
inventing new family IDs.

## 3. Packet freeze

A normal packet contains:

- at least two competing trace-grounded hypotheses;
- exactly one target hypothesis and one packet-level primary_family;
- one public reproducer, one hidden structural sibling, a positive witness, and
  an off-target regression;
- a bridge when a downstream behavioral bridge can be represented;
- an evidence_role for every case.

Tests must be grounded in real model behavior: prefer a model-backed probe or a
replay of real trace structure over a scripted fake client that branches on
prompt keywords. A keyword-matching stub certifies string content, not
behavior, and its verdicts will not track the search distribution.

The controller runs the proposed packet on the incumbent. Every case must meet
its declared incumbent outcome. Only then is the packet content-addressed and
marked admission_passed=true. The packet is immutable after admission, and one
fresh packet is authored per iteration.

## 4. Candidate and the inner unit loop

The Candidate Editor receives the incumbent's failing search traces, the
current aggregate search score, the decision history, the archived-candidate
records, the public part of the frozen packet, and (in C3) a read-only copy of
the UT-design world model. It implements one general mechanism-level edit:

- local_repair;
- capability_augmentation, such as a red-team agent, debate, self-critique,
  retrieval component, or test writer;
- orchestration_change.

The frozen packet is the editor's cheap alignment check. After each proposed
patch the controller runs the full frozen suite plus every preserved contract
host-side; on failure the concrete results go back to the same editor for
another attempt, up to loop.max_inner_retries times. Public cases feed back
their captured output; hidden cases reveal only their declared description and
pass/fail, so the hidden tier keeps measuring generalization. When the loop
ends - pass or retries exhausted - the last attempt's unit evidence is the
authoritative unit verdict, and the candidate proceeds to paired search.

## 5. The five-cell decision

Every mechanically valid candidate is evaluated on the immutable search pool.
The decision is a pure function of the unit verdict (frozen contract,
preserved contracts, regressions) and the paired search delta:

| unit \ search | improved | flat (within margin) | regressed |
| --- | --- | --- | --- |
| passed | **promote** | **archive** (possible credit-assignment gap) | **reject + mismatch** |
| failed | **archive + mismatch** | reject | reject |

- Promote: the candidate becomes the incumbent and its packet becomes a
  preserved contract.
- Archive: the candidate is recorded (diff, record.json) for later agents to
  read and re-litigate; nothing replays or migrates it.
- Mismatch: the unit verdict and paired search disagreed. The controller
  writes mismatch/iter_NNN with the frozen packet, the candidate diff, the
  per-task paired flip table, and pointers to both sides' traces. The next
  Test Author must diagnose it before designing a new packet.

A search improvement without a passed contract is never a certified promotion.

## 6. Archives are records, not capabilities

There is one immutable frozen-packet store for **preserved** contracts (from
promoted candidates); every later candidate must keep satisfying them inside
the inner unit loop.

Archived candidates carry no protocol status. Their records are staged into
later editors' workspaces as reference material; an editor that finds an
archived idea valuable rebuilds it and takes it through the normal
propose -> unit -> search path. Nothing is replayed, realized, or migrated on
the candidate's behalf.

## 7. UT-design world model (C3)

One append-only markdown file, written by the Test Author itself:

~~~
world model + last_iteration.json (+ mismatch evidence)
    -> staged into the next Test Author's workspace
    -> the author reads the file first, appends `## iter_NNN distill`
    -> the harness copies the file back verbatim
~~~

The staged evidence is raw: the previous decision, the per-task paired search
flips, the unit results, and on a mismatch the frozen tests, the diff, and the
failed traces. The harness owns no schema, sanitization, or fallback text; a
skipped distill is recorded as a world_model_not_updated event. The world
model is guidance for designing later tests; it never overrides current trace
evidence and never ranks L0 directions.

## 8. Resume

decision.json and evidence.json are a commit boundary. If a process stops
after writing them, resume loads those artifacts, commits any missing state
effect, and advances the iteration. It does not rerun local tests or search
evaluation. Inside an iteration, packet_ref.json and inner_state.json make
packet authoring and the inner unit loop resumable.

## 9. Final evaluation

Final evaluation is a distinct sealed command. It consumes the final manifest
only after search finishes. Its result does not alter run state, the world
model, the packet store, or decisions.

## 10. Scope of claims

The main experiment can claim that trace-conditioned UT design, used as a
cheap inner alignment check with an online self-written world model, helps
search-pool optimization under the stated protocol. It cannot claim that a
particular L0 direction has intrinsically higher transfer value. Transfer
itself is measured only by the sealed final evaluation.
