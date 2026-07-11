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

The final manifest is never exposed to Test Author, Candidate Editor, UT
Critic, or ordinary search code.

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

The controller runs the proposed packet on the incumbent. Every case must meet
its declared incumbent outcome. Only then is the packet content-addressed and
marked admission_passed=true.

The packet is immutable after admission. Case roles are not L0 labels.

## 4. Candidate and local evidence

The candidate may be:

- local_repair;
- capability_augmentation, such as a red-team agent, debate, self-critique,
  retrieval component, or test writer;
- orchestration_change.

The Candidate Editor additionally receives the latent capabilities (see §6):
public contract, mechanism, and a reference patch for each. It may realize any
of them alongside the new mechanism edit; there is nothing to declare, because
realization is measured afterwards by replay.

Generated deterministic tests run in an isolated snapshot. Model-backed probes,
when enabled by an adapter, are host-controlled declarative probes with fixed
call, token, and repetition budgets.

The controller records separately:

- the full frozen candidate contract;
- bridge-contract status;
- real preservation/regression loss;
- preserved-packet replay (gating) and latent-packet replay (observational);
- public, hidden, and bridge gains as descriptive measurements.

A target-contract failure must not be converted into regression loss.

## 5. Search outcome and decision

After smoke/policy checks, every mechanically valid candidate is evaluated
against the immutable search pool. The Test Author did not see current
candidate search outcomes at packet freeze.

The decision order is:

1. Reject a real regression or a failed preservation replay.
2. Reject a mechanically valid candidate missing paired search evidence.
3. Promote only when the full frozen contract passes and paired search improves.
4. Archive a non-promoted candidate only when its full contract and bridge pass
   and paired search is non-inferior.
5. Quarantine a bridge-certified candidate whose paired search regresses.
6. Reject the remaining candidates.

Search improvement without contract satisfaction remains logged but is not a
certified promotion.

## 6. Latent capabilities

There is one immutable frozen-packet store. Each retained packet is in one of
two states:

- **preserved** — from a promoted candidate. Every later candidate must keep
  satisfying it; a failure is a rejection.
- **latent** — from an archived candidate. It is a certified but not yet
  promoted behavior contract, stored with a reference patch from the source
  tree it was written against. Latent replay never gates a decision.

Every candidate evaluation replays all latent packets observationally. A latent
packet whose candidate contract passes is **realized** by that candidate. When
the candidate is promoted, its realized latent packets migrate into the
preserved set; unrealized ones stay latent. Reuse is therefore a measured
behavioral fact — a replayed frozen contract — never an unverifiable claim
about which patch was ported.

A candidate that realizes one or more latent packets records
attribution_scope=composition, and its search outcome belongs to the complete
edit; it must not update a purported atomic L0 effect. Factorial ablation is
required before making any per-capability credit claim.

## 7. Online UT-design memory

C3 reflects after each completed candidate-parent search comparison:

~~~
frozen packet + aggregate local evidence + categorical search outcome
    -> sanitized UT Critic recommendation
    -> append-only episode
    -> chronological world_model.md
    -> next Test Author
~~~

The critic receives no task IDs, task content, exact paired delta, final data,
or L0 ranking. The world model is guidance for writing later tests; it cannot
override current trace evidence.

Episodes are idempotent by (candidate_id, iteration). The world model keeps
recent deduplicated lessons by iteration, not lexicographic ordering.

## 8. Resume

decision.json and evidence.json are a commit boundary. If a process stops
after writing them, resume loads those artifacts, commits any missing state
effect, applies an idempotent memory reflection, and advances the iteration.
It does not rerun local tests or search evaluation.

## 9. Final evaluation

Final evaluation is a distinct sealed command. It consumes the final manifest
only after search finishes. Its result does not alter run state, UT memory,
the packet store, or decisions.

## 10. Scope of claims

The main experiment can claim that online, trace-conditioned UT design helps
search-pool optimization under the stated protocol. It cannot claim that a
particular L0 direction has intrinsically higher transfer value, nor that the
world model is independently calibrated on held-out tasks. Those require
separate prospective ablations.