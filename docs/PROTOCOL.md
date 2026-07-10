# TraceUnit operational protocol

This document defines the information flow and state transitions implemented by
TraceUnit. It is normative for the full method. The experimental conditions
that remove parts of this protocol are defined separately in
[EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Objective

TraceUnit treats an agent harness edit as a causal intervention. The protocol
uses a frozen, pre-edit TestPacket to obtain mechanism-level evidence before it
spends a natural-task search evaluation. It preserves useful but globally
masked edits as reusable archive components and uses sparse natural-task
calibration observations to estimate which unit families transfer.

The protocol does not treat a unit pass as proof of natural-task
generalization. It separates:

- **local certificate:** immutable facts established by frozen unit,
  bridge, regression, and replay checks;
- **alignment prior:** versioned, uncertain, aggregate evidence that a unit
  family has transferred for earlier candidates;
- **final evidence:** a one-way seed-versus-terminal measurement on a pool that
  never participates in search or calibration.

## 2. Immutable benchmark plan

The benchmark adapter freezes a BenchmarkPlan before search. It contains:

- one search PoolSliceRef;
- zero or more ordered calibration shard references;
- one final PoolSliceRef;
- a plan hash derived from every slice identity and manifest hash.

The three roles are disjoint. Each evaluation verifies the selected manifest
hash and includes its slice identity in the evaluation cache fingerprint.

Clustering prevents correlated task variants from leaking across roles or
rotations:

- AppWorld uses the scenario identifier and never divides a scenario across
  search, calibration shards, or final;
- SWE-bench uses repository as the cluster, falling back to the individual
  instance when repository metadata is absent.

Limits are applied without cutting a cluster. A calibration shard may therefore
slightly exceed its requested size when a whole cluster is larger than the
remaining capacity.

## 3. Information boundaries

| Artifact | Controller | Test Author | Search Planner | Candidate Editor |
|---|---:|---:|---:|---:|
| Sanitized search traces | Yes | Yes | No | No |
| Incumbent source | Yes | Read-only copy | Read-only copy | Materialized editable copy |
| Public TestPacket and public test | Yes | Authored here | Yes | Yes |
| Hidden TestPacket cases | Yes | Authored here | No | No |
| Archive public catalog | Yes | No | Yes | Only selected materialization |
| Private calibration observations | Yes | No | No | No |
| Public calibration cards | Yes | Yes | Yes | No |
| Calibration task contents/outcomes | Yes | No | No | No |
| Final pool and report | Final runner only | No | No | No |

The Test Author and Search Planner receive separate, frozen input files for
each iteration. Both contain the same sanitized card payload plus an audience
marker. No role receives candidate identifiers, task identifiers, exact
calibration deltas, exact support counts, or per-task calibration outcomes.

## 4. Baseline

For a new run, the controller:

1. copies the benchmark seed source into the candidate store;
2. evaluates the seed on the search slice;
3. records its normalized traces, score, cost, and source lineage;
4. initializes the seed as the incumbent;
5. writes empty version-zero public calibration cards.

The baseline does not open a calibration shard or the final pool.

## 5. One search iteration

### 5.1 Freeze delayed card inputs

At the start of iteration t, the controller snapshots the current public
cards into:

    iterations/iter_NNN/inputs/test_author_cards.json
    iterations/iter_NNN/inputs/search_cards.json

Cards produced after iteration (t)'s decision are not visible until a later
iteration.

### 5.2 Author and admit a TestPacket

The Test Author receives failed search traces, a read-only incumbent snapshot,
benchmark constraints, and delayed cards. The packet is created before a
candidate edit or composition plan exists.

A normal packet must contain:

- at least two distinct trace-supported failure hypotheses;
- a target hypothesis that names at least one competing alternative;
- exactly one visible public reproducer contract;
- at least one hidden structural sibling;
- a positive-witness admission intervention;
- regression or admission controls that pass on the incumbent;
- a downstream bridge whenever the claimed mechanism can be continued without
  a benchmark grader;
- stable mechanism-level family identifiers without task, repository, or
  free-form hypothesis identifiers.

Generated tests cannot use network access, benchmark evaluators, ground truth,
gold/test patches, task identifiers, protected environment overrides, or
absolute host-data paths.

The packet is executed against the incumbent. Admission compares every result
with its declared incumbent expectation. Any missing or mismatched result makes
admission fail. An admitted packet is content-hashed and immutable. A packet may
be reused for a bounded number of attempts; it is retired after promotion,
archive, quarantine, challenge, or the configured attempt limit.

### 5.3 Freeze an autonomous composition plan

Before source editing, the Search Planner receives:

- the public packet;
- a read-only parent source;
- aggregate prior decisions;
- the public archive catalog;
- delayed calibration cards.

It selects zero, one, or any number of components. There is no fixed top-k
limit. The requested plan is normalized by the archive catalog:

- dependencies are inserted;
- dependencies precede dependants;
- duplicate or conflicting selections are rejected;
- cycles and missing references are rejected;
- the base source hash and application modes enter the attempt fingerprint.

Repeated attempts with the same parent, packet, and normalized plan are
rejected.

### 5.4 Materialize and edit

The controller copies the parent into an isolated staging tree.

- **Exact mode:** the archived patch hash is verified, patch applicability is
  checked, and the patch is applied before the Candidate Editor runs.
- **Semantic mode:** no patch is silently applied. The component and explicit
  porting instructions are recorded for the Candidate Editor.

The materialization receipt records the before/after source hashes and whether
each component was physically applied. The Candidate Editor may implement only
declared semantic ports, integration work, and the new mechanism edit. The
proposal must reference the frozen plan fingerprint and exactly the selected
component order.

### 5.5 Mechanical and unit evaluation

The controller first rejects:

- a source identical to its parent;
- a source that fails the benchmark smoke check;
- evaluator, task-specific, or policy violations;
- source symlinks that escape the candidate snapshot.

For a mechanically valid source it runs:

1. the current packet on incumbent and candidate;
2. optional post-edit regression-author checks;
3. every selected archive component's original frozen packet, including
   dependency and constituent certificates;
4. every cumulative preservation packet attached to promoted incumbents.

Missing, corrupted, or modified archived packets fail closed. Replays check the
packet's declared candidate contract, not merely whether a process exited.

The resulting evidence includes public gain, hidden gain, bridge gain,
regression loss, archive replay status, preservation status, unit-test wall
time, and family keys.

### 5.6 Conditional search evaluation

The deterministic decision policy examines frozen mechanical and unit evidence.
Only a candidate that satisfies admission, regression, replay, preservation,
public, and hidden thresholds receives a paired evaluation on the search pool.
The search delta is the mean matched-task candidate-minus-parent difference.

### 5.7 Decision table

Rules are evaluated from top to bottom.

| Condition | Decision | State effect |
|---|---|---|
| Mechanical violation | Reject | Keep incumbent |
| Packet admission below threshold | Challenge packet | Retire packet |
| Regression loss above threshold | Reject | Keep incumbent |
| Selected archive certificate replay fails | Reject | Keep incumbent |
| Cumulative promoted preservation replay fails | Reject | Keep incumbent |
| Public or hidden gain below threshold | Reject | Keep incumbent |
| Unit evidence passes but search delta is absent | Evaluate search | Internal, nonterminal action |
| Search delta exceeds the configured minimum | Promote | Candidate becomes incumbent; packet enters preservation set |
| Search is not positive, bridge passes, and search delta is within the noninferiority margin | Archive | Store content-addressed partial component |
| Bridge passes but search delta is below the noninferiority margin | Quarantine | Exclude from default reusable archive |
| No bridge-supported or search-supported propagation | Reject | Keep incumbent |

LLM roles propose tests, plans, and code; they never choose the terminal
decision.

### 5.8 Freeze, commit, and enqueue

The controller writes evidence and decision files before calibration. It then
commits the state effect and queues every candidate without a mechanical policy
violation as a CalibrationSubject. The subject binds:

- candidate and parent source paths;
- the frozen decision-file hash;
- family keys and unit profile;
- search/composition stratum;
- composition fingerprint;
- lineage and candidate identifiers kept private.

This ordering is the delayed-feedback guarantee: a candidate cannot use its own
calibration observation to change its already committed decision.

## 6. Partial archive semantics

An archive component is content-addressed and contains:

- an atomic or composite provenance kind;
- parent and candidate source hashes;
- patch path and patch hash;
- mechanism and target boundary;
- trace signature and applicability notes;
- dependency and constituent references;
- one or more frozen packet references;
- an immutable LocalCertificate containing public, hidden, bridge, and
  regression evidence.

The local certificate does not absorb later calibration results. Public
calibration cards remain separate, mutable priors. A future composition must
replay the original certificate regardless of how favorable its alignment card
looks.

When a composite is archived, its constituent identifiers remain provenance
obligations. Retrieving the composite therefore also retrieves the relevant
frozen packet closure for replay.

## 7. Delayed alignment calibration

### 7.1 Natural-transfer observation

For a frozen candidate-parent pair and one calibration shard, define the
matched-task difference as:

    d_j = score_j(candidate) - score_j(parent)

The paired delta is the mean of d_j. With at least two matched tasks, the
current uncertainty radius is:

    u = 1.96 * stdev(d) / sqrt(n)

For fewer than two matched tasks, the radius is conservatively set to 1.0.

The private label is:

- **positive** when the lower bound exceeds the configured positive effect;
- **negative** when the upper bound is below the negative noninferiority
  boundary;
- **noninferior** when the lower bound remains above that boundary;
- **inconclusive** otherwise.

This label belongs to the candidate-parent comparison. It does not prove that
every family tag in the packet caused the transfer.

### 7.2 Calibrate-then-freeze

The controller accumulates mechanically valid subjects until the configured
minimum cohort is available. The first checkpoint opens an initial calibration
shard, records paired observations, and publishes a new card version. That
version remains frozen until another checkpoint is triggered.

The checkpoint is content-bound to its subject cohort, decision hashes, prior
card version, iteration, epoch, and shard reference. A shard moves from sealed
to reserved and then spent. Spent shards cannot be selected by later
checkpoints.

### 7.3 Triggered rotating shards

After initial calibration, a fresh shard is opened only when there is a pending
cohort and at least one enabled trigger:

- an unseen unit family;
- high uncertainty for a relevant family/context;
- disagreement between unit and search signs;
- a previously unseen composition signature;
- pending-cohort capacity reached.

This is trigger-based rotation, not evaluation every fixed number of
iterations. If no trigger fires, cards and unopened shards remain unchanged.

### 7.4 Public cards

Private observations are aggregated by family key and unit profile with a
smoothed low-capacity model. One observation contributes total weight one
across all of its family tags, preventing packets with more tags from receiving
more statistical weight.

A public card contains only:

- family key and unit profile;
- transfer band: low, medium, or high;
- uncertainty band: low, medium, or high;
- coarse support bucket;
- status: supported, uncertain, or challenged;
- card version.

Candidates, lineages, shards, tasks, exact counts, exact posterior values,
paired deltas, and per-task outcomes remain private.

Cards may guide future test design and archive retrieval. They are never local
certificates and never directly override the deterministic decision table.

## 8. Final evaluation

Search writes **final_evaluation: not_opened** in its summary. Opening final is a
separate command.

The final runner requires a completed or converged search state and seals:

- the search-state hash;
- benchmark-plan hash;
- final pool reference;
- seed and terminal candidate identifiers;
- seed and terminal source hashes.

It refuses to replace an existing sealed plan with a different one and refuses
to evaluate a source that changed after sealing. The report contains seed
score, terminal score, matched-task paired delta, matched-task count, and final
cost.

The final-evaluation module has no calibration dependency. Its result cannot
update cards, thresholds, archive contents, prompts, or search decisions.

## 9. Resume and failure behavior

Run state, candidate inputs, plans, packet references, decisions, checkpoint
files, shard states, and source hashes are persisted. Re-running with resume
enabled reuses content-bound artifacts. Re-running with resume disabled against
an existing state is rejected.

The protocol fails closed on:

- changed frozen packet or pool hashes;
- invalid archive references or cycles;
- patch-check failures;
- source-policy or sandbox violations;
- missing candidate artifacts;
- a final source or plan that changed after sealing.

A failed Test Author attempt retires the active packet and advances the
iteration without modifying the incumbent.

## 10. Interpretation boundary

Passing admission means the packet is executable, reproduces its declared
incumbent behavior, contains hidden variation and controls, and has a
satisfiable positive witness. It does not establish that the target hypothesis
is uniquely causal.

Likewise, archive replay establishes preservation of declared local contracts,
not universal compatibility. The empirical burden remains:

- whether unit evidence improves edit-level credit assignment beyond natural
  search score alone;
- whether archived partial edits produce useful later compositions;
- whether delayed cards predict future natural transfer;
- whether the terminal system improves on the independent final pool.
