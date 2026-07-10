# TraceUnit experiment specification

This document defines the paper-facing experiment matrix. It separates the
scientific claims, the information available to each condition, and the final
evaluation procedure.

The four main conditions are executable with `traceunit optimize` by setting
`protocol.condition` to `c0_score_only`, `c1_raw_traceunit`, `c2_archive`, or
`c3_full`. There is no batch matrix launcher; scheduling benchmark × condition ×
replicate runs remains external. A run is valid only when its prohibited artifacts
and feedback paths are absent.

## 1. Claims

### Primary claim: edit-level credit assignment

Agent-authored unit tests conditioned on the incumbent's traces and search
history provide useful mechanism-level supervision beyond a sparse natural-task
score. They should distinguish locally meaningful edits, including edits whose
end-to-end effect is initially masked.

The primary analysis must therefore operate at the candidate-edit level, not
only compare terminal benchmark scores.

### Archive claim

Unit- and bridge-certified partial edits can be retained, retrieved, composed,
and revalidated. This should recover useful multi-edit trajectories that a
strict positive-score hill climber would discard.

### Alignment claim

Sparse candidate-parent observations on natural calibration tasks can estimate
which unit families transfer. Delayed family-level cards should improve future
test construction and archive retrieval without exposing task-level
calibration evidence.

### Final generalization claim

The selected terminal agent should improve over the fixed baseline on a completely
unseen, in-distribution final pool. Calibration results cannot support this
claim because they influence later search.

## 2. Experimental unit

The experimental unit is:

    benchmark × condition × replicate

A replicate is an independent rerun of a potentially stochastic optimization
procedure. It uses the same baseline harness and frozen BenchmarkPlan as the
other conditions. It is not a different baseline or a different pool split. If
the complete procedure is deterministic, identical reruns add no information.

Each paired comparison must share:

- benchmark version and immutable BenchmarkPlan;
- search, calibration-shard, and final manifest hashes;
- target model and sampling/runtime settings;
- baseline source;
- maximum search opportunities and convergence rule;
- task-cluster policy;
- final-evaluation procedure.

AppWorld uncertainty and bootstrap units are scenarios. SWE-bench units are
repositories when repository metadata is available and instances otherwise.

## 3. Four main conditions

| ID | Condition | Generated TestPackets | Partial archive and 0..N composition | Delayed calibration cards |
|---|---|---:|---:|---:|
| C0 | Meta-Harness-style score-only | No | No | No |
| C1 | Raw TraceUnit | Yes | No | No |
| C2 | TraceUnit plus certified partial archive | Yes | Yes | No |
| C3 | Full TraceUnit | Yes | Yes | Yes |

### C0: Meta-Harness-style score-only

The Search Agent iterates from search-pool score, trace, and history. There is
no generated TestPacket, hidden sibling, bridge certificate, partial archive,
composition replay, or calibration feedback. Candidate selection uses the
natural search objective only.

This condition answers whether the additional TraceUnit machinery improves on a
straightforward train/search optimization loop.

### C1: Raw TraceUnit

The Test Author creates pre-edit frozen packets and the deterministic unit gate
is active. Partial candidates are not reusable, the archive catalog presented
to search is empty, and no calibration shard affects search.

This condition isolates the credit-assignment value of trace-conditioned unit
evidence.

### C2: TraceUnit plus certified partial archive

C1 is extended with content-addressed partial components, autonomous selection
of zero to any number of components, exact or explicit semantic application,
and replay of every selected component's original packet. Calibration cards
remain unavailable.

The C2 versus C1 contrast isolates the value of retaining and composing partial
edits.

### C3: Full TraceUnit

C2 is extended with delayed calibrate-then-freeze and triggered rotation over
fresh calibration shards. Aggregate public cards are available to both the
future Test Author and Search Planner.

The C3 versus C2 contrast isolates the value of alignment calibration.

### Implementation contract

The example configurations select C3. Condition capabilities are derived from
the enum rather than independent booleans, and the run state binds both the
condition and its capability manifest. In particular:

- C0 requires a score-only search implementation, not merely empty cards;
- C1 must prevent both archive retrieval and archive-derived replay;
- C2 must never consume a calibration shard during search;
- C3 must preserve delayed feedback and single-use shard semantics.

## 4. Supplementary equal-information control

Add a direct natural-validation gate as a supplementary control with the same
number of calibration task calls as C3. In this control, the current
candidate's natural calibration result may directly affect its own promotion.

This is not the proposed method: it intentionally violates delayed feedback.
Its purpose is to test whether C3's gain comes merely from purchasing extra
natural labels. Report it separately from the four main conditions.

## 5. Pool use and opening order

### Search

The search pool may be evaluated repeatedly according to the condition. Its
traces and aggregate outcomes may enter future search history.

### Calibration

Only C3 consumes calibration shards adaptively during search. A candidate's
decision is written and committed before it enters a checkpoint. Each opened
shard is single-use and can update only later card snapshots.

For candidate-level research analysis in C0-C2, optional calibration labels may
be collected only after those runs have frozen all search decisions. Such
labels are nonadaptive measurements, must not be copied into any workspace, and
must be reported separately from the method's online cost.

### Final

Do not open final while any condition, planned replicate, threshold choice, or terminal
selection can still change. The recommended order is:

1. finish every search run;
2. freeze terminal candidate identifiers and source hashes;
3. freeze analysis code, exclusions, and primary metrics;
4. seal a final plan for every run;
5. execute final evaluation;
6. do not launch replacement search runs based on final results.

The final pool is not a source of candidate-level calibration labels.

## 6. Candidate-level credit-assignment analysis

Let U_i contain only evidence available from the frozen packet and replay:

- public gain;
- hidden gain;
- bridge gain and bridge presence;
- regression loss;
- archive replay and preservation status;
- packet family/context;
- composition indicator;
- edit metadata fixed before a natural transfer label.

Let S_i be visible search evidence, including a missingness indicator when a
unit-negative edit never receives full search evaluation. Let Y_i be a
paired natural-transfer label from a calibration observation collected after
the candidate decision.

The central statistical comparison is:

    baseline model: Y_i <- S_i
    proxy model:    Y_i <- (S_i, U_i)

Use a low-capacity, predeclared model and evaluate out of sample. Preferred
splits are leave-one-replicate-out or leave-one-lineage-out; do not train and score
on the same candidate cohort. Report:

- log loss and Brier score for positive natural transfer;
- the improvement from adding unit evidence;
- precision and recall of unit-positive predictions;
- false-positive rate: unit positive but natural negative;
- false-negative rate: unit negative but natural positive;
- coverage and outcome of the archive-eligible stratum.

Do not evaluate only unit-positive candidates. Mechanically valid unit-negative
edits are necessary to estimate false negatives and avoid selection bias.

The strongest evidence for the primary claim is not simply a correlation
between unit and natural scores. It is an out-of-sample improvement in
candidate-level prediction or selection after conditioning on visible search
evidence.

The implemented offline analysis is:

    traceunit analyze-proxy \
      --run-dir runs/<run-a> \
      --run-dir runs/<run-b> \
      --output runs/proxy_analysis.json

Every prediction is produced by a regularized logistic model trained without
the prediction's run lineage. Categorical family and unit-profile
vocabularies are also fitted inside each training fold. Inconclusive transfer
labels are excluded from the binary positive-transfer target but remain in the
reported label coverage table. The report contains four complementary views:

1. an OOF proxy-alignment reliability curve;
2. incumbent search score by optimization iteration;
3. incumbent search score by cumulative search-plus-calibration natural-task
   cost;
4. the sealed final baseline-versus-terminal paired outcome, when available.

The first three use search/calibration artifacts. Final results never enter
proxy fitting, predictions, thresholds, or search curves. An already sealed
final report is copied only into the terminal-outcome section; the analysis
command never launches final evaluation.

### 6.1 Selective natural-task evaluation

Once unit evidence has demonstrated prospective alignment, it may be used as a
cheap first stage in a multi-fidelity evaluation cascade. A candidate below a
frozen proxy threshold can skip the complete search-pool evaluation, while
promising or uncertain candidates advance to more expensive natural-task
evidence. This saves target-system calls only if positive-transfer recall
remains acceptable.

The proxy report therefore includes a selective full-evaluation curve. For each
predeclared threshold it reports the full-evaluation rate, avoided-evaluation
rate, positive recall, strict-negative skip rate, and the expected rates after a
random audit of skipped candidates. The audit is required to estimate false
negatives and detect proxy drift; evaluating only proxy-positive candidates
would make the gate appear safer than it is.

Progressive evaluation allocates test budget; it does not disclose hidden test
contents or final-pool evidence to an agent. A threshold chosen after inspecting
the current cohort is descriptive only. It may control search only after being
frozen and validated on a future cohort.

## 7. Archive analysis

For C2 and C3 report:

- number of eligible partial components;
- atomic versus composite components;
- retrieval frequency and number of selected components per plan;
- exact versus semantic application frequency;
- dependency closure size;
- patch-application failure rate;
- original-certificate replay failure rate;
- cumulative-preservation failure rate;
- fraction of archived components later used;
- fraction of compositions that become promoted incumbents;
- final-score contribution of successful component lineages.

The key comparison is C2 versus C1 under the same search opportunities and
replicates. A component counts as useful only when it is physically retrieved or
ported, its original packet is replayed, and the resulting composition survives
the normal decision protocol. Merely listing prior edits in history is not
composition.

## 8. Alignment analysis

Calibration must be evaluated prequentially:

1. version v predicts a future candidate cohort;
2. the cohort's decisions are frozen;
3. a fresh shard supplies paired labels;
4. prediction error is measured;
5. only then may the observations produce version v+1.

Report:

- number of checkpoints and consumed shards;
- candidates and effective family weight per checkpoint;
- trigger reasons;
- positive, noninferior, negative, and inconclusive counts;
- public-card coverage;
- Brier score and log loss of prior card versions on future cohorts;
- calibration error by transfer and uncertainty band;
- challenged-family recovery or retirement;
- C3 versus C2 final performance;
- C3 versus C2 search and calibration cost.

Exact private deltas may appear in offline analysis tables but never in agent
inputs. Public-card analyses must use only the information actually published
at that version.

## 9. Final outcome analysis

For every run, the final report compares baseline and terminal on identical matched
tasks. Report:

- baseline score;
- terminal score;
- matched-task paired delta;
- number of matched tasks;
- condition-level mean and uncertainty interval across independent replicates;
- failure/status composition, not just an aggregate mean;
- final-evaluation cost.

Primary terminal contrasts are nested:

- C1 versus C0: unit-test credit assignment;
- C2 versus C1: certified partial archive;
- C3 versus C2: delayed alignment calibration;
- C3 versus C0: complete-system effect.

Use paired replicate-level contrasts across conditions wherever pool plans match. For
within-run uncertainty, resample at the benchmark cluster level, not individual
correlated variants.

## 10. Task counts and stopping rules

There is no fixed total-cost budget as a primary constraint. Effect quality is
the first priority; cost is secondary and must be measured rather than forced
equal.

Task counts should nevertheless be chosen before the main experiment:

1. run a pilot that is excluded from final claims;
2. find the smallest search pool that produces stable failure traces and
   candidate ordering;
3. find the smallest calibration shard that yields informative paired
   uncertainty at the cluster level;
4. retain enough disjoint shards for initial calibration and plausible
   distribution-shift triggers;
5. choose a final pool large enough for a meaningful paired confidence
   interval;
6. freeze the pool plan and thresholds before main runs.

The values in the example configurations are starting points, not universal
sample-size claims.

All conditions should use the same maximum search opportunities and the same
convergence definition. Early convergence is allowed and should be reported.
Calibration does not run every fixed number of iterations. It stops when no
registered trigger fires or no unopened shard remains.

Do not continue search because a final result is disappointing. Do not choose
the reported replicate, checkpoint, or condition after viewing final.

## 11. Cost accounting

Report realized cost even though it is not fixed in advance.

### Natural-task cost

Separate:

- search calls, tasks, model tokens, wall time, and monetary cost;
- calibration calls, shards, tasks, model tokens, wall time, and monetary cost;
- final calls, tasks, model tokens, wall time, and monetary cost.

### Unit and controller cost

Report:

- TestPacket executions and wall time;
- archive and preservation replay executions;
- Test Author calls/tokens/time;
- Search Planner calls/tokens/time;
- Candidate Editor calls/tokens/time;
- optional Regression Author calls/tokens/time;
- patch application and sandbox failures.

### Cost-effectiveness

Provide realized final gain versus total natural-task calls and versus total
model tokens as secondary curves. Do not replace the main effect comparison
with a fixed-budget leaderboard.

The current run summary contains search cost and unit-test wall time. Calibration
evaluation records and the final report store their own costs. The publication
ledger must aggregate all sources explicitly.

## 12. Reproducibility and leakage checks

For every run archive:

- configuration snapshot;
- BenchmarkPlan and every manifest hash;
- baseline source hash;
- target and role-agent model identifiers;
- candidate lineage and source hashes;
- packet and archive content hashes;
- composition plans and materialization receipts;
- evidence and frozen decisions;
- calibration checkpoint/card versions and shard states;
- sealed final plan and report;
- realized cost ledger.

Automated checks should establish:

- search, every calibration shard, and final are cluster-disjoint;
- no final evaluation exists before search completion;
- no private calibration observation or task identifier enters a role
  workspace;
- a candidate's own observation does not appear in its card snapshot;
- every selected component is materially applied or explicitly semantically
  ported;
- every selected component's frozen packet is replayed;
- every promoted preservation packet is replayed;
- the final runner leaves the calibration ledger unchanged.

## 13. Reporting checklist

The main paper table should include all four conditions, every planned replicate,
paired final delta, uncertainty interval, search cost, calibration cost, final
cost, and unit wall time.

The candidate-level table should include the conditional value of unit evidence,
false-positive and false-negative rates, and archive-stratum outcomes.

The archive table should demonstrate actual retrieval, application, replay, and
promotion—not only stored component counts.

The alignment table should be prequential and versioned. A post-hoc fit using
all labels is descriptive only and cannot establish that calibration improved
search.

Any failed or incomplete run, infrastructure exclusion, changed pool plan, or
manual intervention must be listed rather than silently replaced.
