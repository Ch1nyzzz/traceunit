# TraceUnit

TraceUnit is a research system for improving an agent harness from its own
execution traces. It converts trace-supported failure hypotheses into frozen,
mechanism-level tests before an edit exists, uses those tests to assign credit
to candidate edits, and measures whether that local evidence predicts transfer
to unseen natural tasks.

The optimization target is **not unit-test pass rate**. The target is the paired
baseline-to-terminal improvement on a sealed, in-distribution final pool. Unit tests
are useful only insofar as they are a cheaper, more diagnostic surrogate for
that target.

The paper-facing thesis is therefore conditional and falsifiable:

> Frozen, trace-conditioned capability tests can reduce the credit-assignment
> ambiguity and target-system cost of sparse end-to-end optimization, but only
> when their transfer value is demonstrated out of sample.

The implementation enforces the protocol needed to test this thesis. It does
not assume that a unit pass proves generalization, that every authored family is
well aligned, or that the example configurations have enough statistical power.
Those are empirical questions defined in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## First-principles argument

### 1. The terminal objective is natural-task generalization

For a terminal harness \(h_T\) and fixed baseline harness \(h_0\), the scientific outcome
is the paired score difference on tasks that never influenced search,
calibration, thresholds, or model selection:

    final effect = mean_j(score(h_T, final_j) - score(h_0, final_j))

Search score, calibration labels, unit tests, archive certificates, and proxy
curves are intermediate evidence. None can replace the sealed final comparison.

### 2. End-to-end reward is expensive, sparse, and entangled

A natural-task score reveals whether a complete trajectory succeeded, but not
which internal decision caused success or failure. Planning, retrieval, tool
selection, argument construction, state tracking, verification, recovery, and
stopping behavior may all contribute to one scalar reward. Re-evaluating every
candidate on a full natural-task pool is also the dominant target-system cost.

This creates two problems:

- **credit assignment:** the optimizer cannot tell which mechanism to repair or
  preserve;
- **evaluation cost:** weak candidates consume the same expensive signal as
  promising candidates.

### 3. Treat an edit as a causal intervention

Before the candidate or composition plan exists, a Test Author uses failed
search traces to propose at least two competing failure hypotheses and freezes a
TestPacket around one target mechanism. Freezing first prevents a candidate from
choosing the test it already satisfies and prevents post-hoc success criteria.

A normal packet separates several kinds of local evidence:

| Test evidence | Question answered |
|---|---|
| **public reproducer** | Can the candidate repair the visible minimal contract? |
| **hidden structural sibling** | Does the repair survive a mechanism-preserving variation? |
| **positive witness** | Is the proposed test/intervention satisfiable rather than vacuous? |
| **regression control** | Did the edit destroy incumbent behavior that should remain valid? |
| **bridge test** | Does the local repair propagate to a downstream boundary without using the benchmark grader? |
| **archive/preservation replay** | Does a composition retain previously certified behavior? |

`family_id` is intended to identify a stable, task-independent mechanism such
as tool selection or verification, not a benchmark item, repository, or free-
form failure description. A packet should make one dominant mechanism easy to
attribute. Bridge and composition tests remain necessary because end-to-end
success is generally an interaction among several atomic capabilities, not the
arithmetic mean of independent unit passes.

The intended family governance is deterministic at the experiment level:

1. each benchmark freezes a capability-ontology version before the main runs;
2. every condition and independent replicate for that benchmark uses the same
   version;
3. the Test Author diagnoses the trace and selects the closest canonical family;
4. if no family fits, the agent may propose a provisional family, but it cannot
   enter current-cohort calibration until reviewed and released in a later
   ontology version;
5. benchmarks may add domain-specific extensions, while semantically identical
   cross-benchmark capabilities should reuse shared core IDs.

This prevents the optimizing agent from changing the label space that is used
to judge its own transfer. The registry and admission enforcement for this
governance are not yet implemented; current packets still carry authored string
keys, as stated in the implementation boundary below.

### 4. A unit test becomes a proxy only after independent alignment

For candidate edit \(i\), TraceUnit distinguishes:

- \(U_i\): frozen unit evidence, including public/hidden/bridge gains,
  regression loss, replay status, family, and composition context;
- \(S_i\): visible search evidence, including whether search evaluation is
  missing because the unit gate rejected the candidate;
- \(Y_i\): candidate-minus-parent transfer measured later on a fresh natural-
  task calibration shard.

The central comparison is:

    baseline:  Y_i <- S_i
    proxy:     Y_i <- (S_i, U_i)

The proxy claim is supported only if adding \(U_i\) improves prediction on a
candidate cohort excluded from fitting, preferably under leave-one-lineage-out
evaluation across independent optimization replicates. Relevant evidence includes Brier score, log
loss, pairwise selection accuracy, false positives, false negatives, selection
regret, and the OOF reliability curve. In-sample correlation is not sufficient.

### 5. Cheaper evaluation is a consequence, not an assumption

Under the full C3 condition, the implemented online loop forms a coarse multi-
fidelity cascade; ablation conditions remove the corresponding stages:

    mechanical checks
      -> frozen public/hidden/regression/bridge evidence
      -> archive and preservation replay
      -> paired natural-task search evaluation only when the unit gate passes
      -> triggered calibration on fresh shards
      -> one-way final evaluation after search terminates

If OOF and prospective results show that a proxy threshold retains acceptable
positive-transfer recall, later runs may use it to skip more full natural-task
evaluations. Rejected candidates must still receive a predeclared random audit;
otherwise false negatives become unobservable and proxy drift looks like cost
savings. Thresholds chosen on the current cohort are descriptive and cannot
control that same cohort.

“Progressive disclosure” here means allocating progressively more **evaluation
budget**. Hidden tests, private calibration outcomes, and final-task contents are
never progressively disclosed to the editing agents.

### 6. Keep three claims and three evidence levels separate

TraceUnit distinguishes:

1. **local certificate:** immutable unit, bridge, regression, and replay facts;
2. **alignment prior:** delayed, uncertain evidence that a test family predicted
   natural transfer for earlier candidates;
3. **final evidence:** a one-way paired measurement of the selected terminal
   system on a pool that never participates in optimization.

A strong local certificate may still have weak transfer. A well-aligned family
is still not a universal correctness proof. A positive search or calibration
trajectory is still not a final generalization result.

## Protocol at a glance

    search trace/history
      -> competing failure hypotheses
      -> pre-edit TestPacket admission and freeze
      -> delayed alignment-card snapshot
      -> autonomous 0..N archive retrieval plan
      -> exact patch materialization / explicit semantic port
      -> candidate edit
      -> paired public + hidden + bridge + regression tests
      -> replay selected components' original packets
      -> replay cumulative promoted packets
      -> paired search-pool evaluation when unit evidence passes
      -> reject / quarantine / archive / promote
      -> freeze the decision
      -> optionally consume one fresh calibration shard
      -> publish aggregate cards for future iterations only

After search reaches a terminal state, a separate command seals and evaluates
the baseline and terminal sources on the final pool:

    completed search -> seal final plan -> final-evaluate

The final result never updates the calibrator, archive, prompts, thresholds, or
search state.

## Three natural-task roles

| Role | Used for | May affect later search? | Visible to agents? |
|---|---|---:|---:|
| **search** | Traces, end-to-end search score, candidate promotion/archive decisions | Yes | Sanitized traces and aggregate score evidence |
| **calibration** | Paired candidate-parent transfer observations and unit-family reliability cards | Yes, after a delay | Only discrete aggregate cards |
| **final** | Seed-versus-terminal generalization report | No | Never |

Every pool slice is content-bound by a manifest hash. Search, calibration
shards, and final are disjoint. AppWorld keeps an entire scenario in one role
and one shard. SWE-bench keeps a repository together; rows without a repository
fall back to instance-level clusters.

Calibration is therefore a validation signal, not a second test set. A
candidate's calibration observation is collected only after its decision file
has been written. It can update cards for later iterations, but cannot change
that candidate's promotion decision. Repeated feedback uses fresh,
single-use calibration shards. See [docs/PROTOCOL.md](docs/PROTOCOL.md) for the
normative sequence and information boundaries.

## Partial archive and composition

A unit- and bridge-supported edit whose search score is noninferior but not
positive is stored as a content-addressed archive component. Its local
certificate and frozen TestPacket are immutable. Alignment cards are uncertain,
versioned priors and are not part of that certificate.

Before editing, the Search Planner may select zero, one, or any number of
components. There is no hard-coded top-three rule. Dependencies are expanded
and ordered by the archive catalog. Exact patches are checked and applied in an
isolated staging tree; semantic ports must be declared explicitly. The
resulting candidate must re-run every selected component's original frozen
packet and all cumulative promoted preservation packets.

## Falsifiable claims and experiment matrix

The experimental unit is `benchmark × condition × replicate`. Conditions share the
same benchmark plan, target model, baseline source, search opportunities, and final
procedure.

A replicate is an independent rerun of the optimization procedure, not a
different baseline harness or benchmark split. Replicates are useful only when
model sampling, tool execution, or the optimizer is stochastic; a fully
deterministic procedure does not gain information from identical reruns.
`loop.run_id` identifies the replicate. The benchmark's `benchmark_seed` is
fixed once for pool construction and reproducible evaluator randomness; it is
part of the shared, content-hashed benchmark protocol and is not varied as an
experimental factor.

| Condition | Unit packets | Partial archive/composition | Delayed alignment |
|---|---:|---:|---:|
| **C0 `c0_score_only`** | No | No | No |
| **C1 `c1_raw_traceunit`** | Yes | No | No |
| **C2 `c2_archive`** | Yes | Yes | No |
| **C3 `c3_full`** | Yes | Yes | Yes |

The nested contrasts answer different questions:

- **C1 − C0:** does frozen unit evidence improve edit-level credit assignment
  and terminal performance beyond score-only search?
- **C2 − C1:** do certified partial edits become useful when they can be
  retrieved, composed, and replayed?
- **C3 − C2:** do delayed family-level alignment priors improve future search
  rather than merely purchase more natural-task labels?
- **C3 − C0:** does the complete system improve sealed final performance?

The evidence reported for these claims must remain distinct:

| View | Primary x/y | What it establishes |
|---|---|---|
| **OOF proxy alignment** | predicted transfer probability / observed positive-transfer rate | whether unit evidence is a calibrated candidate-level surrogate |
| **iteration-score** | search iteration / incumbent search score | optimization progress under a fixed opportunity budget |
| **cost-score** | cumulative search+calibration natural-task cost / incumbent search score | realized search efficiency |
| **selective-evaluation curve** | proxy threshold / full-evaluation rate and positive recall | possible future target-call savings and their false-negative tradeoff |
| **sealed final outcome** | baseline score / terminal score on matched final tasks | terminal generalization; the primary optimization result |

Final outcomes must be reported per condition, using paired replicate-level contrasts
where benchmark plans match. Search trajectories or proxy curves cannot be
presented as substitutes for the final paired effect.

## Current implementation boundary

The repository implements the frozen-packet protocol, coarse unit-to-search
gate, archive and replay machinery, delayed calibration, sealed final runner,
and offline OOF proxy/trajectory analysis. It deliberately does **not**
automatically deploy a learned proxy threshold into the online optimizer.
Threshold deployment requires a threshold frozen on an earlier cohort and a
prospective cohort with random audits of rejected candidates.

The current `family_id` mechanism keys are authored per packet; a complete,
versioned capability ontology and empirical coverage audit remain research
artifacts rather than assumptions in the runtime. Likewise, the checked-in
`runs/` directory contains prepared manifests, not completed evidence for the
paper-facing claims. Passing the software test suite establishes protocol
behavior, not benchmark generalization.

## Install and commands

TraceUnit requires Python 3.11 or newer.

    cd /data/home/yuhan/putty
    python -m pip install -e ".[dev]"

Validate a strict configuration and freeze benchmark pools:

    traceunit validate-config --config configs/swebench_verified.yaml
    traceunit prepare --config configs/swebench_verified.yaml

Run or resume the condition selected by `protocol.condition`:

    traceunit optimize --config configs/swebench_verified.yaml
    traceunit optimize --config configs/appworld.yaml

Inspect a run and validate an authored packet:

    traceunit inspect --run-dir runs/swebench_verified_traceunit
    traceunit validate-packet --bundle /path/to/frozen/packet

After at least two independent run lineages have accumulated informative private
calibration observations, evaluate whether unit evidence predicts natural
transfer out of sample:

    traceunit analyze-proxy \
      --run-dir runs/replicate_1 \
      --run-dir runs/replicate_2 \
      --run-dir runs/replicate_3 \
      --skip-below 0.10 \
      --skip-below 0.25 \
      --audit-rate 0.10 \
      --alignment-bins 10 \
      --output runs/proxy_analysis.json

The command fits search-only and search-plus-unit models under leave-one-group-
out evaluation. Its primary outputs are:

- an OOF proxy-alignment reliability curve: predicted positive-transfer
  probability versus observed positive-transfer rate and mean paired delta;
- an iteration-score curve for the incumbent search score;
- a cost-score curve using cumulative search plus calibration natural-task cost
  in the benchmark adapter's cost units, with unit-test wall time kept separate;
- the sealed final baseline-versus-terminal outcome, when final evaluation has
  already been completed.

It also reports Brier score, log loss, pairwise selection accuracy,
false-positive/negative rates, selection regret, and a selective full-evaluation
curve. The selective curve estimates how many complete natural-task evaluations
could be skipped at each proxy threshold and how much positive-transfer coverage
remains with a random audit of skipped candidates. Final results never enter
proxy features, labels, model fitting, or threshold selection. The command does
not execute or open final evaluation; it only copies an already sealed final
report into the optimization-outcome section. Thresholds must be selected on an
earlier cohort, frozen, and evaluated prospectively before controlling online
search.

Only after every condition and planned replicate that may influence model selection has
finished should the terminal sources be frozen and the final pool opened:

    traceunit final-evaluate --config configs/swebench_verified.yaml

The **optimize** command never invokes **final-evaluate**. The latter refuses to
seal a plan unless search is **completed** or **converged**, binds the search
state, benchmark plan, source hashes, and final slice, and writes a separate
report.

## Configuration

The parser rejects unknown keys; legacy pool or role names are not accepted.
Current examples are:

- [configs/swebench_verified.yaml](configs/swebench_verified.yaml)
- [configs/appworld.yaml](configs/appworld.yaml)

Important sections are:

- **loop:** run identity, directory, iterations, resume behavior, and trace staging
  and TestPacket-reuse limits.
- **protocol:** the single experiment condition: `c0_score_only`,
  `c1_raw_traceunit`, `c2_archive`, or `c3_full`.
- **benchmark:** immutable pool sources/splits, search, calibration, and final
  limits, calibration shard size, model/runtime settings, and one fixed
  fixed `benchmark_seed` used for the shared BenchmarkPlan and evaluator
  randomization.
- **agents:** test author, search, and optional regression author.
- **decision:** admission, public/hidden/bridge, regression, search-delta, and
  noninferiority thresholds.
- **alignment:** checkpoint cohort sizes, minimum effective support, positive
  margin, and rotation triggers.
- **archive:** semantic-port policy for conditions with a component archive.

For SWE-bench, providing only **search_data_path** treats that file as the source
dataset and deterministically partitions it into the three roles. Supplying
explicit pools requires all of **search_data_path**, **calibration_data_path**,
and **final_data_path**. AppWorld normally uses **split_manifest_path**,
**search_split**, and **heldout_split**; the held-out source is divided into
calibration and final scenarios.

## Runtime and information security

The included configurations expect local WorldCalib, AppWorld, SWE-bench, and
Docker assets. The target-model key is read from **benchmark.api_key_env**. If it
is absent from the process environment, TraceUnit reads only that named value
from **benchmark.env_file**; unrelated secrets are not loaded.

Generated tests run without network access against read-only source and packet
snapshots with a minimal credential-free environment. Packet and source hashes
are checked around execution. Evaluator APIs, ground truth, gold/test patches,
absolute host-data paths, and protected environment overrides are rejected.

Role workspaces are separated:

- the Test Author sees sanitized search traces, the incumbent, and delayed
  public calibration cards;
- the Search Planner sees the public packet, prior aggregate decisions, archive
  catalog, parent source, and delayed cards;
- the Candidate Editor sees the frozen plan, materialization receipt, public
  packet, and editable staged source;
- no role sees hidden tests, private calibration observations, calibration task
  contents, or final-evaluation artifacts.

## Run layout

    runs/<run>/
      config.snapshot.json
      run_state.json
      events.jsonl
      summary.json
      benchmark_data/
        plan.json
        <benchmark>/plan.json
        <benchmark>/<pool-slice>.json
      candidates/<candidate>/
        parent_source/
        source/
        composition_plan.json
        materialization_receipt.json
        proposal.json
      iterations/iter_NNN/
        inputs/test_author_cards.json
        inputs/search_cards.json
        packet_ref.json
        candidate.diff
        paired_tests/
        archive_replay/
        preservation_replay/
        evidence.json
        decision.json
      frozen_packets/                 cumulative promoted-packet certificates
        packets/<packet-hash>/
      component_archive/              present only in C2/C3
        <component-hash>/manifest.json
        <component-hash>/component.patch
        usage.jsonl
      calibration/                    present only in C3
        private_observations.jsonl
        public_cards.json
        pending.jsonl
        pending/<candidate>.json
        shards.json
        checkpoints/<checkpoint>/checkpoint.json
      evaluations/<candidate>/<pool-slice>/
      sealed/final/
        plan.json
        evaluations/
        report.json

`analyze-proxy` may aggregate several run directories, so its report is written
to the explicit `--output` path rather than owned by any one run.

**summary.json** separately reports search cost, calibration cost, their total,
and unit-test wall time. In addition,
**sealed/final/report.json** reports final-evaluation cost. Publication tables
must aggregate and report all three rather than treating the search summary as
total experiment cost.

## Research protocol

- Full operational protocol: [docs/PROTOCOL.md](docs/PROTOCOL.md)
- Main conditions, metrics, stopping rules, and cost accounting:
  [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

The four main conditions are executable through `protocol.condition`; use a
different `loop.run_dir` for every benchmark × condition × replicate run. There is no
batch matrix launcher, so experiment scheduling remains external. The runtime
derives capabilities from the condition and verifies forbidden workspaces and
artifacts are absent instead of relying on independent feature booleans.
