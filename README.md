# TraceUnit

TraceUnit is a research system for improving an agent harness from its own
execution traces. Before an edit exists, a Test Author turns trace-supported
failure hypotheses into a frozen unit-test packet. A Search Agent then proposes
an edit, optionally retrieves and composes previously certified partial edits,
and receives deterministic evidence from unit tests and the visible natural-task
search pool.

The paper-facing primary hypothesis is:

> Agent-authored, trace- and history-conditioned unit tests reduce the
> credit-assignment ambiguity of sparse end-to-end scores.

The accompanying alignment hypothesis is that a small number of natural-task
calibration observations can identify which unit-test families predict positive
transfer. Most candidate screening can then use cheap unit evidence, while a
completely separate, unseen, in-distribution final pool measures terminal
generalization.

These are hypotheses to be tested by the experiment matrix in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), not conclusions assumed by the
implementation.

## First-principles design

A scalar task score says that an agent failed, but usually not which internal
mechanism caused the failure. TraceUnit separates four questions:

1. **Localization:** what trace boundary or policy mechanism is plausibly
   responsible?
2. **Intervention:** does a pre-edit test distinguish that mechanism from
   competing explanations?
3. **Credit:** did the candidate repair the frozen mechanism without breaking
   certified behavior?
4. **Transfer:** does that kind of unit evidence predict improvement on unseen
   natural tasks?

The separation matters because unit tests can overfit too. Hidden structural
siblings, downstream bridge tests, regression controls, sparse natural-task
calibration, and a final sealed evaluation address different failure modes and
must not be collapsed into one score.

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
the seed and terminal sources on the final pool:

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

Only after every condition and seed that may influence model selection has
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

- **loop:** run directory, iterations, seed, resume behavior, and trace staging
  and TestPacket-reuse limits.
- **protocol:** the single experiment condition: `c0_score_only`,
  `c1_raw_traceunit`, `c2_archive`, or `c3_full`.
- **benchmark:** immutable pool sources/splits, search, calibration, and final
  limits, calibration shard size, model/runtime settings, and clustering seed.
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
different `loop.run_dir` for every benchmark × condition × seed run. There is no
batch matrix launcher, so experiment scheduling remains external. The runtime
derives capabilities from the condition and verifies forbidden workspaces and
artifacts are absent instead of relying on independent feature booleans.
