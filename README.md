# TraceUnit

TraceUnit implements a trace-conditioned causal proxy-test optimization loop:

    clean baseline traces
      -> competing failure hypotheses
      -> pre-edit causal TestPacket
      -> reproducer + hidden sibling + positive-witness admission and freeze
      -> candidate source edit
      -> incumbent/candidate paired proxy tests
      -> natural diagnostic -> hidden canary
      -> reject / challenge test / partial archive / promote
      -> post-hoc sealed audit of all valid candidates
      -> unit/train/held-out calibration ledger

The control plane is benchmark-independent. The first two executable adapters
are SWE-bench Verified and AppWorld.

## What is different from a normal optimizer

- The Experimentalist authors a test family before seeing a candidate diff.
- A TestPacket is fallible: it must reproduce the incumbent failure, preserve
  incumbent-passing controls, include a controlled positive witness, and pass
  every declared incumbent contract before use.
- A packet is immutable while candidates are compared. It can be reused for a
  bounded number of failed candidates, challenged by sampled natural-task
  evidence, and replaced in a later iteration.
- Unit-positive/train-flat candidates can enter a structured partial archive
  when a downstream bridge test establishes counterfactual task value.
- Promotion is a deterministic gate. LLM agents never directly commit changes.
- The default protocol never uses audit results inside the search loop. After
  search ends, it labels all valid candidates on the sealed pool and upserts a
  unit-to-heldout calibration ledger. This supports the PACE-style cross-layer
  question without turning held-out feedback into an optimizer target. The
  report compares leave-one-edit-out `H ~ T` and `H ~ (U,T)` log loss and emits
  their conditional information gain in bits.

## Information boundaries

SWE-bench visible pool files contain only instance_id, public problem text,
repository, and base commit. Gold patches and test patches are stripped before
the Test Author and Optimizer workspaces are built.

AppWorld candidates run in a read-only Docker filesystem containing only the
current task's public specs/input databases, the candidate source, AppWorld
runtime packages, and a private output mount. Ground truth is not mounted. Only
after a successful candidate exit is its uniquely named state copied to a host
evaluator; failed or timed-out runs are never evaluated.

Generated unit tests run without network in a read-only container against source
and TestPacket snapshots. They receive an empty HOME and a minimal, credential-
free environment. Source and packet hashes are checked after execution. Test
files that reference evaluator APIs, gold/test patches, ground truth, absolute
host data paths, or protected environment overrides are rejected.

The built-in Codex role runner is also containerized. Experimentalist, Optimizer,
and Auditor each see only their own workspace and ephemeral CLI credentials;
host paths in prompts are rewritten under `/workspace`. Hidden tests, other role
workspaces, evaluator storage, and `sealed/posthoc_audit.json` are not mounted.
An explicit custom command can use `isolation: external`; `isolation: none` is an
unsafe opt-out and should not be used for held-out claims.

SWE-bench evaluation uses source/pool/config/harness content fingerprints and a
patch-specific official run identity. This prevents WorldCalib or the official
harness from silently reusing another candidate's report. Infrastructure,
timeout, empty-patch, unresolved, and resolved outcomes remain distinct.

## Install and run

    cd /data/home/yuhan/putty
    python -m pip install -e .
    traceunit validate-config --config configs/swebench_verified.yaml
    traceunit prepare --config configs/appworld.yaml

Run an optimization:

    traceunit optimize --config configs/swebench_verified.yaml
    traceunit optimize --config configs/appworld.yaml

Both commands resume from `run_state.json`. Benchmark caches are content-bound,
not keyed only by candidate name. Search natural-task cost is reported in model
tokens, unit-proxy cost in wall seconds, and post-hoc audit cost separately.

## Runtime requirements

The included configurations reuse local evaluation assets:

- /data/home/yuhan/WorldCalib/references/vendor/mini-swe-agent
- /data/home/yuhan/Optimizer1/data/swebench_verified_all500_test.json
- cached SWE-bench Verified Docker images and official harness
- /data/home/yuhan/WorldCalib/.venv-appworld
- /data/home/yuhan/appworld_home
- Docker with cached `node:20-slim` and `python:3.12-slim` images
- a current host Codex CLI and `~/.codex/auth.json` for built-in role agents

The target-model API key is read from `benchmark.api_key_env`. If it is not
already exported, TraceUnit reads only that named value from
`benchmark.env_file` (default: WorldCalib's `.env`); unrelated secrets are not
loaded. The key is passed to natural-task workers, never to generated unit tests.

## Run layout

    runs/<run>/
      run_state.json
      events.jsonl
      calibration.json
      benchmark_data/             immutable pool manifests
      candidates/<id>/source/     source lineage
      test_library/<packet>/      admitted immutable TestPackets
      evaluations/<id>/<pool>/    normalized traces and raw task dumps
      iterations/iter_NNN/
        packet_ref.json
        candidate.diff
        paired_tests/
        evidence.json
        decision.json
      partial_archive/<id>/manifest.json
      sealed/posthoc_audit.json       never mounted into a role workspace

Audit trajectories and evaluator details remain in evaluation storage and are
not staged into Experimentalist or Optimizer evidence. Set
`TRACEUNIT_TEST_SANDBOX_MODE=copy` only for explicit unit-test/dry-run debugging;
production `auto` mode fails closed if Docker and bubblewrap are unavailable.

## Current research boundary

Admission establishes executable reproduction, hidden variation, preservation
controls, and witness satisfiability. It does not prove that an authored witness
is the unique causal mechanism. The empirical claim must therefore be evaluated
from the post-hoc edit matrix: whether unit evidence adds held-out information
beyond visible end-to-end score and edit metadata, and whether certified partial
edits improve search efficiency under a fixed natural-evaluation budget.
