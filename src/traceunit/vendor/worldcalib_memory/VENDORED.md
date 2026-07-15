# Vendored: memory evaluation substrate

`src/worldcalib/` is the minimal LoCoMo + LongMemEval evaluation substrate,
vendored from the WorldCalib repo. Only the import closure the memory adapter
needs is included (17 modules: the `EvaluationRunner`, dynamic candidate loader,
memory data loaders, LongMemEval judge, memory scaffolds, and their shared
`schemas`/`model`/`metrics`/`utils`/`pareto` deps). WorldCalib's multi-benchmark
code (agentic/reasoning backends, CLIs, optimizer) is excluded; `dynamic.py`'s
non-memory branches are dead code here and never import.

**The Python package name is deliberately kept as `worldcalib`.** The candidate
seed the Candidate Editor edits is a `src/worldcalib/` subset, and the dynamic
loader (`worldcalib.dynamic._isolated_memomemo_project`) swaps candidate modules
in and out of `sys.modules` by that exact name, so renaming would require
rewriting the loader, the seed package structure, and every internal import.

It is never imported as part of `traceunit`; the adapter puts
`vendor/worldcalib_memory/src` on `sys.path` on demand via
`common.worldcalib_import`. Third-party deps: `httpx` (model/judge client) and
optionally `rank_bm25` (a pure-Python fallback exists). Raw datasets
(locomo10.json, longmemeval_*.json) stay external via `benchmark.data_path`.
