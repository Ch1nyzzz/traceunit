# Vendored: editable mini-SWE-agent baseline

This directory is a vendored copy of **mini-SWE-agent** (upstream:
https://github.com/SWE-agent/mini-swe-agent), trimmed to `src/` + `pyproject.toml`
+ `LICENSE.md` (the original MIT license is preserved). It is TraceUnit's
*editable baseline scaffold* for the SWE-bench Verified benchmark: the Candidate
Editor mutates the control loop, prompts, tool execution, context policy, and
patch-submission logic under `src/minisweagent/`.

It is not imported as part of `traceunit`. At evaluation time
`scripts/run_miniswe_swebench_single.py` builds a candidate's copy from source
via `uvx --from <source-path>` and runs `minisweagent.run.benchmarks.swebench`.
The frozen solver model and the sealed official SWE-bench grader
(`traceunit/benchmarks/swebench_eval_worker.py`) live outside this tree.

No local modifications to the mini-SWE-agent sources; only `build/`, `docs/`, and
`tests/` were dropped to keep the vendored copy lean.
