# Vendored: editable Terminus scaffold

`editable_terminus/` is a vendored copy of Harbor's **Terminus 2** agent
(`harbor.agents.terminus_2`, Harbor v0.3.0), copied verbatim and then adapted so
TraceUnit can treat it as the *editable baseline scaffold* that the Candidate
Editor mutates — mirroring the "runnable harness (prompts, middleware, memory,
project files)" that SEAGym's AHE method evolves.

Upstream: https://github.com/harbor-framework/harbor — Licensed under Apache-2.0.

## Local modifications

- Intra-package imports rewritten from `harbor.agents.terminus_2.*` to the
  vendored top-level package name `editable_terminus.*`, so editing a sibling
  module (parser, tmux driver, templates) affects this copy rather than the
  installed Harbor one. All other `harbor.*` imports are left intact and resolve
  against the installed Harbor library at runtime.
- `Terminus2.name()` returns `"traceunit-terminus"` instead of `"terminus-2"`,
  so Harbor JobStats keys make the optimized scaffold unambiguous.

## How it is loaded

The package is never imported as part of `traceunit`; it is placed on
`sys.path` explicitly by `traceunit.benchmarks.harbor_worker` (from a candidate's
copied source tree) and loaded by Harbor via the import path
`editable_terminus.terminus_2:Terminus2`.
