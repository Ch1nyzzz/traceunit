# TraceUnit implementation TODO

These items are intentionally outside the current minimal implementation. They
must not be described as completed paper evidence.

## P0 before model-backed benchmark runs

- Implement `BenchmarkAdapter.run_agent_probe()` for SWE-bench Verified and
  AppWorld. The host implementation must parse only the frozen declarative
  probe, run the frozen target/scaffold, and return measured model calls and
  tokens. Generated test code must never receive credentials.
- Freeze a small probe schema for counterexample generation, critique adoption,
  debate, red-team test generation, and final-answer correction. Admission must
  test behavior change, not merely the presence or invocation of a component.
- Include model-probe calls, tokens, latency, and monetary cost in experiment
  reports and decision audits.

## P0 before claiming UT-design learning

- Validate Test Author reflections against a strict schema and reject
  recommendations containing task IDs, candidate IDs, exact natural deltas, or
  unverifiable benchmark-specific claims.
- Add a prospective C3-versus-C2 ablation. A useful result is improvement in
  later hidden siblings, bridges, model-backed probes, unit/natural agreement,
  or sealed final performance—not a high score assigned to an L0 direction.

## P1 stronger diagnosis

- A single unit/natural mismatch cannot identify whether the cause was a weak
  UT, candidate overfit, noisy natural measurement, or trajectory interaction.
  Add counterfactual packet variants, mutation testing, repeated natural
  measurements, and where affordable factorial component ablations.
- Track whether later packets implement a published recommendation and whether
  the same failure recurs. Treat these as design-level outcomes, not family
  posteriors.
- Pilot the frozen L0 registry and report direction coverage plus
  `other`/`uncertain` rates. Change the registry only between versioned
  cohorts.
