# TraceUnit implementation TODO

These items are intentionally outside the current minimal implementation. They
must not be described as completed paper evidence.

## P0 before model-backed benchmark runs

- Make model-backed probes the default in practice, not just the prompt's
  preference: audit authored packets for scripted fake clients that branch on
  prompt keywords, and track the probe/deterministic case ratio per run.
- Include model-probe calls, tokens, latency, and monetary cost in experiment
  reports and decision audits.

## P0 before claiming UT-design learning

- Report the unit/search agreement rate over the run (search outcome
  conditional on unit pass/fail) and whether it rises as the world model
  accumulates distills.
- Add a prospective C3-versus-C2 ablation. A useful result is improvement in
  later hidden siblings, bridges, model-backed probes, unit/search agreement,
  or sealed final performance - not a high score assigned to an L0 direction.

## P1 stronger diagnosis

- A single unit/search mismatch cannot identify whether the cause was a weak
  UT, candidate overfit, noisy natural measurement, or trajectory interaction.
  Add counterfactual packet variants, mutation testing, repeated natural
  measurements, and where affordable factorial component ablations.
- Track whether later packets implement an earlier distill and whether the
  same mismatch kind recurs. Treat these as design-level outcomes, not family
  posteriors.
- Pilot the frozen L0 registry and report direction coverage plus
  `other`/`uncertain` rates. Change the registry only between versioned
  cohorts.
