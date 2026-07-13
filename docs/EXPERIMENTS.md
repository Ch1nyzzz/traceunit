# Experiment design

## Research question

Does a trace-conditioned capability battery - atomic-capability probes used
as a cheap inner alignment check for the proposer - improve agent-harness
optimization, and do host-computed calibration plus an online, self-written
world model improve the design of later battery instances?

This is an optimization study. The search pool is the online objective. It is
unseen to the Test Author when a packet is frozen in an iteration, but it is
not an independently held-out calibration set across iterations because later
traces come from prior search outcomes. Transfer is measured only by the
sealed final evaluation.

## Conditions

| Condition | Editor input | Packet / archive / memory |
| --- | --- | --- |
| C0 | search traces and score history | no battery, no records, no memory |
| C1 | traces plus the capability battery (inner unit loop) | battery only |
| C2 | C1 plus archived-candidate records | battery plus archive records |
| C3 | C2 plus the append-only UT-design world model | battery, records, world model |

All conditions use the same:

- baseline source;
- frozen L0 ontology;
- search and final manifests;
- solver model and benchmark runtime;
- iteration budget and agent budget;
- promotion decision rule where applicable.

## Main outcomes

Report for every condition:

- incumbent search score by iteration;
- cumulative search cost and unit-loop cost (wall seconds, probe calls);
- number of promotions, archives (by kind), mismatches, and rejects;
- inner-loop statistics: attempts per candidate and how often the loop
  converted a failing first patch into a passing one;
- battery composition over time (groups, instances, retirements) and
  instance admission pass rates;
- search outcome conditional on the battery verdict (the unit/search
  agreement rate - the direct measure of whether the battery tracks the
  search distribution), per capability from the calibration ledger;
- C3 world-model distill count and whether distills respond to mismatches;
- final sealed paired outcome after the search run is complete.

Do not report an L0 direction ranking or a family posterior. L0 is only a
coarse trace-diagnosis descriptor.

## Interpretation

A useful C3 result is not merely a higher immediate search score. It should
show that the battery becomes more predictive: rising unit/search agreement
per capability, fewer repeated mismatches of the same kind, retirement of
information-free instances, and stronger sealed-final performance.

A unit/search mismatch is evidence to inspect, not proof that a UT was bad.
Possible causes include a weak packet, edit overfit, an incomplete trace
hypothesis, or natural-task noise. The protocol stages every mismatch for the
next Test Author; whether the diagnosis actually improves later packets is
the C3 claim under test.

## Sealed final evaluation

The final pool is not opened during optimization. Once all conditions finish,
evaluate each terminal incumbent with the sealed final runner. Report paired
final deltas and uncertainty. Final results cannot update the world model,
candidate selection, or the packet store.

## Planned ablations

These are separate experiments, not part of the current main loop:

1. C3 world model versus C2 archive records with identical iteration and
   model budgets.
2. Inner unit loop depth (max_inner_retries 0 vs 3): does the cheap retry
   loop pay for itself in promotions per search evaluation?
3. Free-form family names versus the frozen L0 registry.
4. Flat local tests versus structural hidden siblings and bridges.
5. Probe/replay-grounded tests versus deterministic stubs, measured by
   unit/search agreement.
6. A prospective held-out-UT alignment study with a separate pool, designed
   and reported as an ablation, not silently mixed into search.

## Budget and safety notes

Model-backed probes require host-side enforcement of model calls, tokens,
repetitions, latency, and cost. Generated tests must never receive
credentials or evaluator access. Capability scaffolds must be tested for
behavioral adoption and correction, not only for component presence.
