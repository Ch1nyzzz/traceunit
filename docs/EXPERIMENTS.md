# Experiment design
## Research question
Does trace-conditioned unit-test design improve agent-harness optimization, and
does online reflection over earlier frozen packets and search outcomes improve
the design of later packets?

This is an optimization study. The search pool is the online objective. It is
unseen to the Test Author when a packet is frozen in an iteration, but it is
not an independently held-out calibration set across iterations because later
traces come from prior search outcomes.

## Conditions

| Condition | Editor input | Packet / latent / memory |
| --- | --- | --- |
| C0 | search traces and score history | no packet, no latent set, no memory |
| C1 | trace-conditioned packet | packet only |
| C2 | C1 plus latent capabilities (contracts + reference patches) | packet plus latent set |
| C3 | C2 plus sanitized prior UT lessons | packet, latent set, online UT memory |

All conditions use the same:

- baseline source;
- frozen L0 ontology;
- search and final manifests;
- solver model and benchmark runtime;
- iteration budget and agent budget;
- promotion decision rule where applicable.

The provided configs use 12 iterations. This remains below the original
20-iteration optimization budget while allowing C3 to use a chain of
per-iteration reflections rather than receiving one late checkpoint.

## Main outcomes

Report for every condition:

- incumbent search score by iteration;
- cumulative search cost;
- number of promotions, latent retentions, quarantines, and rejected candidates;
- latent realizations: how many retained contracts were later realized by a
  promoted candidate and migrated into the preserved set;
- packet admission and candidate-contract pass rates;
- search outcome conditional on local-contract pass or failure;
- C3 episode count and recommendation usage audit;
- final sealed paired outcome after the search run is complete.

Do not report an L0 direction ranking or a family posterior. L0 is only a
coarse trace-diagnosis descriptor.

## Interpretation

A useful C3 result is not merely a higher immediate search score. It should
also show that later packets become more diagnostic, for example through
improved hidden siblings, bridges, model-backed probes, fewer repeated packet
failures, or stronger sealed-final performance.

A local/search mismatch is evidence to inspect, not proof that a UT was bad.
Possible causes include a weak packet, edit overfit, an incomplete trace
hypothesis, natural-task noise, or a composition interaction.

## Sealed final evaluation

The final pool is not opened during optimization. Once all conditions finish,
evaluate each terminal incumbent with the sealed final runner. Report paired
final deltas and uncertainty. Final results cannot update UT memory, candidate
selection, or the packet store.

## Planned ablations

These are separate experiments, not part of the current main loop:

1. C3 online memory versus C2 latent capabilities with identical iteration and
   model budgets.
2. Free-form family names versus the frozen L0 registry.
3. Frozen-contract-only attribution versus the old all-case family-union
   attribution.
4. Flat local tests versus structural hidden siblings and bridges.
5. Capability augmentations: no scaffold versus red-team test writer, debate,
   or multi-agent review, measured with behavior-level probes.
6. A prospective held-out-UT alignment study with a separate pool. It must be
   designed and reported as an ablation, not silently mixed into search.

## Budget and safety notes

Model-backed probes are an extension point and require host-side enforcement of
model calls, tokens, repetitions, latency, and cost. Generated tests must never
receive credentials or evaluator access. Capability scaffolds must be tested
for behavioral adoption and correction, not only for component presence.