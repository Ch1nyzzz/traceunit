# TraceUnit protocol

This document is normative for the current main experiment. It describes online
search-pool optimization, not an independent transfer-calibration experiment.

## 1. Immutable inputs

Before a run:

- freeze the packaged L0 ontology into the run directory;
- freeze one search manifest and one sealed final manifest, each with a content
  hash and cluster-disjoint membership;
- freeze the baseline source and run configuration;
- reject resume if configuration, capability flags, ontology, or plan hash
  differs.

The final manifest is never exposed to the Test Author, the Candidate Editor,
or ordinary search code.

## 2. Capability diagnosis

The canonical L0 families are:

~~~
instruction context planning retrieval tool state
verification recovery termination other uncertain
~~~

A capability is a freeform slug (e.g. `evidence-before-mutation`) mapped to
one L0 family. The Test Author diagnoses the root-cause capability behind the
incumbent's failing traces - the first-principles deficit, never the surface
of one task. `other` and `uncertain` remain valid honest family outcomes.

## 3. The capability battery

The battery is the persistent unit-test axis: capability groups, each holding
several instances. One instance is a frozen single-case packet bundle
(`packet_kind: battery_instance`) probing one decision boundary, executed in
the sandboxed deterministic/probe runtime.

Instance rules:

- **Cross-domain**: an instance never reuses the search tasks' app names,
  APIs, entities, or literal values. Sibling instances in a group vary the
  surface (domain, entities, phrasing) while keeping the mechanism, so a
  one-domain verbal prompt reminder cannot move the group's pass rate.
- **Computed expectations**: probe patterns demand computed output over
  injected observations (exact identifiers, quantities, exclusions), never a
  bare API-name regex.
- **Format fairness** (host-enforced): a contains-expectation must be text
  that literally appears in the probe's staged messages - a computed
  identifier from the injected observations, or an output format the
  instructions spell out verbatim. An invented exact-format line fails
  behaviorally correct candidates on spelling and passes format-matching
  patches: false negatives and false positives at once.
- **Budget headroom** (host-enforced): at admission the host measures the
  incumbent's token usage; a probe whose max_tokens is below 2x that usage
  is rejected. A thin budget judges candidates on verbosity, not behavior,
  and systematically vetoes verification-style patches.
- **Admission**: the author declares expected_incumbent_pass per instance;
  the host measures it on the incumbent and rejects the whole update on any
  mismatch. Admitted instances are content-hashed and immutable.
- **Bounded groups**: at most `loop.max_instances_per_capability` active
  instances per group; the author retires before adding beyond the cap. The
  target group must retain at least one active incumbent-failing instance.

On a cold start the author clusters the baseline's failing traces into 4-6
capabilities and builds the initial battery (3-4 instances each).

**Visibility principle**: every agent gets the maximum evidence its role
permits; blindness is never a default. The Test Author owns the battery and
sees everything about it: the frozen probe bundles, the admission transcripts
of its own rejected attempts (previous_output/, previous_admission/), the
mismatch probe transcripts, the archived candidates' records and diffs, and
the calibration table. The Candidate Editor sees all real-task evidence -
failing traces, the full decision history with reasons, claimed mechanisms
and diffs, archive records, the world model (C3) - but never the probes'
surfaces (instance ids, descriptions, files): the battery remains a
measurement only while the measured party cannot read the questions. The
only other restrictions are the sealed final pool and gold/evaluator data.

## 4. Candidate and the inner battery loop

The Candidate Editor receives the incumbent's failing search traces, the
current aggregate search score, the decision history, the archived-candidate
records, the target capability's spec (the group's mechanism description and
per-instance incumbent results under opaque codes - never instance ids,
descriptions, or probe files, whose fictional vocabulary invites
keyword-matching patches), and (in C3) a read-only world model copy. It
implements one general mechanism-level edit (local_repair,
capability_augmentation, or orchestration_change).

After each proposed patch the controller runs **every** battery instance
host-side. The unit verdict is two-sided: the target capability's pass count
must exceed the incumbent reference, and no other capability's pass rate may
drop by more than `decision.max_battery_regression` - collateral damage fails
the verdict while the patch is still cheap to change. On failure the concrete
per-instance results return to the same editor, up to
`loop.max_inner_retries` times. The last attempt's battery evidence is
authoritative, and the candidate proceeds to paired search regardless of the
verdict.

## 5. The five-cell decision

Every mechanically valid candidate is evaluated on the immutable search pool.
The decision is a function of the battery verdict and the paired search
delta, with search boundaries at the noise margin: **improved** means
`delta >= decision.noninferiority_margin` (and positive), **regressed** means
falling below `-noninferiority_margin`, everything between is **flat**. A
one-task swing on a small pool is noise, not signal.

| unit \ search | improved | flat (within margin) | regressed |
| --- | --- | --- | --- |
| passed | **promote** | **archive** (possible credit-assignment gap) | **reject + mismatch** |
| failed | **confirm once** -> promote / archive | reject | reject |

- Promote: the candidate becomes the incumbent and its full-battery results
  become the new incumbent reference all later candidates pair against.
- Confirm: when search clears the margin but the battery did not certify, a
  battery miss must not become a permanent loss - the controller runs one
  independent paired re-evaluation of the candidate. If the confirmation
  also clears the margin the candidate **promotes** and the disagreement is
  still staged as a mismatch (the battery must catch up); otherwise the
  improvement was probably noise and the candidate is archived without a
  mismatch. Pairing for later candidates keeps using the first run.
- Archive: the candidate is recorded (diff, record.json) for later agents to
  read and re-litigate; nothing replays or migrates it.
- Mismatch: the battery and paired search disagreed beyond noise. The
  controller writes mismatch/iter_NNN with the diff, the battery deltas and
  instance results, and the per-task paired flip table; the next Test Author
  must diagnose it before updating the battery. After a promoted mismatch
  the author must make the target group sensitive to the mechanism the
  battery missed.

A search improvement without battery certification promotes only through the
confirmation re-evaluation; a single lucky run never does.

## 6. Calibration

For every search-evaluated candidate the host appends one row to
`battery/calibration.jsonl`: per-capability battery deltas, per-instance
results, and the paired search delta. From these it derives per-capability
direction agreement and the list of constant (information-free) instances,
staged to the Test Author as a markdown table. Calibration informs the
author's attention and retirements; it never gates a decision by itself.

## 7. UT-design world model (C3)

One append-only markdown file, written by the Test Author itself:

~~~
world model + last_iteration.json (+ mismatch evidence)
    -> staged into the next Test Author's workspace
    -> the author reads the file first, appends `## iter_NNN distill`
    -> the harness copies the file back verbatim
~~~

The staged evidence is raw: the previous decision, the per-task paired search
flips, the battery deltas, and on a mismatch the failing instances, the diff,
and the failed traces. The harness owns no schema, sanitization, or fallback
text; a skipped distill is recorded as a world_model_not_updated event. The
world model guides later battery design; it never overrides current trace
evidence and never ranks L0 families.

## 8. Resume

decision.json and evidence.json are a commit boundary. If a process stops
after writing them, resume loads those artifacts, commits any missing state
effect, and advances the iteration. It does not rerun battery instances or
search evaluation. Inside an iteration, battery_update_ref.json and
inner_state.json make the author update and the inner loop resumable. Three
consecutive skipped iterations (agent failures) halt the run, hand the
skipped iterations back, and leave it resumable.

## 9. Final evaluation

Final evaluation is a distinct sealed command, chained automatically after a
completed search run (`--no-final` disables). It consumes the final manifest
only after search finishes. Its result does not alter run state, the battery,
the world model, or decisions.

## 10. Scope of claims

The main experiment can claim that a trace-conditioned capability battery,
used as a cheap inner alignment check with host-computed calibration and an
online self-written world model, helps search-pool optimization under the
stated protocol. It cannot claim that a particular capability has
intrinsically higher transfer value. Transfer itself is measured only by the
sealed final evaluation.
