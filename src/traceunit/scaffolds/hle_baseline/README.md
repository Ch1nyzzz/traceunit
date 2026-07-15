# Editable HLE scaffold

`hle_qa/` is TraceUnit's original editable answering scaffold for Humanity's Last
Exam — the optimization target the Candidate Editor mutates. It is **not**
vendored from anywhere; it is a minimal, deliberately-editable QA policy.

Editable surface (`hle_qa/policy.py`): the system prompt, the answer-format
instruction, answer extraction, and the answering strategy (single pass vs.
self-consistency voting via `N_SAMPLES` / `TEMPERATURE`).

Stable contract (`hle_qa/__init__.py`): the host runner
(`traceunit.benchmarks.hle_worker`) imports `answer_question` from a candidate's
copied source tree. The scaffold receives only the question text and a
host-provided `call_model` client; it never sees the gold answer, which stays
with the host LLM judge.
