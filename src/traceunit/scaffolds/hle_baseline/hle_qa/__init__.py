"""Editable HLE question-answering scaffold.

This package is the *optimization target* for the HLE benchmark: the Candidate
Editor mutates the prompt, answer-extraction, and answering strategy here to
improve accuracy under a frozen solver model. It is imported by
``traceunit.benchmarks.hle_worker`` from a candidate's copied source tree and is
never part of the installed ``traceunit`` package.

Contract (kept stable so the host runner can call any edited version):

    from hle_qa import answer_question

    result = answer_question(
        question=str,            # the HLE question text (never the gold answer)
        answer_type=str,         # "exactMatch" or "multipleChoice"
        call_model=callable,     # host-provided frozen-model client (see below)
        max_output_tokens=int,   # per-call output cap
    )

``call_model(messages, max_tokens, temperature) -> dict`` returns
``{"content": str, "prompt_tokens": int, "completion_tokens": int}``. The
scaffold owns the answering loop; the host owns the API client and the (hidden)
LLM judge. The scaffold must return::

    {"prediction": str, "prompt_tokens": int, "completion_tokens": int,
     "raw": str, "metadata": dict}
"""

from __future__ import annotations

from hle_qa.policy import answer_question

__all__ = ["answer_question"]
