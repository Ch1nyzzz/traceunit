"""Baseline HLE answering policy -- the editable surface for the Candidate Editor.

Everything in this file is fair game to edit: the system prompt, the answer
format instruction, how the final answer is extracted from the model's reply,
and the answering strategy (single pass vs. self-consistency voting). The public
``answer_question`` signature must stay stable; its internals may change freely.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

# --- editable knobs ---------------------------------------------------------

#: How many independent samples to draw before voting on a final answer. 1 keeps
#: the baseline cheap; raise it (with temperature > 0) for self-consistency.
N_SAMPLES = 1
#: Sampling temperature. 0.0 is deterministic; self-consistency wants > 0.
TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are an expert problem solver taking a rigorous written exam spanning "
    "mathematics, physics, computer science, and engineering. Reason carefully "
    "and show the key steps, then commit to a single best answer."
)

#: The model is told to end with this exact marker so the answer is machine-
#: extractable. Editing the marker means editing ``extract_answer`` too.
ANSWER_MARKER = "FINAL ANSWER:"

FORMAT_INSTRUCTION = (
    "First work through the problem, then on the last line write exactly:\n"
    f"{ANSWER_MARKER} <your answer>\n"
    "For multiple-choice questions the answer must be the letter of the correct "
    "option. For exact-match questions give only the final value or expression, "
    "with no extra words."
)


# --- policy implementation --------------------------------------------------


def build_messages(question: str, answer_type: str) -> list[dict[str, str]]:
    """Construct the chat messages sent to the frozen solver model."""

    kind = (
        "This is a multiple-choice question."
        if answer_type == "multipleChoice"
        else "This is an exact-match question."
    )
    user = f"{kind}\n\n{question.strip()}\n\n{FORMAT_INSTRUCTION}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_answer(response_text: str, answer_type: str) -> str:
    """Pull the committed answer out of a model reply."""

    text = response_text or ""
    marker_hits = list(
        re.finditer(re.escape(ANSWER_MARKER), text, flags=re.IGNORECASE)
    )
    if marker_hits:
        answer = text[marker_hits[-1].end():].strip()
    else:
        # No marker: fall back to the last non-empty line.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        answer = lines[-1] if lines else text.strip()
    # Trim surrounding markup/punctuation the model sometimes adds.
    answer = answer.strip().strip("*").strip()
    if answer_type == "multipleChoice":
        letter = re.search(r"[A-Za-z]", answer)
        if letter:
            answer = letter.group(0).upper()
    return answer[:2000]


def _vote(answers: list[str]) -> str:
    """Majority vote over samples, preferring the first on ties."""

    cleaned = [a for a in answers if a]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    best = max(cleaned, key=lambda a: (counts[a], -cleaned.index(a)))
    return best


def answer_question(
    *,
    question: str,
    answer_type: str,
    call_model: Callable[..., dict[str, Any]],
    max_output_tokens: int,
) -> dict[str, Any]:
    """Answer one HLE question with the frozen solver model.

    ``call_model(messages, max_tokens, temperature)`` is provided by the host and
    returns ``{"content", "prompt_tokens", "completion_tokens"}``.
    """

    messages = build_messages(question, answer_type)
    samples: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    last_raw = ""
    for _ in range(max(1, N_SAMPLES)):
        response = call_model(
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=TEMPERATURE,
        )
        last_raw = str(response.get("content") or "")
        prompt_tokens += int(response.get("prompt_tokens") or 0)
        completion_tokens += int(response.get("completion_tokens") or 0)
        samples.append(extract_answer(last_raw, answer_type))

    prediction = _vote(samples)
    return {
        "prediction": prediction,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "raw": last_raw,
        "metadata": {"samples": samples, "n_samples": len(samples)},
    }
