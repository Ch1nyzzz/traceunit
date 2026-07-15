"""Passrate and answer scoring."""

from __future__ import annotations

from worldcalib.utils.text import extract_final_answer, f1_score, normalize_answer


PASS_F1_THRESHOLD = 0.8


def score_prediction(prediction: str, gold_answer: str) -> float:
    """Score one prediction with exact, containment, then token F1."""

    answer = extract_final_answer(prediction)
    if not answer:
        return 0.0
    pred_norm = normalize_answer(answer)
    gold_norm = normalize_answer(gold_answer)
    if pred_norm == gold_norm:
        return 1.0
    if gold_norm and gold_norm in pred_norm:
        return 1.0
    return f1_score(answer, gold_answer)


def passed(score: float) -> bool:
    """Convert scalar score to pass/fail for passrate."""

    return score >= PASS_F1_THRESHOLD


def retrieval_oracle_prediction(retrieved_text: str, gold_answer: str) -> str:
    """Dry-run prediction: pass if retrieved context contains the gold."""

    gold_norm = normalize_answer(gold_answer)
    context_norm = normalize_answer(retrieved_text)
    if gold_norm and gold_norm in context_norm:
        return f"FINAL ANSWER: {gold_answer}"
    return "FINAL ANSWER: unknown"
