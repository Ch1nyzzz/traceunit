"""Text normalization and token helpers."""

from __future__ import annotations

import re
import string
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Tokenize into lowercase alphanumeric chunks."""

    return [
        chunk.lower()
        for chunk in re.sub(r"[^0-9A-Za-z]+", " ", text or "").split()
        if len(chunk) > 1
    ]


def normalize_answer(text: str) -> str:
    """Normalize an answer for exact/containment matching."""

    text = (text or "").lower().strip()
    text = re.sub(r"^final answer:\s*", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def extract_final_answer(prediction: str) -> str:
    """Extract the final answer marker when present."""

    if not prediction:
        return ""
    matches = re.findall(
        r"FINAL ANSWER:\s*(.+)",
        prediction,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    return lines[-1] if lines else prediction.strip()


def f1_score(prediction: str, gold: str) -> float:
    """Token F1 after answer normalization."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used when the model endpoint omits usage."""

    if not text:
        return 0
    return max(1, int(len(tokenize(text)) * 1.33))
