"""BM25 memory scaffold compatible with dorianbrown/rank_bm25."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from worldcalib.schemas import ConversationTurn, LocomoExample, RetrievalHit
from worldcalib.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from worldcalib.utils.text import tokenize

try:  # pragma: no cover - optional dependency path
    from rank_bm25 import BM25Okapi as ExternalBM25Okapi
except Exception:  # pragma: no cover - import availability is environment-specific
    ExternalBM25Okapi = None


@dataclass
class BM25State:
    turns: tuple[ConversationTurn, ...]
    docs: list[str]
    tokenized_docs: list[list[str]]
    bm25: Any


class SimpleBM25Okapi:
    """Small BM25 fallback with the same `get_scores` surface we need."""

    def __init__(self, tokenized_docs: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in tokenized_docs) / max(1, len(tokenized_docs))
        self.doc_freq: Counter[str] = Counter()
        for doc in tokenized_docs:
            self.doc_freq.update(set(doc))
        self.n_docs = len(tokenized_docs)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc in self.docs:
            freq = Counter(doc)
            doc_len = len(doc) or 1
            score = 0.0
            for token in query_tokens:
                if token not in freq:
                    continue
                df = self.doc_freq[token]
                idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
                numerator = freq[token] * (self.k1 + 1.0)
                denominator = freq[token] + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
            scores.append(score)
        return scores


class RankBM25Scaffold(RetrievalMemoryScaffold):
    """Lexical-turn memory scaffold."""

    name = "bm25"
    reference_urls = ("https://github.com/dorianbrown/rank_bm25.git",)

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> BM25State:
        docs = [turn.render() for turn in example.conversation]
        tokenized_docs = [tokenize(doc) for doc in docs]
        bm25_cls = ExternalBM25Okapi or SimpleBM25Okapi
        return BM25State(
            turns=example.conversation,
            docs=docs,
            tokenized_docs=tokenized_docs,
            bm25=bm25_cls(tokenized_docs),
        )

    def retrieve(self, state: BM25State, question: str, config: ScaffoldConfig) -> list[RetrievalHit]:
        query = tokenize(question)
        if not query:
            return []
        scores = list(state.bm25.get_scores(query))
        if not any(score > 0 for score in scores):
            query_set = set(query)
            scores = [
                float(len(query_set & set(doc_tokens)))
                for doc_tokens in state.tokenized_docs
            ]
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        anchors = [idx for idx in ranked[: max(1, config.top_k)] if scores[idx] > 0]
        selected = _expand_indices(anchors, len(state.turns), config.window)
        return [
            RetrievalHit(
                text=state.turns[idx].render(),
                score=float(scores[idx]),
                source=self.name,
                metadata={"turn_index": idx},
            )
            for idx in selected
        ]


def _expand_indices(anchors: list[int], n: int, window: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for anchor in anchors:
        for idx in range(max(0, anchor - window), min(n, anchor + window + 1)):
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out
