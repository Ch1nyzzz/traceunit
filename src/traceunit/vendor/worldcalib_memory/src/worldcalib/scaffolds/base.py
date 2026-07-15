"""Memory scaffold protocol for LOCOMO QA agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from worldcalib.metrics import retrieval_oracle_prediction
from worldcalib.model import LocalModelClient, build_answer_messages
from worldcalib.schemas import LocomoExample, RetrievalHit
from worldcalib.utils.text import estimate_tokens


@dataclass(frozen=True)
class ScaffoldConfig:
    """Runtime configuration for a memory scaffold candidate."""

    top_k: int = 8
    window: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "window": self.window,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ScaffoldRun:
    """One scaffold-produced answer plus evaluation-facing traces."""

    prediction: str
    prompt_tokens: int
    completion_tokens: int
    retrieved: list[RetrievalHit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryScaffold(ABC):
    """Full memory-to-answer scaffold wrapped around the answer LLM."""

    name: str
    reference_urls: tuple[str, ...] = ()

    @abstractmethod
    def build(self, example: LocomoExample, config: ScaffoldConfig) -> Any:
        """Build per-example memory state from the conversation."""

    @abstractmethod
    def answer(
        self,
        state: Any,
        example: LocomoExample,
        client: LocalModelClient,
        config: ScaffoldConfig,
        *,
        max_context_chars: int,
        dry_run: bool,
    ) -> ScaffoldRun:
        """Use memory state and the answer LLM to answer one LOCOMO task."""

    def run(
        self,
        example: LocomoExample,
        client: LocalModelClient,
        config: ScaffoldConfig,
        *,
        max_context_chars: int,
        dry_run: bool,
    ) -> ScaffoldRun:
        """Build memory state and answer the task."""

        state = self.build(example, config)
        return self.answer(
            state,
            example,
            client,
            config,
            max_context_chars=max_context_chars,
            dry_run=dry_run,
        )


class RetrievalMemoryScaffold(MemoryScaffold):
    """Scaffold that retrieves hits, then uses the default grounded QA prompt."""

    @abstractmethod
    def retrieve(
        self,
        state: Any,
        question: str,
        config: ScaffoldConfig,
    ) -> list[RetrievalHit]:
        """Return retrieved memory hits for a question."""

    def answer(
        self,
        state: Any,
        example: LocomoExample,
        client: LocalModelClient,
        config: ScaffoldConfig,
        *,
        max_context_chars: int,
        dry_run: bool,
    ) -> ScaffoldRun:
        hits = self.retrieve(state, example.question, config)
        return answer_from_hits(
            example=example,
            hits=hits,
            client=client,
            max_context_chars=max_context_chars,
            dry_run=dry_run,
        )


def answer_from_hits(
    *,
    example: LocomoExample,
    hits: list[RetrievalHit],
    client: LocalModelClient,
    max_context_chars: int,
    dry_run: bool,
) -> ScaffoldRun:
    """Answer with the default grounded-memory prompt."""

    retrieved_text = "\n\n".join(hit.text for hit in hits)
    if dry_run:
        prediction = retrieval_oracle_prediction(retrieved_text, example.answer)
        prompt_tokens = estimate_tokens(example.question + "\n" + retrieved_text)
        completion_tokens = estimate_tokens(prediction)
    else:
        messages = build_answer_messages(
            question=example.question,
            hits=hits,
            category=example.category,
            max_context_chars=max_context_chars,
        )
        response = client.chat(messages, max_tokens=256, temperature=0.0)
        prediction = response.content
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.completion_tokens
    return ScaffoldRun(
        prediction=prediction,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        retrieved=hits,
    )
