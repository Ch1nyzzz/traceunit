"""Shared data schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConversationTurn:
    """One flattened LOCOMO conversation turn."""

    session: str
    session_date: str
    dia_id: str
    speaker: str
    text: str
    global_index: int

    def render(self, *, max_chars: int = 500) -> str:
        text = self.text if len(self.text) <= max_chars else self.text[:max_chars] + "..."
        return f"[{self.session} | {self.session_date} | {self.dia_id}] {self.speaker}: {text}"


@dataclass(frozen=True)
class LocomoExample:
    """One answerable LOCOMO QA row plus its conversation."""

    task_id: str
    sample_id: str
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]
    conversation: tuple[ConversationTurn, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    """One memory retrieval result."""

    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    """Evaluation result for one task."""

    task_id: str
    question: str
    gold_answer: str
    prediction: str
    score: float
    passed: bool
    prompt_tokens: int
    completion_tokens: int
    retrieved: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    """Evaluation result for one scaffold/config candidate."""

    candidate_id: str
    scaffold_name: str
    passrate: float
    average_score: float
    token_consuming: int
    avg_token_consuming: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    count: int
    config: dict[str, Any]
    result_path: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateResult":
        """Build from current or legacy candidate-result JSON."""

        data = dict(payload)
        if "scaffold_name" not in data and "seed_name" in data:
            data["scaffold_name"] = data.pop("seed_name")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
