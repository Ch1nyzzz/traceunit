"""MemGPT/Letta-style hierarchical memory scaffold.

This is a source-informed reproduction of the core MemGPT memory architecture,
not a dependency on the full Letta service stack. The upstream implementation
keeps editable core memory in context, stores all conversation history in recall
memory, stores long-term passages in archival memory, and compacts hidden
conversation history into a summary message when the context window is under
pressure. For LOCOMO, we make those same tiers explicit and deterministic.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from worldcalib.schemas import ConversationTurn, LocomoExample, RetrievalHit
from worldcalib.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from worldcalib.memory.scaffolds.bm25_scaffold import SimpleBM25Okapi
from worldcalib.utils.text import estimate_tokens, tokenize


MEMGPT_SOURCE_BUILD_FORMAT = "core_recall_archival_v1"
MEMGPT_SOURCE_IMPL = "MemGPT/Letta"


@dataclass(frozen=True)
class MemGPTMemoryBlock:
    label: str
    description: str
    value: str
    limit: int


@dataclass(frozen=True)
class MemGPTRecallMessage:
    role: str
    timestamp: str
    speaker: str
    dia_id: str
    session: str
    turn_index: int
    content: str

    def render(self) -> str:
        return (
            "{"
            f'"timestamp": "{self.timestamp}", '
            f'"role": "{self.role}", '
            f'"speaker": "{self.speaker}", '
            f'"dia_id": "{self.dia_id}", '
            f'"message": "{_json_safe(self.content)}"'
            "}"
        )


@dataclass(frozen=True)
class MemGPTArchivalPassage:
    passage_id: str
    text: str
    tags: tuple[str, ...]
    created_at: str
    turn_indices: tuple[int, ...]


@dataclass
class MemGPTHierarchicalState:
    core_memory: tuple[MemGPTMemoryBlock, ...]
    recall_messages: tuple[MemGPTRecallMessage, ...]
    archival_passages: tuple[MemGPTArchivalPassage, ...]
    summary_message: str | None
    active_turn_indices: tuple[int, ...]
    memory_metadata: str
    recall_index: SimpleBM25Okapi
    archival_index: SimpleBM25Okapi
    recall_docs: tuple[str, ...]
    archival_docs: tuple[str, ...]
    recall_tokens: tuple[list[str], ...]
    archival_tokens: tuple[list[str], ...]


class MemGPTSourceScaffold(RetrievalMemoryScaffold):
    """Hierarchical memory seed based on the MemGPT/Letta source architecture."""

    name = "memgpt_source"
    reference_urls = (
        "https://arxiv.org/abs/2310.08560",
        "https://github.com/cpacker/MemGPT.git",
        "https://github.com/letta-ai/letta.git",
    )

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> MemGPTHierarchicalState:
        recall_messages = tuple(_build_recall_messages(example))
        archival_passages = tuple(_build_archival_passages(example, config.extra))
        context_turns = max(1, int(config.extra.get("context_window_turns", 12)))
        active_turn_indices = tuple(turn.global_index for turn in example.conversation[-context_turns:])
        summary_message = _build_summary_message(example, hidden_count=max(0, len(example.conversation) - context_turns))
        core_memory = tuple(_build_core_memory(example, config.extra))
        memory_metadata = _compile_memory_metadata(
            example=example,
            recall_count=len(recall_messages),
            archival_count=len(archival_passages),
            archive_tags=sorted({tag for passage in archival_passages for tag in passage.tags})[:20],
        )

        recall_docs = tuple(message.render() for message in recall_messages)
        archival_docs = tuple(passage.text for passage in archival_passages)
        recall_tokens = tuple(tokenize(doc) for doc in recall_docs)
        archival_tokens = tuple(tokenize(doc) for doc in archival_docs)
        return MemGPTHierarchicalState(
            core_memory=core_memory,
            recall_messages=recall_messages,
            archival_passages=archival_passages,
            summary_message=summary_message,
            active_turn_indices=active_turn_indices,
            memory_metadata=memory_metadata,
            recall_index=SimpleBM25Okapi([list(tokens) for tokens in recall_tokens]),
            archival_index=SimpleBM25Okapi([list(tokens) for tokens in archival_tokens]),
            recall_docs=recall_docs,
            archival_docs=archival_docs,
            recall_tokens=recall_tokens,
            archival_tokens=archival_tokens,
        )

    def retrieve(
        self,
        state: MemGPTHierarchicalState,
        question: str,
        config: ScaffoldConfig,
    ) -> list[RetrievalHit]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        top_k = max(1, int(config.top_k))
        archival_limit = max(1, int(config.extra.get("archival_top_k", math.ceil(top_k * 0.55))))
        recall_limit = max(1, int(config.extra.get("recall_top_k", top_k - archival_limit)))
        rrf_k = float(config.extra.get("rrf_k", 60.0))

        hits: list[RetrievalHit] = []
        if bool(config.extra.get("include_core_memory", True)):
            hits.append(_core_hit(state, question))

        if state.summary_message and bool(config.extra.get("include_summary_memory", True)):
            summary_score = _cosine_score(query_tokens, tokenize(state.summary_message))
            if summary_score > 0:
                hits.append(
                    RetrievalHit(
                        text=state.summary_message,
                        score=0.2 + summary_score,
                        source=self.name,
                        metadata={
                            "memory_tier": "summary",
                            "source_impl": MEMGPT_SOURCE_IMPL,
                            "tool": "context_compaction",
                        },
                    )
                )

        archival_ranked = _hybrid_rank(
            query_tokens=query_tokens,
            docs_tokens=state.archival_tokens,
            bm25=state.archival_index,
            rrf_k=rrf_k,
        )
        for rank, scored in enumerate(archival_ranked[:archival_limit]):
            passage = state.archival_passages[scored.index]
            hits.append(
                RetrievalHit(
                    text=_format_archival_result(passage),
                    score=scored.score,
                    source=self.name,
                    metadata={
                        "memory_tier": "archival",
                        "source_impl": MEMGPT_SOURCE_IMPL,
                        "tool": "archival_memory_search",
                        "rank": rank,
                        "passage_id": passage.passage_id,
                        "tags": list(passage.tags),
                        "turn_indices": list(passage.turn_indices),
                        "search_mode": scored.search_mode,
                    },
                )
            )

        recall_ranked = _hybrid_rank(
            query_tokens=query_tokens,
            docs_tokens=state.recall_tokens,
            bm25=state.recall_index,
            rrf_k=rrf_k,
        )
        recall_indices = _expand_recall_indices(
            [scored.index for scored in recall_ranked[:recall_limit]],
            n=len(state.recall_messages),
            window=max(0, int(config.window)),
        )
        recall_scores = {scored.index: scored.score for scored in recall_ranked}
        for rank, idx in enumerate(recall_indices):
            message = state.recall_messages[idx]
            hits.append(
                RetrievalHit(
                    text=_format_recall_result(message),
                    score=recall_scores.get(idx, 0.01),
                    source=self.name,
                    metadata={
                        "memory_tier": "recall",
                        "source_impl": MEMGPT_SOURCE_IMPL,
                        "tool": "conversation_search",
                        "rank": rank,
                        "turn_index": message.turn_index,
                        "active_context": message.turn_index in state.active_turn_indices,
                    },
                )
            )

        return _dedupe_hits(hits)


@dataclass(frozen=True)
class _ScoredIndex:
    index: int
    score: float
    search_mode: str


def _build_core_memory(example: LocomoExample, extra: dict[str, Any]) -> list[MemGPTMemoryBlock]:
    benchmark = str(extra.get("benchmark") or "locomo")
    speakers = _unique(turn.speaker for turn in example.conversation)
    sessions = _unique(turn.session for turn in example.conversation)
    dates = [turn.session_date for turn in example.conversation if turn.session_date]
    first_date = dates[0] if dates else "unknown"
    last_date = dates[-1] if dates else "unknown"
    human_value = (
        f"{benchmark} conversation sample {example.sample_id} contains {len(example.conversation)} messages "
        f"across {len(sessions)} sessions from {first_date} to {last_date}. "
        f"Participants: {', '.join(speakers) if speakers else 'unknown'}."
    )
    if example.category == 2:
        human_value += " Temporal questions should use the session dates attached to recall messages."

    persona_value = str(
        extra.get("persona_block")
        or f"Answer {benchmark} questions by consulting core memory first, then archival and recall memory."
    )
    return [
        MemGPTMemoryBlock(
            label="persona",
            description="The agent role and memory-use policy.",
            value=persona_value,
            limit=int(extra.get("core_block_char_limit", 2000)),
        ),
        MemGPTMemoryBlock(
            label="human",
            description="Stable metadata about this conversation and its participants.",
            value=human_value,
            limit=int(extra.get("core_block_char_limit", 2000)),
        ),
    ]


def _build_recall_messages(example: LocomoExample) -> list[MemGPTRecallMessage]:
    first_speaker = example.conversation[0].speaker if example.conversation else ""
    messages: list[MemGPTRecallMessage] = []
    for turn in example.conversation:
        role = "user" if turn.speaker == first_speaker else "assistant"
        messages.append(
            MemGPTRecallMessage(
                role=role,
                timestamp=turn.session_date,
                speaker=turn.speaker,
                dia_id=turn.dia_id,
                session=turn.session,
                turn_index=turn.global_index,
                content=turn.text,
            )
        )
    return messages


def _build_archival_passages(example: LocomoExample, extra: dict[str, Any]) -> list[MemGPTArchivalPassage]:
    benchmark = str(extra.get("benchmark") or "locomo")
    chunk_size = max(1, int(extra.get("archival_chunk_size", 4)))
    overlap = max(0, int(extra.get("archival_chunk_overlap", 1)))
    passages: list[MemGPTArchivalPassage] = []
    for session, turns in _session_groups(example.conversation):
        step = max(1, chunk_size - overlap)
        for start in range(0, len(turns), step):
            chunk = turns[start : start + chunk_size]
            if not chunk:
                continue
            session_date = chunk[0].session_date
            speakers = _unique(turn.speaker for turn in chunk)
            body = "\n".join(f"- {turn.speaker} ({turn.dia_id}): {turn.text}" for turn in chunk)
            passage_text = (
                f"Archival memory inserted from {session} on {session_date}. "
                f"Speakers: {', '.join(speakers)}.\n{body}"
            )
            passages.append(
                MemGPTArchivalPassage(
                    passage_id=f"{example.sample_id}:{session}:{start}",
                    text=passage_text,
                    tags=tuple([benchmark, session, *speakers]),
                    created_at=session_date,
                    turn_indices=tuple(turn.global_index for turn in chunk),
                )
            )
            if start + chunk_size >= len(turns):
                break
    return passages


def _build_summary_message(example: LocomoExample, *, hidden_count: int) -> str | None:
    if hidden_count <= 0:
        return None
    hidden = example.conversation[:hidden_count]
    sessions = _unique(turn.session for turn in hidden)
    speakers = _unique(turn.speaker for turn in hidden)
    preview = "; ".join(_shorten(turn.text, 90) for turn in hidden[:5])
    return (
        "Note: prior messages have been hidden from view due to conversation memory constraints.\n"
        f"The following is a deterministic summary of the previous messages: "
        f"{hidden_count} older messages across {len(sessions)} sessions involving "
        f"{', '.join(speakers)}. Early topics include: {preview}."
    )


def _compile_memory_metadata(
    *,
    example: LocomoExample,
    recall_count: int,
    archival_count: int,
    archive_tags: list[str],
) -> str:
    latest_date = example.conversation[-1].session_date if example.conversation else "unknown"
    lines = [
        "<memory_metadata>",
        f"- AGENT_ID: worldcalib-{example.sample_id}",
        f"- CONVERSATION_ID: {example.task_id}",
        f"- System prompt last recompiled: {latest_date}",
        f"- {recall_count} previous messages between you and the user are stored in recall memory",
    ]
    if archival_count > 0:
        lines.append(f"- {archival_count} total memories you created are stored in archival memory (use tools to access them)")
    if archive_tags:
        lines.append(f"- Available archival memory tags: {', '.join(archive_tags)}")
    lines.append("</memory_metadata>")
    return "\n".join(lines)


def _compile_core_memory(state: MemGPTHierarchicalState) -> str:
    out = [state.memory_metadata, "<memory_blocks>", "The following memory blocks are currently engaged in your core memory unit:"]
    for block in state.core_memory:
        value = block.value or ""
        out.extend(
            [
                "",
                f"<{block.label}>",
                "<description>",
                block.description,
                "</description>",
                "<metadata>",
                f"- chars_current={len(value)}",
                f"- chars_limit={block.limit}",
                "</metadata>",
                "<value>",
                value,
                "</value>",
                f"</{block.label}>",
            ]
        )
    out.append("</memory_blocks>")
    return "\n".join(out)


def _core_hit(state: MemGPTHierarchicalState, question: str) -> RetrievalHit:
    text = _compile_core_memory(state)
    score = 0.1 + _cosine_score(tokenize(question), tokenize(text))
    return RetrievalHit(
        text=text,
        score=score,
        source=MemGPTSourceScaffold.name,
        metadata={
            "memory_tier": "core",
            "source_impl": MEMGPT_SOURCE_IMPL,
            "tool": "core_memory",
            "context_tokens": estimate_tokens(text),
        },
    )


def _hybrid_rank(
    *,
    query_tokens: list[str],
    docs_tokens: tuple[list[str], ...],
    bm25: SimpleBM25Okapi,
    rrf_k: float,
) -> list[_ScoredIndex]:
    if not docs_tokens:
        return []

    lexical_scores = list(bm25.get_scores(query_tokens))
    semantic_scores = [_cosine_score(query_tokens, tokens) for tokens in docs_tokens]
    lexical_order = _positive_rank_order(lexical_scores)
    semantic_order = _positive_rank_order(semantic_scores)

    combined: dict[int, float] = defaultdict(float)
    modes: dict[int, set[str]] = defaultdict(set)
    for rank, idx in enumerate(lexical_order):
        combined[idx] += 1.0 / (rrf_k + rank + 1.0)
        modes[idx].add("bm25")
    for rank, idx in enumerate(semantic_order):
        combined[idx] += 0.8 / (rrf_k + rank + 1.0)
        modes[idx].add("semantic")

    if not combined:
        return []

    return [
        _ScoredIndex(index=idx, score=score, search_mode="+".join(sorted(modes[idx])))
        for idx, score in sorted(combined.items(), key=lambda item: item[1], reverse=True)
    ]


def _positive_rank_order(scores: list[float]) -> list[int]:
    return [idx for idx in sorted(range(len(scores)), key=lambda item: scores[item], reverse=True) if scores[idx] > 0]


def _cosine_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    query = Counter(query_tokens)
    doc = Counter(doc_tokens)
    dot = sum(query[token] * doc.get(token, 0) for token in query)
    if dot <= 0:
        return 0.0
    query_norm = math.sqrt(sum(value * value for value in query.values()))
    doc_norm = math.sqrt(sum(value * value for value in doc.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)


def _expand_recall_indices(anchors: list[int], *, n: int, window: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for anchor in anchors:
        for idx in range(max(0, anchor - window), min(n, anchor + window + 1)):
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def _format_archival_result(passage: MemGPTArchivalPassage) -> str:
    return (
        "archival_memory_search result\n"
        f"id: {passage.passage_id}\n"
        f"created_at: {passage.created_at}\n"
        f"tags: {', '.join(passage.tags)}\n"
        f"content:\n{passage.text}"
    )


def _format_recall_result(message: MemGPTRecallMessage) -> str:
    return "conversation_search result\n" + message.render()


def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    seen: set[str] = set()
    out: list[RetrievalHit] = []
    for hit in sorted(hits, key=lambda item: _tier_sort_key(item), reverse=True):
        key = hit.text
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _tier_sort_key(hit: RetrievalHit) -> tuple[int, float]:
    tier = str(hit.metadata.get("memory_tier") or "")
    priority = {
        "core": 4,
        "summary": 3,
        "archival": 2,
        "recall": 1,
    }.get(tier, 0)
    return priority, float(hit.score)


def _session_groups(turns: Iterable[ConversationTurn]) -> list[tuple[str, list[ConversationTurn]]]:
    groups: list[tuple[str, list[ConversationTurn]]] = []
    by_session: dict[str, list[ConversationTurn]] = {}
    for turn in turns:
        if turn.session not in by_session:
            by_session[turn.session] = []
            groups.append((turn.session, by_session[turn.session]))
        by_session[turn.session].append(turn)
    return groups


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _json_safe(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
