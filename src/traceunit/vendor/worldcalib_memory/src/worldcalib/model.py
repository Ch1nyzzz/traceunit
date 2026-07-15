"""OpenAI-compatible local model client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from worldcalib.schemas import RetrievalHit
from worldcalib.utils.text import estimate_tokens


DEFAULT_MODEL = "/data/home/yuhan/model_zoo/Qwen3-8B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


@dataclass(frozen=True)
class ModelResponse:
    """Text plus usage returned by a model call."""

    content: str
    prompt_tokens: int
    completion_tokens: int


class LocalModelClient:
    """Synchronous OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        timeout_s: int = 300,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        if (not api_key or api_key == "EMPTY") and "api.deepseek.com" in self.base_url:
            import os

            api_key = os.environ.get("DEEPSEEK_API_KEY", api_key)
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.chat_template_kwargs = (
            {"enable_thinking": False}
            if chat_template_kwargs is None
            else dict(chat_template_kwargs)
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = str(message.get("content") or "")
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(_messages_text(messages)))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
        return ModelResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def build_answer_messages(
    *,
    question: str,
    hits: list[RetrievalHit],
    category: int | None = None,
    max_context_chars: int = 6000,
) -> list[dict[str, str]]:
    """Build a grounded QA prompt from retrieved memory hits."""

    answer_instruction = _locomo_answer_instruction(category)
    if not hits:
        return [
            {
                "role": "system",
                "content": f"{answer_instruction} End with exactly one line: FINAL ANSWER: <answer>",
            },
            {
                "role": "user",
                "content": f"Question: {question}",
            },
        ]

    context_parts: list[str] = []
    used = 0
    for idx, hit in enumerate(hits, start=1):
        block = f"[{idx}] {hit.text}"
        if used + len(block) > max_context_chars:
            break
        context_parts.append(block)
        used += len(block)
    context = "\n\n".join(context_parts)
    return [
        {
            "role": "system",
            "content": (
                "You answer questions using only the retrieved memory context. "
                "If the context is insufficient, answer unknown. "
                f"{answer_instruction} End with exactly one line: FINAL ANSWER: <answer>"
            ),
        },
        {
            "role": "user",
            "content": f"Retrieved memory:\n{context}\n\nQuestion: {question}",
        },
    ]


def _locomo_answer_instruction(category: int | None) -> str:
    """Return LOCOMO-style answer guidance for the question category."""

    if category == 2:
        return (
            "Use the date of the conversation when answering temporal questions. "
            "Prefer the shortest answer that is directly supported by the context."
        )
    if category == 3:
        return "Write a short phrase and use exact words from the context whenever possible."
    if category == 5:
        return "Choose the answer supported by the context, or answer unknown if neither option is supported."
    return "Write a concise short answer, using exact words from the context whenever possible."


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages)
