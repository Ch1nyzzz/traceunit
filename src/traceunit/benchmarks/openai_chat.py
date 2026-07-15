"""Minimal OpenAI-compatible chat client (standard library only).

Used by the native HLE runner for both the frozen solver model and the hidden
LLM judge. Keeping it dependency-free means the HLE worker stays self-contained
and importable under any interpreter, mirroring how ``harbor_worker`` avoids
depending on ``traceunit``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence


class ChatError(RuntimeError):
    """Raised when the chat endpoint cannot be reached or returns an error."""


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    timeout_s: int = 300,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call ``POST {base_url}/chat/completions`` and return content + usage.

    Returns ``{"content", "prompt_tokens", "completion_tokens"}``.
    """

    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if extra_body:
        payload.update(dict(extra_body))
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise ChatError(f"chat endpoint HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChatError(f"chat endpoint unreachable: {exc}") from exc

    choices = body.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    return {
        "content": content,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }
