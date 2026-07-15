"""AppWorld minimal ReAct *code* agent — the editable seed scaffold.

This module is intentionally **worldcalib-free**: AppWorld pins pydantic v1 /
SQLAlchemy 1.4, which conflict with the main worldcalib venv (pydantic v2), so the
whole AppWorld eval runs in the isolated ``.venv-appworld`` interpreter. Keep this
file importable with only the stdlib + ``openai`` + ``appworld`` so the eval
subprocess can load it without importing worldcalib.

The agent is a faithful *minimal* ReAct code agent (the official AppWorld
recommended starting point): each turn the model emits ONE ```python block, the
block is executed in the AppWorld IPython shell, and the stdout/result is fed
back. The agent discovers APIs through ``apis.api_docs`` and finishes by calling
``apis.supervisor.complete_task(...)``. It deliberately contains NONE of the
high-value levers (API-doc retrieval policy, reflection, planning, error-feedback
shaping, state summaries) — those are exactly the search space the optimizer's
proposer is meant to discover.

The single editable seam is this file. The grading side (AppWorld env creation +
``world.evaluate()``) lives in the eval harness and is never handed to a candidate.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

# ── SUT (locked) chat client ────────────────────────────────────────────────
# The SUT model is frozen; only the endpoint/key come from env. The AppWorld
# deepseek-v4-flash endpoint is the same unified provider the other backends use.

_MODEL = os.environ.get("APPWORLD_MODEL") or os.environ.get("MODEL_NAME") or "deepseek-v4-flash"
_BASE_URL = (
    os.environ.get("APPWORLD_OPENAI_BASE_URL")
    or os.environ.get("TOOLATHLON_OPENAI_BASE_URL")
    or os.environ.get("DEEPSEEK_BASE_URL")
    or "https://api.deepseek.com"
)
_API_KEY = (
    os.environ.get("APPWORLD_OPENAI_API_KEY")
    or os.environ.get("TOOLATHLON_OPENAI_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
)

MAX_STEPS = 100  # max ReAct turns — AppWorld paper's official ReAct budget (100 LLM calls)
MAX_TOKENS = 4000  # per call; deepseek-v4-flash is a reasoning model
OUTPUT_CLIP = 6000  # chars of execution output fed back per turn


def _client():
    from openai import OpenAI

    if not _API_KEY:
        raise RuntimeError(
            "No SUT API key: set APPWORLD_OPENAI_API_KEY / TOOLATHLON_OPENAI_API_KEY / DEEPSEEK_API_KEY"
        )
    return OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a model turn (```python preferred)."""
    if not text:
        return ""
    m = _CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def build_system_prompt(instruction: str, supervisor: dict[str, Any], app_descriptions: dict[str, str]) -> str:
    sup = supervisor or {}
    sup_line = (
        f"{sup.get('first_name','')} {sup.get('last_name','')}".strip()
        + f" | email: {sup.get('email','?')} | phone: {sup.get('phone_number','?')}"
    )
    apps = "\n".join(f"  - {name}: {desc}" for name, desc in (app_descriptions or {}).items())
    return (
        "You are a coding agent. You complete your supervisor's task by writing "
        "Python code that calls app APIs. Your code runs in a STATEFUL IPython "
        "shell: variables persist across turns, and you see each snippet's output "
        "before writing the next.\n\n"
        f"# Supervisor (you act on their behalf)\n{sup_line}\n\n"
        f"# Available apps\n{apps}\n\n"
        "# How to work\n"
        "- Discover an app's APIs:\n"
        "    print(apis.api_docs.show_api_descriptions(app_name='spotify'))\n"
        "- See one API's exact spec (arguments, response):\n"
        "    print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))\n"
        "- Most apps need an access_token; get the supervisor's credentials via the\n"
        "  supervisor app (read its api_docs first), then log in, e.g.\n"
        "    apis.<app>.login(username=..., password=...)\n"
        "- Call any API as apis.<app>.<api>(<args>).\n"
        "- When (and ONLY when) the task is fully done, finish with:\n"
        "    apis.supervisor.complete_task()                 # action tasks\n"
        "    apis.supervisor.complete_task(answer=<value>)   # if an answer is asked for\n\n"
        "# Output format (strict)\n"
        "Reply with EXACTLY ONE Python code block per turn and nothing else, e.g.\n"
        "```python\n"
        "print(apis.api_docs.show_api_descriptions(app_name='spotify'))\n"
        "```\n"
        "Inspect each output before proceeding. Do not call complete_task until the "
        "task is actually finished.\n\n"
        f"# Your supervisor's task\n{instruction}\n"
    )


def solve(world: Any, *, chat: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Drive one AppWorld task to completion. Returns telemetry (never raises).

    ``world`` is a live ``appworld.AppWorld``; we use only ``world.execute`` and
    ``world.task_completed`` (plus read-only ``world.task`` fields for the prompt)
    — never ``world.evaluate`` / ``world.task.ground_truth`` (that is grading).
    """
    task = world.task
    supervisor = task.supervisor
    sup = {
        "first_name": getattr(supervisor, "first_name", ""),
        "last_name": getattr(supervisor, "last_name", ""),
        "email": getattr(supervisor, "email", ""),
        "phone_number": getattr(supervisor, "phone_number", ""),
    }
    messages = [
        {"role": "system", "content": build_system_prompt(task.instruction, sup, dict(task.app_descriptions))},
        {"role": "user", "content": "Write your first Python code block."},
    ]
    client = _client()
    ptok = ctok = 0
    steps = 0
    last_error: str | None = None

    for steps in range(1, MAX_STEPS + 1):
        try:
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                timeout=150,
            )
        except Exception as e:  # noqa: BLE001 — surface as episode error
            last_error = f"{type(e).__name__}: {e}"
            break

        usage = getattr(resp, "usage", None)
        ptok += int(getattr(usage, "prompt_tokens", 0) or 0)
        ctok += int(getattr(usage, "completion_tokens", 0) or 0)
        content = resp.choices[0].message.content or ""
        code = extract_code(content)

        if not code:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "Reply with exactly one ```python code block and nothing else.",
            })
            continue

        try:
            output = world.execute(code)
        except Exception as e:  # noqa: BLE001 — a bad snippet must not kill the episode
            output = f"Execution error: {type(e).__name__}: {e}"
        output = (output or "")[:OUTPUT_CLIP]

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Output:\n{output}"})

        try:
            if world.task_completed():
                break
        except Exception:  # noqa: BLE001
            pass

    return {
        "steps": steps,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "error": last_error,
        # Full ReAct conversation (system + every assistant code turn + every
        # execution output). The eval harness dumps this as raw evidence so the
        # proposer can read exactly what code the agent wrote and what failed.
        "transcript": messages,
    }
