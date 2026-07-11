"""Host-controlled declarative model probes.

A probe is data, never code: the frozen packet ships a JSON file, and the
host process - the only party holding credentials - renders it against the
subject source tree, performs one budget-capped temperature-0 chat
completion, and checks the declared expectations. Generated test code is
never executed here and never sees an API key.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from traceunit.io import safe_relative_path, write_json
from traceunit.models import TestCaseSpec, TestExecution, TestExecutionMode

Transport = Callable[[str, dict[str, str], dict[str, Any]], Mapping[str, Any]]

_SOURCE_TEMPLATE = re.compile(r"\{\{source_file:([^}]+)\}\}")
_MAX_INLINE_CHARS = 20_000
_ALLOWED_ROLES = {"system", "user", "assistant"}
_ALLOWED_EXPECTATION_KINDS = {"regex", "contains"}


class ProbeSpecError(ValueError):
    pass


def run_declarative_probe(
    *,
    case: TestCaseSpec,
    bundle: Path,
    source: Path,
    subject: str,
    output_dir: Path,
    model: str,
    base_url: str,
    api_key_env: str,
    transport: Transport | None = None,
) -> TestExecution:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result_path = output_dir / f"{case.case_id}_{subject}_probe.json"
    error_path = output_dir / f"{case.case_id}_{subject}_probe_error.txt"
    record: dict[str, Any] = {"case_id": case.case_id, "subject": subject}
    calls = 0
    tokens = 0

    def finish(passed: bool, error: str = "") -> TestExecution:
        record.update(
            {
                "passed": passed,
                "error": error,
                "model_calls": calls,
                "reported_tokens": min(tokens, case.max_tokens),
                "actual_tokens": tokens,
            }
        )
        write_json(result_path, record)
        if error:
            error_path.write_text(error + "\n", encoding="utf-8")
        return TestExecution(
            case_id=case.case_id,
            tier=case.tier,
            evidence_role=case.evidence_role,
            execution_mode=TestExecutionMode.MODEL_BACKED_PROBE,
            subject=subject,
            passed=passed,
            returncode=None,
            duration_s=round(time.monotonic() - started, 3),
            stdout_path=str(result_path),
            stderr_path=str(error_path),
            error=error,
            model_calls=min(calls, case.max_model_calls),
            # The runtime rejects results above the frozen budget, so budget
            # overruns are reported as a capped, failed execution; the real
            # usage stays in the artifact.
            tokens=min(tokens, case.max_tokens),
        )

    try:
        spec = _load_spec(bundle, case)
        messages = [
            {
                "role": message["role"],
                "content": _render(message["content"], source),
            }
            for message in spec["messages"]
        ]
    except (ProbeSpecError, json.JSONDecodeError, OSError) as exc:
        return finish(False, f"invalid probe specification: {exc}")
    record["messages"] = messages

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return finish(False, f"probe credential is not available: {api_key_env}")

    url = _chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max(1, case.max_tokens),
    }
    send = transport or _http_transport(timeout_s=max(30, case.timeout_s))
    response: Mapping[str, Any] | None = None
    last_error = ""
    attempts = min(2, max(1, case.max_model_calls))
    for _ in range(attempts):
        calls += 1
        try:
            response = send(url, headers, payload)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    if response is None:
        return finish(False, f"probe transport failed: {last_error}")

    try:
        reply = str(response["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return finish(False, "probe response has no assistant message")
    usage = response.get("usage") or {}
    tokens = int(usage.get("total_tokens") or 0)
    record["reply"] = reply
    record["usage"] = dict(usage)

    checks: list[dict[str, Any]] = []
    all_ok = True
    for expectation in spec["expect"]:
        matched = _matches(expectation, reply)
        ok = matched != bool(expectation.get("negate"))
        all_ok = all_ok and ok
        checks.append({**expectation, "ok": ok})
    record["expectations"] = checks

    if tokens > case.max_tokens:
        return finish(False, "probe exceeded its frozen token budget")
    return finish(all_ok)


def _load_spec(bundle: Path, case: TestCaseSpec) -> dict[str, Any]:
    path = safe_relative_path(bundle, case.path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ProbeSpecError("probe file must contain a JSON object")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProbeSpecError("probe requires a non-empty messages list")
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or str(message.get("role")) not in _ALLOWED_ROLES
            or not isinstance(message.get("content"), str)
        ):
            raise ProbeSpecError(
                "each message needs a system/user/assistant role and string content"
            )
    if str(messages[-1].get("role")) == "assistant":
        raise ProbeSpecError(
            "the last message must be a user or system turn; the live completion "
            "is the assistant reply under test"
        )
    expectations = raw.get("expect")
    if not isinstance(expectations, list) or not expectations:
        raise ProbeSpecError("probe requires a non-empty expect list")
    for expectation in expectations:
        if (
            not isinstance(expectation, Mapping)
            or str(expectation.get("kind")) not in _ALLOWED_EXPECTATION_KINDS
        ):
            raise ProbeSpecError("expectations must use kind 'regex' or 'contains'")
        if expectation.get("kind") == "regex":
            try:
                re.compile(str(expectation.get("pattern") or ""))
            except re.error as exc:
                raise ProbeSpecError(f"invalid expectation regex: {exc}") from exc
        elif not str(expectation.get("value") or ""):
            raise ProbeSpecError("contains expectations need a non-empty value")
    return {"messages": messages, "expect": expectations}


def _render(text: str, source: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        relative = match.group(1).strip()
        try:
            path = safe_relative_path(source, relative)
        except ValueError as exc:
            raise ProbeSpecError(str(exc)) from exc
        if not path.is_file():
            raise ProbeSpecError(f"probe references a missing source file: {relative}")
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_INLINE_CHARS:
            content = content[:_MAX_INLINE_CHARS] + "\n[truncated]\n"
        return content

    return _SOURCE_TEMPLATE.sub(replace, text)


def _matches(expectation: Mapping[str, Any], reply: str) -> bool:
    if expectation.get("kind") == "regex":
        return re.search(str(expectation.get("pattern") or ""), reply) is not None
    return str(expectation.get("value") or "") in reply


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def _http_transport(*, timeout_s: int) -> Transport:
    def send(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as raw:
            return json.loads(raw.read().decode("utf-8"))

    return send
