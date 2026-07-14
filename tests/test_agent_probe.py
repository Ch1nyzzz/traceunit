from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceunit.agent_probe import run_declarative_probe
from traceunit.models import EvidenceRole, TestCaseSpec, TestTier


def _case(**overrides) -> TestCaseSpec:
    values = {
        "case_id": "capability_probe",
        "tier": TestTier.HIDDEN,
        "evidence_role": EvidenceRole.STRUCTURAL_SIBLING,
        "path": "tests/hidden/probe.json",
        "driver": "agent_probe",
        "execution_mode": "model_backed_probe",
        "max_model_calls": 2,
        "max_tokens": 4096,
        "timeout_s": 60,
    }
    values.update(overrides)
    return TestCaseSpec.from_dict(values)


def _bundle(tmp_path: Path, spec: dict) -> Path:
    bundle = tmp_path / "bundle"
    probe = bundle / "tests/hidden/probe.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(json.dumps(spec), encoding="utf-8")
    return bundle


def _source(tmp_path: Path, policy: str) -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "policy.md").write_text(policy, encoding="utf-8")
    return source


def _transport(reply: str, *, tokens: int = 100, calls: list | None = None):
    def send(url, headers, payload):
        if calls is not None:
            calls.append(payload)
        assert headers["Authorization"].startswith("Bearer ")
        assert payload["temperature"] == 0.0
        return {
            "choices": [{"message": {"content": reply}}],
            # completion_tokens drives the budget; total includes the prompt
            # and must not count against the reply cap.
            "usage": {
                "completion_tokens": tokens,
                "total_tokens": tokens + 10_000,
            },
        }

    return send


SPEC = {
    "description": "does the scaffolded model surface a counterexample",
    "messages": [
        {"role": "system", "content": "Follow this policy:\n{{source_file:policy.md}}"},
        {"role": "user", "content": "Review: def add(a, b): return a - b"},
    ],
    "expect": [
        {"kind": "regex", "pattern": "counterexample|add\\(1, 1\\)"},
        {"kind": "contains", "value": "praise", "negate": True},
    ],
}


def test_probe_renders_source_and_passes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROBE_KEY", "secret")
    calls: list = []
    result = run_declarative_probe(
        case=_case(),
        bundle=_bundle(tmp_path, SPEC),
        source=_source(tmp_path, "Always produce a counterexample."),
        subject="candidate",
        output_dir=tmp_path / "out",
        model="test-model",
        base_url="https://example.invalid",
        api_key_env="PROBE_KEY",
        transport=_transport("Here is a counterexample: add(1, 1) == 0", calls=calls),
    )
    assert result.passed and not result.error
    assert result.model_calls == 1 and result.tokens == 100
    assert "Always produce a counterexample." in calls[0]["messages"][0]["content"]
    record = json.loads(Path(result.stdout_path).read_text(encoding="utf-8"))
    assert record["expectations"][0]["ok"] and record["expectations"][1]["ok"]


def test_probe_fails_on_unmet_expectation_without_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROBE_KEY", "secret")
    result = run_declarative_probe(
        case=_case(),
        bundle=_bundle(tmp_path, SPEC),
        source=_source(tmp_path, "policy"),
        subject="incumbent",
        output_dir=tmp_path / "out",
        model="test-model",
        base_url="https://example.invalid",
        api_key_env="PROBE_KEY",
        transport=_transport("Looks great, nothing but praise."),
    )
    assert not result.passed and not result.error


def test_probe_fails_closed_without_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PROBE_KEY", raising=False)
    result = run_declarative_probe(
        case=_case(),
        bundle=_bundle(tmp_path, SPEC),
        source=_source(tmp_path, "policy"),
        subject="candidate",
        output_dir=tmp_path / "out",
        model="test-model",
        base_url="https://example.invalid",
        api_key_env="PROBE_KEY",
        transport=_transport("irrelevant"),
    )
    assert not result.passed and "PROBE_KEY" in result.error
    assert result.model_calls == 0


def test_probe_rejects_source_escape_and_missing_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROBE_KEY", "secret")
    for content in ("{{source_file:../secrets.txt}}", "{{source_file:absent.md}}"):
        spec = {**SPEC, "messages": [{"role": "user", "content": content}]}
        result = run_declarative_probe(
            case=_case(),
            bundle=_bundle(tmp_path / content[14:20].strip("./"), spec),
            source=_source(tmp_path, "policy"),
            subject="candidate",
            output_dir=tmp_path / "out",
            model="test-model",
            base_url="https://example.invalid",
            api_key_env="PROBE_KEY",
            transport=_transport("irrelevant"),
        )
        assert not result.passed and "invalid probe specification" in result.error
        assert result.model_calls == 0


def test_probe_budget_overrun_fails_and_caps_reported_tokens(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PROBE_KEY", "secret")
    result = run_declarative_probe(
        case=_case(max_tokens=50),
        bundle=_bundle(tmp_path, SPEC),
        source=_source(tmp_path, "policy"),
        subject="candidate",
        output_dir=tmp_path / "out",
        model="test-model",
        base_url="https://example.invalid",
        api_key_env="PROBE_KEY",
        transport=_transport("counterexample: add(1, 1)", tokens=5000),
    )
    assert not result.passed and "token budget" in result.error
    assert result.tokens == 50
    record = json.loads(Path(result.stdout_path).read_text(encoding="utf-8"))
    assert record["actual_tokens"] == 5000


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"messages": [], "expect": SPEC["expect"]}, "non-empty messages"),
        ({"messages": SPEC["messages"], "expect": []}, "non-empty expect"),
        (
            {
                "messages": [{"role": "assistant", "content": "scripted"}],
                "expect": SPEC["expect"],
            },
            "user or system turn",
        ),
        (
            {
                "messages": SPEC["messages"],
                "expect": [{"kind": "regex", "pattern": "["}],
            },
            "invalid expectation regex",
        ),
    ],
)
def test_probe_rejects_malformed_specs(tmp_path, monkeypatch, spec, message) -> None:
    monkeypatch.setenv("PROBE_KEY", "secret")
    result = run_declarative_probe(
        case=_case(),
        bundle=_bundle(tmp_path, spec),
        source=_source(tmp_path, "policy"),
        subject="candidate",
        output_dir=tmp_path / "out",
        model="test-model",
        base_url="https://example.invalid",
        api_key_env="PROBE_KEY",
        transport=_transport("irrelevant"),
    )
    assert not result.passed and message in result.error
