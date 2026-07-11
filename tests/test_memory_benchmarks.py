from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from traceunit.benchmarks.common import normalize_worldcalib_result
from traceunit.benchmarks.factory import build_benchmark
from traceunit.benchmarks.memory import (
    LocomoAdapter,
    LongMemEvalAdapter,
    _public_memory_example,
    _split_clustered_task_ids,
    _validate_memory_pools,
)
from traceunit.config import BenchmarkConfig, load_config


def test_memory_cluster_split_is_stable_and_disjoint() -> None:
    task_to_cluster = {
        f"{cluster}-{index}": cluster
        for cluster in ("alpha", "beta", "gamma", "delta", "epsilon")
        for index in range(3)
    }
    first = _split_clustered_task_ids(task_to_cluster, seed=7, search_fraction=0.6)
    second = _split_clustered_task_ids(
        dict(reversed(list(task_to_cluster.items()))),
        seed=7,
        search_fraction=0.6,
    )
    assert first == second
    search, final = first
    assert set(search).isdisjoint(final)
    assert set(search) | set(final) == set(task_to_cluster)
    assert {task_to_cluster[item] for item in search}.isdisjoint(
        {task_to_cluster[item] for item in final}
    )


def test_memory_pool_validation_rejects_shared_conversation() -> None:
    with pytest.raises(ValueError, match="conversation clusters overlap"):
        _validate_memory_pools(
            ["a-1"],
            ["a-2"],
            {"a-1": "conversation-a", "a-2": "conversation-a"},
        )


def test_memory_adapters_are_registered() -> None:
    assert isinstance(build_benchmark(BenchmarkConfig(name="locomo")), LocomoAdapter)
    assert isinstance(
        build_benchmark(BenchmarkConfig(name="longmemeval")), LongMemEvalAdapter
    )
    assert isinstance(build_benchmark(BenchmarkConfig(name="lme")), LongMemEvalAdapter)


def test_longmemeval_config_parses_memory_options(tmp_path: Path) -> None:
    config_path = tmp_path / "longmemeval.yaml"
    config_path.write_text(
        """
loop:
  run_dir: run
benchmark:
  name: longmemeval
  data_path: data/lme.json
  dataset_variant: m
  memory_question_types: temporal-reasoning, knowledge-update
  memory_top_k: 9
  memory_window: 2
  max_context_chars: 7000
  judge_model: judge-model
  judge_api_key_env: JUDGE_KEY
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.benchmark.name == "longmemeval"
    assert config.benchmark.data_path == (tmp_path / "data/lme.json").resolve()
    assert config.benchmark.dataset_variant == "m"
    assert config.benchmark.memory_question_types == (
        "temporal-reasoning",
        "knowledge-update",
    )
    assert config.benchmark.memory_top_k == 9
    assert config.benchmark.judge_model == "judge-model"
    assert config.benchmark.judge_api_key_env == "JUDGE_KEY"


def test_lme_config_alias_normalizes_to_longmemeval(tmp_path: Path) -> None:
    config_path = tmp_path / "lme.yaml"
    config_path.write_text(
        "loop:\n  run_dir: run\nbenchmark:\n  name: lme\n",
        encoding="utf-8",
    )
    assert load_config(config_path).benchmark.name == "longmemeval"


def test_memory_policy_rejects_raw_data_and_gold_access() -> None:
    adapter = LocomoAdapter(BenchmarkConfig(name="locomo"))
    violations = adapter.policy_violations(
        Path("source"),
        """+++ b/source.py
+answer = example.answer
+load_locomo_examples()
+open('/tmp/locomo10.json')
""",
    )
    assert any("gold answer" in item for item in violations)
    assert any("raw LoCoMo" in item for item in violations)
    assert any("filesystem" in item for item in violations)


@dataclass(frozen=True)
class _Example:
    task_id: str
    sample_id: str
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]
    conversation: tuple[str, ...]
    metadata: dict[str, object]


def test_public_memory_example_redacts_gold_identity_and_evidence() -> None:
    redacted = _public_memory_example(
        _Example(
            task_id="private-task",
            sample_id="private-sample",
            question="What happened?",
            answer="secret gold",
            category=2,
            evidence=("supporting-session",),
            conversation=("public conversation",),
            metadata={"question_type": "temporal", "question_id": "private-id"},
        )
    )
    assert redacted.task_id == "traceunit-redacted"
    assert redacted.sample_id == "traceunit-redacted"
    assert redacted.answer == ""
    assert redacted.evidence == ()
    assert redacted.metadata == {"question_type": "temporal"}


def test_normalizer_surfaces_retrieval_without_gold_answer(tmp_path: Path) -> None:
    result = tmp_path / "candidate_result.json"
    result.write_text(
        json.dumps(
            {
                "candidate": {
                    "candidate_id": "candidate",
                    "average_score": 0.0,
                    "passrate": 0.0,
                    "token_consuming": 11,
                    "count": 1,
                },
                "tasks": [
                    {
                        "task_id": "LOCOMO::sample::qa::0",
                        "question": "Where did Alex go?",
                        "gold_answer": "secret gold answer",
                        "prediction": "I do not know.",
                        "score": 0.0,
                        "passed": False,
                        "retrieved": [
                            {
                                "text": "[session] Alex went to the library.",
                                "score": 1.0,
                                "source": "memgpt",
                                "metadata": {"memory_tier": "recall", "ignored": "x"},
                            }
                        ],
                        "metadata": {"scoring_method": "token_f1"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluation = normalize_worldcalib_result(
        result_path=result,
        benchmark="locomo",
        split="search",
        candidate_id="candidate",
        out_dir=tmp_path / "out",
    )
    row = json.loads(Path(evaluation.trace_path).read_text(encoding="utf-8"))
    assert row["events"][0]["kind"] == "retrieval"
    assert row["events"][0]["output"]["documents"][0]["metadata"] == {
        "memory_tier": "recall"
    }
    assert "secret gold answer" not in json.dumps(row)
