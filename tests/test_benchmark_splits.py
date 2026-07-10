from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from traceunit.benchmarks.appworld import (
    AppWorldAdapter,
    _sandboxed_appworld_command,
    _split_heldout_scenarios,
)
from traceunit.benchmarks.swebench import (
    SwebenchVerifiedAdapter,
    _augment_miniswe_trajectories,
    _evaluation_cache_fingerprint,
    _finalize_evaluation,
    _official_eval_identity,
    _representative_order,
    _run_official_patch_evaluation,
    _split_rows,
    _validate_disjoint_pools,
)
from traceunit.config import BenchmarkConfig, load_config
from traceunit.models import BenchmarkEvaluation, TaskOutcome


def test_swebench_split_is_stable_and_disjoint() -> None:
    rows = [{"instance_id": f"repo__issue-{i}"} for i in range(100)]
    first = _split_rows(rows, seed=7, diagnostic_fraction=0.6, canary_fraction=0.2)
    second = _split_rows(rows, seed=7, diagnostic_fraction=0.6, canary_fraction=0.2)
    assert first == second
    ids = [{row["instance_id"] for row in first[name]} for name in first]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert set.union(*ids) == {row["instance_id"] for row in rows}


def test_swebench_representative_order_is_input_independent_and_interleaved() -> None:
    rows = [
        {"instance_id": f"{repo}__issue-{index}", "repo": repo}
        for repo in ("org/alpha", "org/beta", "org/gamma")
        for index in range(4)
    ]
    first = _representative_order(rows, seed=13, namespace="diagnostic")
    second = _representative_order(
        list(reversed(rows)), seed=13, namespace="diagnostic"
    )
    assert [row["instance_id"] for row in first] == [
        row["instance_id"] for row in second
    ]
    assert len({row["repo"] for row in first[:3]}) == 3


def test_swebench_rejects_overlapping_explicit_pools() -> None:
    with pytest.raises(ValueError, match="appears in both"):
        _validate_disjoint_pools(
            {
                "diagnostic": [{"instance_id": "org__issue-1"}],
                "canary": [{"instance_id": "org__issue-1"}],
                "audit": [],
            }
        )


def test_swebench_prepare_strips_private_fields(tmp_path: Path) -> None:
    worldcalib = tmp_path / "worldcalib"
    (worldcalib / "src/worldcalib/coding").mkdir(parents=True)
    (worldcalib / "src/worldcalib/coding/swebench.py").write_text("", encoding="utf-8")
    seed = worldcalib / "references/vendor/mini-swe-agent"
    seed.mkdir(parents=True)
    rows = [
        {
            "instance_id": f"repo__issue-{i}",
            "problem_statement": "public",
            "repo": "org/repo",
            "base_commit": "abc",
            "patch": "SECRET",
            "test_patch": "PRIVATE",
        }
        for i in range(20)
    ]
    data = tmp_path / "verified.json"
    data.write_text(json.dumps(rows), encoding="utf-8")
    adapter = SwebenchVerifiedAdapter(
        BenchmarkConfig(
            name="swebench_verified",
            worldcalib_root=worldcalib,
            seed_source_path=seed,
            diagnostic_data_path=data,
        )
    )
    run = tmp_path / "run"
    adapter.prepare(run)
    text = (run / "benchmark_data/swebench_verified/diagnostic.json").read_text()
    assert "SECRET" not in text
    assert "PRIVATE" not in text
    assert "problem_statement" in text


def test_swebench_cache_fingerprint_binds_pool_limit_config_and_harness(
    tmp_path: Path,
) -> None:
    worldcalib = tmp_path / "worldcalib"
    runner = worldcalib / "src/worldcalib/coding/swebench.py"
    entry = worldcalib / "scripts/run_miniswe_swebench_single.py"
    runner.parent.mkdir(parents=True)
    entry.parent.mkdir(parents=True)
    runner.write_text("runner-v1", encoding="utf-8")
    entry.write_text("entry-v1", encoding="utf-8")
    pool = tmp_path / "pool.json"
    pool.write_text('[{"instance_id":"a"}]', encoding="utf-8")
    config = BenchmarkConfig(
        name="swebench_verified",
        worldcalib_root=worldcalib,
        model="model-a",
    )

    first, payload = _evaluation_cache_fingerprint(
        source_hash="source-a",
        pool_path=pool,
        split="diagnostic",
        limit=5,
        config=config,
    )
    different_limit, _ = _evaluation_cache_fingerprint(
        source_hash="source-a",
        pool_path=pool,
        split="diagnostic",
        limit=6,
        config=config,
    )
    different_model, _ = _evaluation_cache_fingerprint(
        source_hash="source-a",
        pool_path=pool,
        split="diagnostic",
        limit=5,
        config=BenchmarkConfig(
            name="swebench_verified",
            worldcalib_root=worldcalib,
            model="model-b",
        ),
    )
    pool.write_text('[{"instance_id":"b"}]', encoding="utf-8")
    different_pool, _ = _evaluation_cache_fingerprint(
        source_hash="source-a",
        pool_path=pool,
        split="diagnostic",
        limit=5,
        config=config,
    )
    pool.write_text('[{"instance_id":"a"}]', encoding="utf-8")
    runner.write_text("runner-v2", encoding="utf-8")
    different_harness, _ = _evaluation_cache_fingerprint(
        source_hash="source-a",
        pool_path=pool,
        split="diagnostic",
        limit=5,
        config=config,
    )

    assert payload["source_sha256"] == "source-a"
    assert (
        len(
            {first, different_limit, different_model, different_pool, different_harness}
        )
        == 5
    )


def test_swebench_default_evaluator_is_adapter_owned(tmp_path: Path) -> None:
    adapter = SwebenchVerifiedAdapter(
        BenchmarkConfig(name="swebench_verified", worldcalib_root=tmp_path)
    )
    command = adapter._default_eval_command(attempt_id="candidate-attempt")
    assert "_eval-patch" in command
    assert "candidate-attempt" in command
    assert "run_miniswe_swebench_single.py" not in command


def test_swebench_official_identity_binds_patch_and_attempt() -> None:
    first = _official_eval_identity(
        attempt_id="candidate-a", instance_id="org__issue-1", patch_text="patch-a"
    )
    assert first == _official_eval_identity(
        attempt_id="candidate-a", instance_id="org__issue-1", patch_text="patch-a"
    )
    assert first != _official_eval_identity(
        attempt_id="candidate-a", instance_id="org__issue-1", patch_text="patch-b"
    )
    assert first != _official_eval_identity(
        attempt_id="candidate-b", instance_id="org__issue-1", patch_text="patch-a"
    )
    long_prefix = "candidate-" + "x" * 100
    assert _official_eval_identity(
        attempt_id=long_prefix + "a",
        instance_id="org__issue-1",
        patch_text="patch-a",
    ) != _official_eval_identity(
        attempt_id=long_prefix + "b",
        instance_id="org__issue-1",
        patch_text="patch-a",
    )


def test_swebench_empty_patch_is_not_evaluator_success(tmp_path: Path) -> None:
    instance = tmp_path / "instance.json"
    patch = tmp_path / "patch.diff"
    task_dir = tmp_path / "task"
    instance.write_text('{"task_id":"org__issue-1"}', encoding="utf-8")
    patch.write_text("\n", encoding="utf-8")

    returncode = _run_official_patch_evaluation(
        instance_path=instance,
        patch_path=patch,
        task_dir=task_dir,
        attempt_id="candidate-a",
        dataset_name="dataset",
        dataset_split="test",
        timeout_s=30,
    )

    verdict = json.loads((task_dir / "official_verdict.json").read_text())
    assert returncode == 1
    assert verdict["status"] == "empty_patch"
    assert verdict["resolved"] is False


def test_swebench_official_evaluator_reads_only_its_unique_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = tmp_path / "instance.json"
    patch = tmp_path / "patch.diff"
    task_dir = tmp_path / "task"
    instance.write_text('{"task_id":"org__issue-1"}', encoding="utf-8")
    patch.write_text("diff --git a/a b/a\n", encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        run_id = command[command.index("-id") + 1]
        instance_id = command[command.index("-i") + 1]
        prediction_path = Path(command[command.index("-p") + 1])
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        model_name = prediction[instance_id]["model_name_or_path"]
        report = (
            Path(str(kwargs["cwd"]))
            / "logs"
            / "run_evaluation"
            / run_id
            / model_name.replace("/", "__")
            / instance_id
            / "report.json"
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps({instance_id: {"resolved": True}}), encoding="utf-8"
        )
        calls.append((run_id, model_name))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("traceunit.benchmarks.swebench.subprocess.run", fake_run)
    first_code = _run_official_patch_evaluation(
        instance_path=instance,
        patch_path=patch,
        task_dir=task_dir,
        attempt_id="candidate-a",
        dataset_name="dataset",
        dataset_split="test",
        timeout_s=30,
    )
    first_verdict = json.loads((task_dir / "official_verdict.json").read_text())
    patch.write_text("diff --git a/b b/b\n", encoding="utf-8")
    second_code = _run_official_patch_evaluation(
        instance_path=instance,
        patch_path=patch,
        task_dir=task_dir,
        attempt_id="candidate-a",
        dataset_name="dataset",
        dataset_split="test",
        timeout_s=30,
    )
    second_verdict = json.loads((task_dir / "official_verdict.json").read_text())

    assert first_code == second_code == 0
    assert first_verdict["status"] == second_verdict["status"] == "resolved"
    assert calls[0][0] != calls[1][0]
    assert calls[0][1] != calls[1][1]


def test_swebench_trajectory_is_sanitized_and_summarized(tmp_path: Path) -> None:
    task_id = "org__issue-1"
    dump = tmp_path / "private-task-dump"
    trajectory = dump / "miniswe_run" / task_id / f"{task_id}.traj.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 7,
                                    "completion_tokens": 5,
                                    "total_tokens": 12,
                                }
                            },
                            "actions": [{"command": "sed -n 1p file.py"}],
                        },
                    },
                    {
                        "role": "tool",
                        "extra": {"raw_output": "line", "returncode": 0},
                    },
                ],
                "info": {
                    "exit_status": "Submitted",
                    "model_stats": {"instance_cost": 0.25, "api_calls": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "trace_id": "trace-1",
                "task_id": task_id,
                "passed": False,
                "status": "ok",
                "events": [
                    {
                        "kind": "artifact",
                        "input": {"name": "official_eval_stdout.txt"},
                    },
                    {"kind": "note", "input": "keep"},
                ],
                "artifact_paths": [
                    str(dump / "official_eval_stdout.txt"),
                    str(dump / "agent_stdout.txt"),
                ],
                "metrics": {
                    "task_dump": str(dump),
                    "task_dir": str(dump),
                    "patch_path": str(dump / "patch.diff"),
                    "source_project_path": str(tmp_path / "candidate"),
                    "repo": "org/repo",
                    "returncode": 0,
                    "evaluator_returncode": 1,
                    "exit_status": "Submitted",
                    "patch_bytes": 10,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = _augment_miniswe_trajectories(trace_path)
    row = json.loads(trace_path.read_text(encoding="utf-8"))

    assert row["status"] == "unresolved"
    assert summaries[task_id] == {
        "status": "unresolved",
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
        "monetary_cost": 0.25,
        "api_calls": 2,
    }
    assert not {"task_dump", "task_dir", "patch_path", "source_project_path"} & set(
        row["metrics"]
    )
    assert row["metrics"]["status_detail"] == "unresolved"
    assert row["metrics"]["total_tokens"] == 12
    assert not any(event.get("kind") == "artifact" for event in row["events"])
    assert {event["kind"] for event in row["events"]} >= {
        "action",
        "observation",
    }
    assert not any("official_eval" in path for path in row["artifact_paths"])


def test_swebench_finalize_evaluation_preserves_task_status_and_usage() -> None:
    evaluation = BenchmarkEvaluation(
        evaluation_id="eval-1",
        benchmark="swebench_verified",
        candidate_id="candidate-a",
        split="diagnostic",
        score=0.0,
        passrate=0.0,
        cost=0.0,
        outcomes=(
            TaskOutcome(
                task_id="org__issue-1",
                score=0.0,
                passed=False,
                trace_id="trace-1",
            ),
        ),
        trace_path="traces.jsonl",
        result_path="result.json",
    )
    finalized = _finalize_evaluation(
        evaluation,
        trajectory_stats={
            "org__issue-1": {
                "status": "agent_timeout",
                "total_tokens": 12,
                "monetary_cost": 0.25,
            }
        },
        cache_fingerprint="fingerprint",
        cache_payload={"limit": 1},
    )

    assert finalized.cost == 12.0
    assert finalized.outcomes[0].metadata["status"] == "agent_timeout"
    assert finalized.metadata["cache_fingerprint"] == "fingerprint"
    assert finalized.metadata["monetary_cost"] == 0.25
    assert finalized.metadata["task_status_counts"] == {"agent_timeout": 1}


def test_appworld_canary_and_audit_are_scenario_disjoint() -> None:
    tasks = [
        f"scenario_{scenario}_{variant}"
        for scenario in range(10)
        for variant in (1, 2, 3)
    ]
    canary, audit = _split_heldout_scenarios(
        tasks, seed=11, canary_limit=6, audit_limit=15
    )
    canary_scenarios = {task.rsplit("_", 1)[0] for task in canary}
    audit_scenarios = {task.rsplit("_", 1)[0] for task in audit}
    assert canary_scenarios
    assert audit_scenarios
    assert not canary_scenarios & audit_scenarios


def test_appworld_candidate_mounts_exclude_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "appworld"
    task = root / "data/tasks/scenario_1"
    (task / "dbs").mkdir(parents=True)
    (task / "specs.json").write_text("{}", encoding="utf-8")
    (task / "ground_truth").mkdir()
    site = tmp_path / "site-packages"
    site.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    agent = source / "agent.py"
    agent.write_text("def solve(world): return {}\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    out = tmp_path / "out"
    outputs = out / "candidate_appworld/experiments/outputs"
    outputs.mkdir(parents=True)
    monkeypatch.setattr(
        "traceunit.benchmarks.appworld.shutil.which", lambda _: "/usr/bin/docker"
    )

    command, _ = _sandboxed_appworld_command(
        argv=[
            "python",
            str(worker),
            "run",
            "--out",
            str(out.resolve()),
            "--agent-path",
            str(agent.resolve()),
        ],
        env={"PATH": "/usr/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        site_packages=site,
        worker=worker,
        source=source,
        task_out=out,
        sandbox_outputs=outputs,
        real_root=root,
        task_id="scenario_1",
    )
    joined = "\n".join(command)
    assert "ground_truth" not in joined
    assert f"src={task / 'specs.json'}" in joined
    assert f"src={task / 'dbs'}" in joined
    assert "--read-only" in command


def test_appworld_failed_candidate_is_not_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("def solve(world): return {}\n", encoding="utf-8")
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path / "appworld"))
    monkeypatch.setattr(
        "traceunit.benchmarks.appworld._sandboxed_appworld_command",
        lambda **_: (["candidate-run"], {}),
    )
    calls: list[list[str]] = []

    def failed_run(
        argv: list[str], *, env: dict[str, str], timeout: int, log_path: Path
    ) -> int:
        calls.append(argv)
        log_path.write_text("failed", encoding="utf-8")
        return 1

    monkeypatch.setattr("traceunit.benchmarks.appworld._run_process", failed_run)
    adapter = AppWorldAdapter(
        BenchmarkConfig(name="appworld", worldcalib_root=tmp_path / "worldcalib")
    )
    row = adapter._run_one(
        source=source,
        candidate_id="candidate",
        task_id="scenario_1",
        rep=0,
        out_dir=tmp_path / "evaluation",
        source_hash="a" * 64,
    )

    assert len(calls) == 1
    assert row["eval_returncode"] is None
    assert row["sealed"]["success"] is False
    assert row["sealed"]["error"].startswith("not evaluated:")


def test_config_loads_only_selected_key_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = "TRACEUNIT_SELECTED_API_KEY"
    unrelated = "TRACEUNIT_UNRELATED_SECRET"
    monkeypatch.delenv(selected, raising=False)
    monkeypatch.delenv(unrelated, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{selected}=selected-value\n{unrelated}=must-not-load\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "loop:\n"
        "  run_dir: run\n"
        "benchmark:\n"
        "  name: appworld\n"
        f"  worldcalib_root: {tmp_path}\n"
        f"  env_file: {env_file}\n"
        f"  api_key_env: {selected}\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.benchmark.env_file == env_file
    assert os.environ[selected] == "selected-value"
    assert unrelated not in os.environ
