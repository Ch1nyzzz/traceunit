from __future__ import annotations

import json
from pathlib import Path

from traceunit.agents.runner import AgentRunResult
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.pools import freeze_benchmark_plan, load_pool_items
from traceunit.config import (
    AgentConfig,
    AgentsConfig,
    MemoryConfig,
    BenchmarkConfig,
    LoopConfig,
    ExperimentCondition,
    ProjectConfig,
    ProtocolConfig,
)
from traceunit.final_evaluation import FinalEvaluationRunner
from traceunit.io import read_json, write_json
from traceunit.models import (
    BenchmarkEvaluation,
    BenchmarkPlan,
    PoolSliceRef,
    TaskOutcome,
)
from traceunit.optimizer import OptimizationLoop
from traceunit.store import RunStore


class FakeBenchmark(BenchmarkAdapter):
    name = "fake"

    def __init__(self, root: Path, *, natural_gain: bool) -> None:
        self.root = root
        self.natural_gain = natural_gain
        self.seed = root / "fake-seed"
        self.seed.mkdir(parents=True)
        (self.seed / "behavior.txt").write_text("bad", encoding="utf-8")
        self.calls: list[str] = []
        self.plan: BenchmarkPlan | None = None

    def prepare(self, work_dir: Path) -> BenchmarkPlan:
        self.plan = freeze_benchmark_plan(
            root=work_dir / "benchmark_data" / "fake",
            benchmark=self.name,
            search_items=["search-task"],
            final_items=["final-task"],
            cluster_key=str,
        )
        return self.plan

    def baseline_source(self) -> Path:
        return self.seed

    def context(self) -> str:
        return "Fake benchmark with one public behavior file."

    def evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        pool: PoolSliceRef,
        out_dir: Path,
    ) -> BenchmarkEvaluation:
        out_dir.mkdir(parents=True, exist_ok=True)
        task_id = str(load_pool_items(pool)[0])
        self.calls.append(pool.slice_id)
        good = (source / "behavior.txt").read_text().strip() == "good"
        score = 1.0 if good and self.natural_gain else 0.0
        artifact = out_dir / "trace.txt"
        artifact.write_text(f"behavior={'good' if good else 'bad'}", encoding="utf-8")
        trace_id = f"fake:{pool.slice_id}:{candidate_id}:{task_id}"
        trace_path = out_dir / "traces.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "benchmark": "fake",
                    "split": pool.slice_id,
                    "candidate_id": candidate_id,
                    "task_id": task_id,
                    "score": score,
                    "passed": bool(score),
                    "status": "ok",
                    "input_summary": "make behavior good",
                    "output_summary": f"behavior={'good' if good else 'bad'}",
                    "events": [],
                    "artifact_paths": [str(artifact)],
                    "metrics": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = BenchmarkEvaluation(
            evaluation_id=f"fake:{pool.slice_id}:{candidate_id}",
            benchmark="fake",
            candidate_id=candidate_id,
            split=pool.slice_id,
            score=score,
            passrate=score,
            cost=1.0,
            outcomes=(
                TaskOutcome(
                    task_id=task_id,
                    score=score,
                    passed=bool(score),
                    trace_id=trace_id,
                ),
            ),
            trace_path=str(trace_path),
            result_path=str(out_dir / "raw.json"),
        )
        write_json(out_dir / "evaluation.json", result.to_dict())
        return result

    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        return (source / "behavior.txt").is_file(), ""

    def policy_violations(self, source: Path, diff_text: str) -> list[str]:
        return []


class TestAuthor:
    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        assert role == "test_author"
        if "reflection.json" in prompt:
            assert (workspace / "previous_outcome.json").is_file()
            write_json(
                workspace / "reflection.json",
                {
                    "assessment": "likely_test_gap",
                    "suspected_gap": "the bridge did not exercise adoption",
                    "recommendation": (
                        "Fake author lesson: vary the hidden sibling structurally."
                    ),
                    "alternative_explanation": "the edit may have overfit",
                    "confidence": "medium",
                },
            )
        output = workspace / "output"
        public = output / "tests/public/target.py"
        hidden = output / "tests/hidden/sibling.py"
        bridge = output / "tests/hidden/bridge.py"
        regression = output / "tests/hidden/regression.py"
        witness = output / "tests/hidden/positive_witness.py"
        for path in (public, hidden, bridge):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "import os, pathlib\n"
                "p = pathlib.Path(os.environ['TRACEUNIT_SOURCE']) / 'behavior.txt'\n"
                "raise SystemExit(0 if p.read_text().strip() == 'good' else 1)\n",
                encoding="utf-8",
            )
        regression.write_text(
            "import os, pathlib\n"
            "p = pathlib.Path(os.environ['TRACEUNIT_SOURCE']) / 'behavior.txt'\n"
            "raise SystemExit(0 if p.exists() else 1)\n",
            encoding="utf-8",
        )
        witness.write_text("raise SystemExit(0)\n", encoding="utf-8")
        write_json(
            output / "test_packet.json",
            {
                "packet_id": "behavior-packet",
                "version": 1,
                "hypotheses": [
                    {
                        "hypothesis_id": "h1",
                        "family": "verification",
                        "intervention_kind": "local_repair",
                        "mechanism": "behavior",
                        "target_boundary": "behavior file",
                        "claim": "candidate changes bad to good",
                        "evidence_trace_ids": ["trace"],
                        "alternatives": ["h2"],
                        "confidence": 1.0,
                    },
                    {
                        "hypothesis_id": "h2",
                        "family": "context",
                        "intervention_kind": "orchestration_change",
                        "mechanism": "missing file",
                        "target_boundary": "behavior file existence",
                        "claim": "the behavior file is absent",
                        "evidence_trace_ids": ["trace"],
                        "alternatives": ["h1"],
                        "confidence": 0.0,
                    },
                ],
                "target_hypothesis_id": "h1",
                "primary_family": "verification",
                "public_contract": "behavior must be good",
                "hidden_variant_strategy": "same mechanism with a structural variant",
                "cases": [
                    _case("public", "public", "tests/public/target.py", False),
                    _case("hidden", "hidden", "tests/hidden/sibling.py", False),
                    _case("bridge", "bridge", "tests/hidden/bridge.py", False),
                    _case(
                        "regression", "regression", "tests/hidden/regression.py", True
                    ),
                    {
                        **_case(
                            "positive_witness",
                            "admission",
                            "tests/hidden/positive_witness.py",
                            True,
                        ),
                    },
                ],
                "metadata": {},
            },
        )
        return _agent_result(role, log_dir)


class SearchAgent:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        assert role == "candidate_editor"
        self.prompts.append(prompt)
        (workspace / "source/behavior.txt").write_text("good", encoding="utf-8")
        write_json(
            workspace / "proposal.json",
            {
                "candidate_id": workspace.name,
                "parent_id": "baseline",
                "hypothesis_id": "h1",
                "intervention_kind": "local_repair",
                "mechanism_claim": "change behavior from bad to good",
                "predicted_effect": "unit and search score improve",
                "regression_risks": [],
            },
        )
        return _agent_result(role, log_dir)


class ScoreOnlyAgent:
    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        assert role == "score_only_editor"
        assert "no generated TestPacket" in prompt
        (workspace / "source/behavior.txt").write_text("good", encoding="utf-8")
        write_json(
            workspace / "score_only_proposal.json",
            {
                "candidate_id": "iter001_candidate",
                "parent_id": "baseline",
                "mechanism_claim": "change bad behavior to good",
                "predicted_effect": "search score improves",
                "regression_risks": [],
            },
        )
        return _agent_result(role, log_dir)


def _case(case_id: str, tier: str, path: str, incumbent_pass: bool):
    roles = {
        "public": "target_reproducer",
        "hidden": "structural_sibling",
        "bridge": "downstream_bridge",
        "regression": "off_target_control",
        "admission": "positive_witness",
    }
    return {
        "case_id": case_id,
        "tier": tier,
        "evidence_role": roles[tier],
        "path": path,
        "driver": "python",
        "expected_incumbent_pass": incumbent_pass,
        "expected_candidate_pass": True,
    }


def _agent_result(role: str, log_dir: Path) -> AgentRunResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stdout.txt", "stderr.txt", "final.txt"):
        (log_dir / name).write_text("", encoding="utf-8")
    return AgentRunResult(
        role=role,
        returncode=0,
        duration_s=0.0,
        stdout_path=str(log_dir / "stdout.txt"),
        stderr_path=str(log_dir / "stderr.txt"),
        final_message_path=str(log_dir / "final.txt"),
    )


def _config(
    tmp_path: Path,
    condition: ExperimentCondition = ExperimentCondition.ARCHIVE,
    iterations: int = 1,
) -> ProjectConfig:
    return ProjectConfig(
        loop=LoopConfig(
            run_dir=tmp_path / "run",
            run_id="test",
            iterations=iterations,
        ),
        benchmark=BenchmarkConfig(name="appworld"),
        agents=AgentsConfig(
            test_author=AgentConfig(enabled=False),
            search=AgentConfig(enabled=False),
            regression_author=AgentConfig(enabled=False),
        ),
        protocol=ProtocolConfig(condition=condition),
        memory=MemoryConfig(),
    )


def test_search_promotes_without_opening_final_pool(tmp_path: Path) -> None:
    config = _config(tmp_path)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    summary = OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()
    assert summary["incumbent_id"] == "iter001_candidate"
    assert summary["promoted_ids"] == ["baseline", "iter001_candidate"]
    assert summary["final_evaluation"] == "not_opened"
    assert benchmark.calls == ["search", "search"]
    assert not (config.loop.run_dir / "sealed/final").exists()


def test_final_evaluation_is_a_separate_sealed_operation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()
    store = RunStore(config.loop.run_dir)
    state = store.load_state()
    assert state is not None
    assert benchmark.plan is not None
    runner = FinalEvaluationRunner(
        store=store,
        benchmark=benchmark,
        benchmark_plan=benchmark.plan,
    )
    report = runner.run(runner.seal(state))
    assert report["paired_delta"] == 1.0
    assert benchmark.calls[-2:] == ["final", "final"]
    assert not store.memory_root.exists()


def test_flat_local_improvement_is_retained_as_latent_packet(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()
    assert summary["incumbent_id"] == "baseline"
    assert len(summary["latent_packets"]) == 1
    ref = summary["latent_packets"][0]
    store = RunStore(config.loop.run_dir)
    bundle = store.packet_store_root / ref["path"]
    assert (bundle / "test_packet.json").is_file()
    patch = store.latent_root / ref["content_sha256"] / "component.patch"
    assert patch.is_file()
    assert "behavior.txt" in patch.read_text(encoding="utf-8")


class LatentFlipBenchmark(FakeBenchmark):
    """Search transfer appears only at the second opportunity."""

    def evaluate(self, *, source: Path, candidate_id: str, pool, out_dir: Path):
        self.natural_gain = candidate_id.startswith("iter002")
        return super().evaluate(
            source=source, candidate_id=candidate_id, pool=pool, out_dir=out_dir
        )


def test_promoted_candidate_realizes_latent_packet_into_preservation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, iterations=2)
    benchmark = LatentFlipBenchmark(tmp_path, natural_gain=False)
    editor = SearchAgent()
    summary = OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": TestAuthor(), "search": editor},
    ).run()

    # Iteration 1 archived a latent capability; iteration 2 realized it.
    events = (config.loop.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "latent_packet_retained" in events
    assert "latent_packet_realized" in events
    assert summary["incumbent_id"] == "iter002_candidate"
    assert summary["latent_packets"] == []
    preserved = {ref["content_sha256"] for ref in summary["preserved_packets"]}
    evidence = read_json(config.loop.run_dir / "iterations/iter_002/evidence.json")
    assert len(evidence["realized_latent"]) == 1
    assert set(evidence["realized_latent"]) <= preserved
    assert evidence["attribution_scope"] == "composition"
    assert evidence["component_families"] == ["verification"]
    # The editor was shown the latent capability and its reference patch.
    assert any("latent_capabilities.json" in prompt for prompt in editor.prompts[1:])
    workspace_patches = list(
        (config.loop.run_dir / "candidates/iter002_candidate").glob(
            "latent_capabilities/*/component.patch"
        )
    )
    assert workspace_patches


def test_score_only_condition_has_no_unit_archive_or_memory_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.SCORE_ONLY)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    summary = OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"search": ScoreOnlyAgent()},
    ).run()

    assert summary["protocol"] == "c0_score_only"
    assert summary["incumbent_id"] == "iter001_candidate"
    assert summary["capabilities"] == {
        "generated_packets": False,
        "unit_gate": False,
        "partial_archive": False,
        "online_ut_memory": False,
    }
    assert benchmark.calls == ["search", "search"]
    for name in (
        "test_library",
        "frozen_packets",
        "ut_memory",
    ):
        assert not (config.loop.run_dir / name).exists()
    evidence = read_json(config.loop.run_dir / "iterations/iter_001/evidence.json")
    assert set(evidence) == {
        "iteration",
        "candidate_id",
        "parent_id",
        "search_delta",
        "total_cost",
        "metadata",
    }


def test_full_condition_reflects_each_completed_search_comparison(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=True),
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()

    store = RunStore(config.loop.run_dir)
    assert summary["ut_memory_version"] == 1
    assert summary["ut_feedback_episodes"] == 1
    assert store.ut_feedback_episodes_path.is_file()
    assert store.ut_world_model_path.is_file()
    content = store.ut_world_model_path.read_text(encoding="utf-8")
    assert content.startswith("# TraceUnit UT-design world model\n")
    assert content.count("\n") >= 4
    assert "\\n" not in content
    assert not (config.loop.run_dir / "calibration").exists()


def test_author_self_reflection_feeds_memory(tmp_path: Path) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL, iterations=2)
    summary = OptimizationLoop(
        config,
        # iter1 archives (flat search), iter2 promotes: both iterations author a
        # fresh packet, so iter2's author consumes iter1's staged digest.
        benchmark=LatentFlipBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()

    store = RunStore(config.loop.run_dir)
    episodes = {
        item["iteration"]: item
        for item in (
            json.loads(line)
            for line in store.ut_feedback_episodes_path.read_text(
                encoding="utf-8"
            ).splitlines()
        )
    }
    assert summary["ut_feedback_episodes"] == 2
    # Iteration 1's lesson was written by the next iteration's Test Author.
    assert episodes[1]["recommendation"].startswith("Fake author lesson")
    assert episodes[1]["confidence"] == "medium"
    # The final iteration has no later author run; the run-end fallback covers it.
    assert not episodes[2]["recommendation"].startswith("Fake author lesson")
    assert "Fake author lesson" in store.ut_world_model_path.read_text(
        encoding="utf-8"
    )
    assert not (store.memory_root / "pending_reflection.json").exists()


def test_commit_pending_tolerates_malformed_reflection(tmp_path: Path) -> None:
    from traceunit.ut_memory import UTMemoryLedger, UTMemoryManager

    root = tmp_path / "memory"
    root.mkdir()
    manager = UTMemoryManager(
        root=root,
        ledger=UTMemoryLedger(root / "episodes.jsonl"),
        max_lessons=8,
        world_model_path=root / "world_model.md",
    )
    write_json(
        root / "pending_reflection.json",
        {
            "candidate_id": "iter001_candidate",
            "iteration": 1,
            "primary_family": "verification",
            "intervention_kind": "local_repair",
            "attribution_scope": "atomic",
            "component_families": [],
            "local_contract_passed": True,
            "bridge_contract_passed": True,
            "search_outcome": "improved",
            "committed_decision": "promote",
        },
    )
    episode = manager.commit_pending(
        {
            "assessment": "made_up_value",
            "recommendation": "still recorded",
            "confidence": "certain",
        }
    )
    assert episode is not None
    assert episode.assessment.value == "insufficient_evidence"
    assert episode.confidence == "low"
    assert episode.recommendation == "still recorded"
    assert not (root / "pending_reflection.json").exists()
    assert manager.commit_pending(None) is None


def test_resume_reuses_frozen_decision_without_re_evaluation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = FakeBenchmark(tmp_path, natural_gain=True)
    OptimizationLoop(
        config,
        benchmark=first,
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()

    store = RunStore(config.loop.run_dir)
    state = store.load_state()
    assert state is not None
    state.next_iteration = 1
    state.status = "running"
    store.save_state(state)

    first.calls.clear()
    summary = OptimizationLoop(
        config,
        benchmark=first,
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()

    assert summary["incumbent_id"] == "iter001_candidate"
    assert first.calls == []
    assert store.load_state() is not None
    assert store.load_state().next_iteration == 2


def test_raw_traceunit_records_partial_eligibility_without_persisting_component(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.RAW_TRACEUNIT)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()

    assert summary["protocol"] == "c1_raw_traceunit"
    assert summary["partial_eligible_ids"] == ["iter001_candidate"]
    assert summary["latent_packets"] == []
    assert not RunStore(config.loop.run_dir).latent_root.exists()
    assert not (config.loop.run_dir / "ut_memory").exists()
    candidate = config.loop.run_dir / "candidates/iter001_candidate"
    assert not (candidate / "latent_capabilities.json").exists()
    assert not (candidate / "ut_design_world_model.md").exists()
