from __future__ import annotations

import json
from pathlib import Path

from traceunit.agents.runner import AgentRunResult
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.pools import freeze_benchmark_plan, load_pool_items
from traceunit.config import (
    AgentConfig,
    AgentsConfig,
    AlignmentConfig,
    BenchmarkConfig,
    LoopConfig,
    ExperimentCondition,
    ProjectConfig,
    ProtocolConfig,
)
from traceunit.final_evaluation import FinalEvaluationRunner
from traceunit.io import read_json, sha256_tree, write_json
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
            calibration_shards=[["calibration-task"]],
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
                "source_trace_ids": ["trace"],
                "hypotheses": [
                    {
                        "hypothesis_id": "h1",
                        "mechanism": "behavior",
                        "target_boundary": "behavior file",
                        "claim": "candidate changes bad to good",
                        "evidence_trace_ids": ["trace"],
                        "alternatives": ["h2"],
                        "confidence": 1.0,
                    },
                    {
                        "hypothesis_id": "h2",
                        "mechanism": "missing file",
                        "target_boundary": "behavior file existence",
                        "claim": "the behavior file is absent",
                        "evidence_trace_ids": ["trace"],
                        "alternatives": ["h1"],
                        "confidence": 0.0,
                    },
                ],
                "target_hypothesis_id": "h1",
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
                        "admission_role": "positive_witness",
                    },
                ],
                "metadata": {},
            },
        )
        return _agent_result(role, log_dir)


class SearchAgent:
    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        if role == "search_planner":
            write_json(
                workspace / "composition_plan.json",
                {
                    "schema_version": 1,
                    "base_source_sha256": sha256_tree(workspace / "parent_source"),
                    "selections": [],
                    "integration_instructions": "change behavior from bad to good",
                },
            )
        elif role == "candidate_editor":
            (workspace / "source/behavior.txt").write_text("good", encoding="utf-8")
            plan = read_json(workspace / "composition_plan.json")
            write_json(
                workspace / "proposal.json",
                {
                    "candidate_id": "iter001_candidate",
                    "parent_id": "baseline",
                    "hypothesis_id": "h1",
                    "mechanism_claim": "change behavior from bad to good",
                    "predicted_effect": "unit and search score improve",
                    "regression_risks": [],
                    "plan_id": plan["attempt_fingerprint"],
                    "selected_archive_ids": [],
                },
            )
        else:
            raise AssertionError(role)
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
    return {
        "case_id": case_id,
        "family_id": "behavior.file.value",
        "tier": tier,
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
) -> ProjectConfig:
    return ProjectConfig(
        loop=LoopConfig(
            run_dir=tmp_path / "run",
            run_id="test",
            iterations=1,
        ),
        benchmark=BenchmarkConfig(name="appworld"),
        agents=AgentsConfig(
            test_author=AgentConfig(enabled=False),
            search=AgentConfig(enabled=False),
            regression_author=AgentConfig(enabled=False),
        ),
        protocol=ProtocolConfig(condition=condition),
        alignment=AlignmentConfig(),
    )


def test_search_promotes_without_opening_calibration_or_final(tmp_path: Path) -> None:
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
    assert not any("calibration" in call for call in benchmark.calls)


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
    assert not store.calibration_observations_path.exists()


def test_flat_local_improvement_becomes_content_addressed_archive(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": TestAuthor(), "search": SearchAgent()},
    ).run()
    assert summary["incumbent_id"] == "baseline"
    assert len(summary["archive_ids"]) == 1
    archive_id = summary["archive_ids"][0]
    manifest = config.loop.run_dir / "component_archive" / archive_id / "manifest.json"
    assert manifest.is_file()
    payload = read_json(manifest)
    assert payload["archive_id"] == archive_id
    assert payload["certificate"]["bridge_passed"] is True
    assert (manifest.parent / "component.patch").is_file()


def test_score_only_condition_has_no_unit_archive_or_calibration_artifacts(
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
        "delayed_alignment": False,
    }
    assert benchmark.calls == ["search", "search"]
    for name in (
        "test_library",
        "frozen_packets",
        "component_archive",
        "calibration",
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
    assert summary["archive_ids"] == []
    assert not (config.loop.run_dir / "component_archive").exists()
    assert not (config.loop.run_dir / "calibration").exists()
    candidate = config.loop.run_dir / "candidates/iter001_candidate"
    assert not (candidate / "archive_catalog.json").exists()
    assert not (candidate / "alignment_cards.json").exists()
