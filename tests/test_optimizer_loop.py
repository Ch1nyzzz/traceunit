from __future__ import annotations

import json
from pathlib import Path

from traceunit.agents.runner import AgentRunResult
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.config import (
    AgentConfig,
    AgentsConfig,
    BenchmarkConfig,
    DecisionConfig,
    LoopConfig,
    ProjectConfig,
)
from traceunit.io import write_json
from traceunit.models import BenchmarkEvaluation, TaskOutcome
from traceunit.optimizer import OptimizationLoop


class FakeBenchmark(BenchmarkAdapter):
    name = "fake"

    def __init__(self, root: Path, *, natural_gain: bool) -> None:
        self.root = root
        self.natural_gain = natural_gain
        self.seed = root / "fake-seed"
        self.seed.mkdir(parents=True)
        (self.seed / "behavior.txt").write_text("bad", encoding="utf-8")

    def prepare(self, work_dir: Path) -> None:
        pass

    def seed_source(self) -> Path:
        return self.seed

    def context(self) -> str:
        return "Fake benchmark with one public behavior file."

    def evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        split: str,
        out_dir: Path,
        limit_override: int | None = None,
    ) -> BenchmarkEvaluation:
        out_dir.mkdir(parents=True, exist_ok=True)
        good = (source / "behavior.txt").read_text().strip() == "good"
        score = 1.0 if good and self.natural_gain else 0.0
        task_id = f"{split}-task"
        artifact = out_dir / "trace.txt"
        artifact.write_text(f"behavior={'good' if good else 'bad'}", encoding="utf-8")
        trace_id = f"fake:{split}:{candidate_id}:{task_id}"
        trace_path = out_dir / "traces.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "benchmark": "fake",
                    "split": split,
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
            evaluation_id=f"fake:{split}:{candidate_id}",
            benchmark="fake",
            candidate_id=candidate_id,
            split=split,
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


class Experimentalist:
    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
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
                "hidden_variant_strategy": "same mechanism",
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


class Optimizer:
    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        (workspace / "source/behavior.txt").write_text("good", encoding="utf-8")
        write_json(
            workspace / "proposal.json",
            {
                "candidate_id": "iter001_candidate",
                "parent_id": "seed",
                "hypothesis_id": "h1",
                "mechanism_claim": "change behavior from bad to good",
                "predicted_effect": "proxy and natural score improve",
                "regression_risks": [],
            },
        )
        return _agent_result(role, log_dir)


def _case(case_id: str, tier: str, path: str, incumbent_pass: bool):
    return {
        "case_id": case_id,
        "family_id": "behavior",
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


def _config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        loop=LoopConfig(
            run_dir=tmp_path / "run",
            run_id="test",
            iterations=1,
        ),
        benchmark=BenchmarkConfig(name="appworld"),
        agents=AgentsConfig(
            experimentalist=AgentConfig(enabled=False),
            optimizer=AgentConfig(enabled=False),
            auditor=AgentConfig(enabled=False),
        ),
        decision=DecisionConfig(rejected_probe_rate=0.0),
    )


def test_complete_loop_promotes_heldout_improvement(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=True),
        agents={"experimentalist": Experimentalist(), "optimizer": Optimizer()},
    ).run()
    assert summary["incumbent_id"] == "iter001_candidate"
    assert summary["promoted_ids"] == ["seed", "iter001_candidate"]
    decision = json.loads(
        (config.loop.run_dir / "iterations/iter_001/decision.json").read_text()
    )
    assert decision["decision"] == "promote"
    assert decision["evidence"]["audit_delta"] is None
    assert summary["protocol"] == "posthoc_nonadaptive"
    assert summary["final_audit_delta"] == 1.0
    posthoc = json.loads(Path(summary["posthoc_audit_path"]).read_text())
    assert posthoc["records"][0]["audit_delta"] == 1.0
    calibration = json.loads(Path(summary["calibration_path"]).read_text())
    assert calibration["records"][0]["evidence"]["audit_delta"] == 1.0
    assert calibration["cross_level_metrics"]["n"] == 1
    assert "conditional_information_bits" in calibration["cross_level_metrics"]
    history = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path / "history", natural_gain=True),
        agents={"experimentalist": Experimentalist(), "optimizer": Optimizer()},
    )._public_history()
    assert "audit_delta" not in json.dumps(history)


def test_complete_loop_archives_masked_improvement(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"experimentalist": Experimentalist(), "optimizer": Optimizer()},
    ).run()
    assert summary["incumbent_id"] == "seed"
    assert summary["partial_archive_ids"] == ["iter001_candidate"]
    assert (
        config.loop.run_dir / "partial_archive/iter001_candidate/manifest.json"
    ).is_file()
