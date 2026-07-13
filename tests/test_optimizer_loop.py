from __future__ import annotations

import json
from pathlib import Path

from traceunit.agents.runner import AgentRunResult
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.pools import freeze_benchmark_plan, load_pool_items
from traceunit.config import (
    AgentConfig,
    AgentsConfig,
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
    """One or more synthetic tasks scored by a per-task rule on behavior.txt.

    ``scorers`` maps task_id -> callable(text) -> float. The default single
    task rewards behavior exactly 'good' when ``natural_gain`` is set, so the
    stock editor (writes 'good') improves paired search.
    """

    name = "fake"

    def __init__(
        self,
        root: Path,
        *,
        natural_gain: bool,
        search_items: list[str] | None = None,
        scorers: dict[str, object] | None = None,
    ) -> None:
        self.root = root
        self.natural_gain = natural_gain
        self.seed = root / "fake-seed"
        self.seed.mkdir(parents=True)
        (self.seed / "behavior.txt").write_text("bad", encoding="utf-8")
        self.calls: list[str] = []
        self.plan: BenchmarkPlan | None = None
        self.search_items = search_items or ["search-task"]
        self.scorers = scorers or {}

    def prepare(self, work_dir: Path) -> BenchmarkPlan:
        self.plan = freeze_benchmark_plan(
            root=work_dir / "benchmark_data" / "fake",
            benchmark=self.name,
            search_items=self.search_items,
            final_items=["final-task"],
            cluster_key=str,
        )
        return self.plan

    def baseline_source(self) -> Path:
        return self.seed

    def context(self) -> str:
        return "Fake benchmark with one public behavior file."

    def _score(self, task_id: str, text: str) -> float:
        scorer = self.scorers.get(task_id)
        if scorer is not None:
            return float(scorer(text))
        return 1.0 if text == "good" and self.natural_gain else 0.0

    def evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        pool: PoolSliceRef,
        out_dir: Path,
    ) -> BenchmarkEvaluation:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.calls.append(pool.slice_id)
        text = (source / "behavior.txt").read_text().strip()
        task_ids = [str(item) for item in load_pool_items(pool)]
        outcomes = []
        trace_rows = []
        for task_id in task_ids:
            score = self._score(task_id, text)
            artifact = out_dir / f"trace_{task_id}.txt"
            artifact.write_text(f"behavior={text}", encoding="utf-8")
            trace_id = f"fake:{pool.slice_id}:{candidate_id}:{task_id}"
            trace_rows.append(
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
                    "output_summary": f"behavior={text}",
                    "events": [],
                    "artifact_paths": [str(artifact)],
                    "metrics": {},
                }
            )
            outcomes.append(
                TaskOutcome(
                    task_id=task_id,
                    score=score,
                    passed=bool(score),
                    trace_id=trace_id,
                )
            )
        trace_path = out_dir / "traces.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in trace_rows),
            encoding="utf-8",
        )
        total = sum(item.score for item in outcomes) / len(outcomes)
        result = BenchmarkEvaluation(
            evaluation_id=f"fake:{pool.slice_id}:{candidate_id}",
            benchmark="fake",
            candidate_id=candidate_id,
            split=pool.slice_id,
            score=total,
            passrate=total,
            cost=1.0,
            outcomes=tuple(outcomes),
            trace_path=str(trace_path),
            result_path=str(out_dir / "raw.json"),
        )
        write_json(out_dir / "evaluation.json", result.to_dict())
        return result

    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        return (source / "behavior.txt").is_file(), ""

    def policy_violations(self, source: Path, diff_text: str) -> list[str]:
        return []


def _write_instance_bundle(
    output: Path,
    *,
    instance_id: str,
    family: str,
    condition: str,
) -> None:
    """An unfrozen battery-instance bundle whose probe runs ``condition`` as a
    python expression over the source's behavior.txt content bound to `text`."""

    bundle = output / "instances" / instance_id
    test_path = bundle / "tests/public/probe.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        "import os, pathlib\n"
        "text = (pathlib.Path(os.environ['TRACEUNIT_SOURCE'])"
        " / 'behavior.txt').read_text()\n"
        f"raise SystemExit(0 if ({condition}) else 1)\n",
        encoding="utf-8",
    )
    write_json(
        bundle / "test_packet.json",
        {
            "packet_id": instance_id,
            "version": 1,
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "family": family,
                    "intervention_kind": "local_repair",
                    "mechanism": "capability probe",
                    "target_boundary": "one decision boundary",
                    "claim": "the capability is deficient",
                    "evidence_trace_ids": ["trace"],
                }
            ],
            "target_hypothesis_id": "h1",
            "primary_family": family,
            "public_contract": "capability check",
            "hidden_variant_strategy": "cross-domain siblings",
            "cases": [
                {
                    "case_id": "probe",
                    "tier": "public",
                    "evidence_role": "target_reproducer",
                    "path": "tests/public/probe.py",
                    "driver": "python",
                    "expected_incumbent_pass": False,
                    "expected_candidate_pass": True,
                }
            ],
            "metadata": {"packet_kind": "battery_instance"},
        },
    )


class BatteryAuthorAgent:
    """Cold start builds a target group (two variants) plus a guard group;
    warm iterations distill into the world model and leave the battery as is.
    """

    def __init__(self, target_text: str = "good") -> None:
        self.target_text = target_text

    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        assert role == "test_author"
        world_model = workspace / "ut_design_world_model.md"
        if world_model.is_file() and "distill" in prompt:
            if (workspace / "last_iteration.json").is_file():
                info = read_json(workspace / "last_iteration.json")
                world_model.write_text(
                    world_model.read_text(encoding="utf-8")
                    + f"\n## iter_{int(info['iteration']):03d} distill\n"
                    f"- Fake author lesson: decision was {info['decision']} with "
                    f"delta {info['search_delta']}.\n",
                    encoding="utf-8",
                )
        output = workspace / "output"
        state = read_json(workspace / "battery_state.json")
        cold_start = not state["capabilities"]
        update: dict = {
            "target_capability": "behavior-repair",
            "target_family": "verification",
            "capability_descriptions": {},
            "new_instances": [],
            "retire_instance_ids": [],
        }
        if cold_start:
            update["capability_descriptions"] = {
                "behavior-repair": "the policy produces the repaired behavior",
                "stability-guard": "unrelated behavior stays intact",
            }
            for suffix in ("a", "b"):
                instance_id = f"behavior-repair-{suffix}"
                _write_instance_bundle(
                    output,
                    instance_id=instance_id,
                    family="verification",
                    condition=f"'{self.target_text}' in text",
                )
                update["new_instances"].append(
                    {
                        "instance_id": instance_id,
                        "capability": "behavior-repair",
                        "family": "verification",
                        "description": f"variant {suffix}: behavior contains "
                        f"'{self.target_text}'",
                        "expected_incumbent_pass": False,
                        "bundle": f"instances/{instance_id}",
                    }
                )
            _write_instance_bundle(
                output,
                instance_id="stability-guard-a",
                family="state",
                condition="'toxic' not in text",
            )
            update["new_instances"].append(
                {
                    "instance_id": "stability-guard-a",
                    "capability": "stability-guard",
                    "family": "state",
                    "description": "behavior must not become toxic",
                    "expected_incumbent_pass": True,
                    "bundle": "instances/stability-guard-a",
                }
            )
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "battery_update.json", update)
        return _agent_result(role, log_dir)


class MisdeclaringAuthor(BatteryAuthorAgent):
    """Declares a wrong incumbent expectation once, then reads the staged
    admission transcript and its own previous output to fix it."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_retry_evidence = False

    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        first_try = not (workspace / "previous_output").is_dir()
        result = super().run(
            role=role, prompt=prompt, workspace=workspace, log_dir=log_dir
        )
        update_path = workspace / "output/battery_update.json"
        update = read_json(update_path)
        if first_try:
            # baseline behavior is 'bad', so declaring pass=True is wrong.
            for item in update["new_instances"]:
                if item["instance_id"] == "behavior-repair-a":
                    item["expected_incumbent_pass"] = True
            write_json(update_path, update)
        else:
            self.saw_retry_evidence = (
                "previous_output" in prompt
                and "previous_admission" in prompt
                and (workspace / "previous_output/battery_update.rejected_1.json").is_file()
                and (
                    workspace / "previous_admission/behavior-repair-a/results.json"
                ).is_file()
            )
        return result


def test_rejected_author_sees_its_own_output_and_admission_transcripts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL)
    author = MisdeclaringAuthor()
    OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=True),
        agents={"test_author": author, "search": SearchAgent()},
    ).run()
    assert author.saw_retry_evidence
    decision = read_json(config.loop.run_dir / "iterations/iter_001/decision.json")
    assert decision["decision"] == "promote"


class SearchAgent:
    def __init__(self, text: str = "good") -> None:
        self.text = text
        self.prompts: list[str] = []

    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        assert role == "candidate_editor"
        self.prompts.append(prompt)
        (workspace / "source/behavior.txt").write_text(self.text, encoding="utf-8")
        write_json(
            workspace / "proposal.json",
            {
                "candidate_id": workspace.name,
                "parent_id": "baseline",
                "mechanism_claim": "change behavior from bad to good",
                "predicted_effect": "battery and search improve",
                "regression_risks": [],
            },
        )
        return _agent_result(role, log_dir)


class BadEditor(SearchAgent):
    """An editor whose change never satisfies the battery."""

    def __init__(self) -> None:
        super().__init__(text="still-bad")


class ToxicEditor(SearchAgent):
    """Repairs the target behavior while damaging the guard capability."""

    def __init__(self) -> None:
        super().__init__(text="good but toxic")


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
        ),
        protocol=ProtocolConfig(condition=condition),
    )


def test_search_promotes_and_updates_battery_reference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    summary = OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()
    assert summary["incumbent_id"] == "iter001_candidate"
    assert summary["promoted_ids"] == ["baseline", "iter001_candidate"]
    assert summary["final_evaluation"] == "not_opened"
    assert benchmark.calls == ["search", "search"]
    assert not (config.loop.run_dir / "sealed/final").exists()
    # Cold start built the battery; the promotion refreshed the reference.
    reference = read_json(config.loop.run_dir / "battery/incumbent_results.json")
    assert reference == {
        "behavior-repair-a": True,
        "behavior-repair-b": True,
        "stability-guard-a": True,
    }
    groups = {
        item["capability"]: item for item in summary["battery"]["capabilities"]
    }
    assert set(groups) == {"behavior-repair", "stability-guard"}
    assert summary["calibration_rows"] == 1


def test_final_evaluation_is_a_separate_sealed_operation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
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


def test_unit_pass_with_flat_search_is_archived_as_record(tmp_path: Path) -> None:
    """Cell 2: battery improved, paired search flat -> archive a plain record."""

    config = _config(tmp_path)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()
    assert summary["incumbent_id"] == "baseline"
    assert summary["archived_ids"] == ["iter001_candidate"]
    store = RunStore(config.loop.run_dir)
    archive_dir = store.archive_root / "iter001_candidate"
    record = read_json(archive_dir / "record.json")
    assert record["kind"] == "unit_passed_search_flat"
    assert record["target_improved"] is True
    assert record["search_delta"] == 0.0
    assert "behavior.txt" in (archive_dir / "candidate.diff").read_text()
    assert not (config.loop.run_dir / "mismatch").exists()
    decision = read_json(config.loop.run_dir / "iterations/iter_001/decision.json")
    assert decision["decision"] == "archive"
    # The reference is untouched: archives never move the incumbent.
    reference = read_json(config.loop.run_dir / "battery/incumbent_results.json")
    assert reference["behavior-repair-a"] is False


def test_confirmed_search_improvement_with_failed_battery_promotes(
    tmp_path: Path,
) -> None:
    """Cell 4, confirmed: the search improvement survives an independent
    re-evaluation, so the candidate promotes and the battery miss is staged
    as a mismatch for the next Test Author."""

    config = _config(tmp_path, ExperimentCondition.FULL)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    OptimizationLoop(
        config,
        # score rewards 'good', but the battery demands 'great': the edit
        # improves paired search yet fails its target group.
        benchmark=benchmark,
        agents={
            "test_author": BatteryAuthorAgent(target_text="great"),
            "search": SearchAgent(),
        },
    ).run()

    run = config.loop.run_dir
    decision = read_json(run / "iterations/iter_001/decision.json")
    assert decision["decision"] == "promote"
    assert "confirmation" in decision["reason"]
    search = decision["evidence"]["metadata"]["search"]
    assert search["confirmation"]["search_delta"] > 0
    # baseline + candidate + confirmation = three search runs.
    assert benchmark.calls.count("search") == 3
    mismatch = read_json(run / "mismatch/iter_001/mismatch.json")
    assert mismatch["kind"] == "search_improved_unit_failed"
    assert mismatch["decision"] == "promote"
    assert mismatch["target_capability"] == "behavior-repair"
    assert any(flip["flipped"] for flip in mismatch["task_flips"])
    # The diagnosing author gets the execution transcripts, not just booleans.
    transcripts = run / "mismatch/iter_001/probe_transcripts"
    assert (transcripts / "behavior-repair-a/results.json").is_file()
    events = (run / "events.jsonl").read_text(encoding="utf-8")
    assert '"search_confirmation_run"' in events
    state = json.loads((run / "run_state.json").read_text())
    assert state["incumbent_id"] == "iter001_candidate"
    assert state["archived_ids"] == []
    # The promoted candidate's own battery results (target still failing)
    # become the reference: the group keeps something for the author to fix.
    reference = read_json(run / "battery/incumbent_results.json")
    assert reference["behavior-repair-a"] is False


class LuckyOnceBenchmark(FakeBenchmark):
    """The candidate's first search run improves; its confirmation does not."""

    def evaluate(self, *, source: Path, candidate_id: str, pool, out_dir: Path):
        self.natural_gain = not candidate_id.endswith("__confirm")
        return super().evaluate(
            source=source, candidate_id=candidate_id, pool=pool, out_dir=out_dir
        )


def test_unconfirmed_search_improvement_is_archived_as_noise(
    tmp_path: Path,
) -> None:
    """Cell 4, unconfirmed: the improvement vanishes on re-evaluation, so the
    candidate stays a record and no mismatch wastes the author's attention."""

    config = _config(tmp_path, ExperimentCondition.FULL)
    OptimizationLoop(
        config,
        benchmark=LuckyOnceBenchmark(tmp_path, natural_gain=True),
        agents={
            "test_author": BatteryAuthorAgent(target_text="great"),
            "search": SearchAgent(),
        },
    ).run()

    run = config.loop.run_dir
    decision = read_json(run / "iterations/iter_001/decision.json")
    assert decision["decision"] == "archive"
    assert "did not survive the confirmation" in decision["reason"]
    record = read_json(run / "archive/iter001_candidate/record.json")
    assert record["kind"] == "search_improved_unit_failed"
    assert record["search_delta"] > 0
    assert not (run / "mismatch").exists()
    state = json.loads((run / "run_state.json").read_text())
    assert state["incumbent_id"] == "baseline"
    assert state["archived_ids"] == ["iter001_candidate"]


def test_unit_pass_with_search_regression_is_rejected_as_mismatch(
    tmp_path: Path,
) -> None:
    """Cell 3: battery improved, paired search regressed -> reject + mismatch."""

    config = _config(tmp_path, ExperimentCondition.FULL)
    benchmark = FakeBenchmark(
        tmp_path,
        natural_gain=False,
        search_items=["hard-task", "fragile-task"],
        scorers={
            # hard-task always fails, so the baseline has a failure trace.
            "hard-task": lambda text: 0.0,
            # fragile-task passes on the baseline and breaks on the edit.
            "fragile-task": lambda text: 1.0 if text == "bad" else 0.0,
        },
    )
    OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()

    run = config.loop.run_dir
    decision = read_json(run / "iterations/iter_001/decision.json")
    assert decision["decision"] == "reject"
    assert "deviates from the search distribution" in decision["reason"]
    assert decision["evidence"]["target_improved"] is True
    assert decision["evidence"]["search_delta"] < 0
    mismatch = read_json(run / "mismatch/iter_001/mismatch.json")
    assert mismatch["kind"] == "unit_passed_search_regressed"
    flips = {flip["task_id"]: flip for flip in mismatch["task_flips"]}
    assert flips["fragile-task"]["flipped"] is True
    assert flips["hard-task"]["flipped"] is False
    state = json.loads((run / "run_state.json").read_text())
    assert state["incumbent_id"] == "baseline"
    assert state["archived_ids"] == []


def test_both_failed_is_a_plain_reject(tmp_path: Path) -> None:
    """Cell 5: battery not improved and search flat -> plain reject."""

    config = _config(tmp_path, ExperimentCondition.FULL)
    OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": BatteryAuthorAgent(), "search": BadEditor()},
    ).run()

    run = config.loop.run_dir
    decision = read_json(run / "iterations/iter_001/decision.json")
    assert decision["decision"] == "reject"
    assert "neither the target capability nor paired search" in decision["reason"]
    assert not (run / "mismatch").exists()
    assert not (run / "archive").exists()


def test_collateral_damage_fails_the_battery_verdict(tmp_path: Path) -> None:
    """An edit that repairs the target while breaking another capability group
    is a unit failure even though the target group improved."""

    config = _config(tmp_path, ExperimentCondition.FULL)
    OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": BatteryAuthorAgent(), "search": ToxicEditor()},
    ).run()

    decision = read_json(
        config.loop.run_dir / "iterations/iter_001/decision.json"
    )
    assert decision["decision"] == "reject"
    assert "damaged other capabilities" in decision["reason"]
    assert decision["evidence"]["target_improved"] is True
    assert decision["evidence"]["collateral_ok"] is False
    assert decision["evidence"]["collateral_delta"] == -1.0


def test_later_editor_sees_archive_records(tmp_path: Path) -> None:
    config = _config(tmp_path, iterations=2)
    editor = SearchAgent()
    OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": BatteryAuthorAgent(), "search": editor},
    ).run()

    # Iteration 1 archived; iteration 2's editor received the staged record.
    assert "archives.json" in editor.prompts[1]
    staged = (
        config.loop.run_dir
        / "candidates/iter002_candidate/archive/iter001_candidate/candidate.diff"
    )
    assert staged.is_file()
    archives = read_json(
        config.loop.run_dir / "candidates/iter002_candidate/archives.json"
    )
    assert archives["archives"][0]["candidate_id"] == "iter001_candidate"
    assert archives["archives"][0]["kind"] == "unit_passed_search_flat"


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
    for name in ("battery", "ut_memory"):
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


def test_full_condition_records_last_iteration_for_the_next_author(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=True),
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()

    store = RunStore(config.loop.run_dir)
    assert store.ut_world_model_path.is_file()
    content = store.ut_world_model_path.read_text(encoding="utf-8")
    assert content.startswith("# UT design world model\n")
    assert summary["world_model_distills"] == 0
    digest = read_json(store.memory_root / "last_iteration.json")
    assert digest["decision"] == "promote"
    assert digest["target_capability"] == "behavior-repair"
    assert digest["search_delta"] > 0
    assert digest["task_flips"] and digest["task_flips"][0]["flipped"] is True
    assert digest["mismatch"] is False
    assert digest["battery_deltas"]["behavior-repair"]["delta"] == 1.0


class FlipBenchmark(FakeBenchmark):
    """Search transfer appears only at the second opportunity."""

    def evaluate(self, *, source: Path, candidate_id: str, pool, out_dir: Path):
        self.natural_gain = candidate_id.startswith("iter002")
        return super().evaluate(
            source=source, candidate_id=candidate_id, pool=pool, out_dir=out_dir
        )


def test_author_distill_is_committed_back_to_the_world_model(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL, iterations=2)
    summary = OptimizationLoop(
        config,
        # iter1 archives (flat search), iter2 promotes: iter2's author reads
        # iter1's digest and appends its distill to the world model.
        benchmark=FlipBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()

    store = RunStore(config.loop.run_dir)
    content = store.ut_world_model_path.read_text(encoding="utf-8")
    assert "## iter_001 distill" in content
    assert "Fake author lesson: decision was archive" in content
    assert summary["world_model_distills"] == 1
    events = (config.loop.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"world_model_updated"' in events
    # Both candidates earned calibration rows.
    assert summary["calibration_rows"] == 2
    # The second author received the frozen probe bundles to audit.
    author_workspace = (
        config.loop.run_dir / "iterations/iter_002/test_author/attempt_1/workspace"
    )
    assert (
        author_workspace / "battery_instances/behavior-repair-a/test_packet.json"
    ).is_file()
    # ... and the archived candidate's record and diff.
    author_archives = read_json(author_workspace / "archives.json")
    assert author_archives["archives"][0]["candidate_id"] == "iter001_candidate"
    assert (author_workspace / "archive/iter001_candidate/candidate.diff").is_file()
    # The second editor sees every prior candidate's reason, mechanism, diff.
    history = read_json(
        config.loop.run_dir / "candidates/iter002_candidate/history.json"
    )
    first = history["decisions"][0]
    assert first["decision"] == "archive"
    assert first["reason"]
    assert first["mechanism_claim"]
    assert Path(first["diff_path"]).is_file()


class SilentAuthor(BatteryAuthorAgent):
    """Authors valid battery updates but never writes its distill."""

    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        world_model = workspace / "ut_design_world_model.md"
        result = super().run(
            role=role, prompt=prompt, workspace=workspace, log_dir=log_dir
        )
        if world_model.is_file():
            text = world_model.read_text(encoding="utf-8")
            head = text.split("\n## iter_")[0]
            world_model.write_text(head, encoding="utf-8")
        return result


def test_skipped_distill_is_recorded_not_papered_over(tmp_path: Path) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL, iterations=2)
    summary = OptimizationLoop(
        config,
        benchmark=FlipBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": SilentAuthor(), "search": SearchAgent()},
    ).run()

    assert summary["world_model_distills"] == 0
    events = (config.loop.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"world_model_not_updated"' in events
    content = RunStore(config.loop.run_dir).ut_world_model_path.read_text(
        encoding="utf-8"
    )
    assert "## iter_0" not in content


def test_resume_reuses_frozen_decision_without_re_evaluation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = FakeBenchmark(tmp_path, natural_gain=True)
    OptimizationLoop(
        config,
        benchmark=first,
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
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
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()

    assert summary["incumbent_id"] == "iter001_candidate"
    assert first.calls == []
    assert store.load_state() is not None
    assert store.load_state().next_iteration == 2


def test_raw_traceunit_records_archive_ids_without_persisting_records(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.RAW_TRACEUNIT)
    summary = OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=False),
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()

    assert summary["protocol"] == "c1_raw_traceunit"
    assert summary["archived_ids"] == ["iter001_candidate"]
    assert summary["archive_refs"] == []
    assert not (config.loop.run_dir / "archive").exists()
    assert not (config.loop.run_dir / "ut_memory").exists()
    candidate = config.loop.run_dir / "candidates/iter001_candidate"
    assert not (candidate / "archives.json").exists()
    assert not (candidate / "ut_design_world_model.md").exists()
    # The battery itself exists in every packet-generating condition.
    assert (config.loop.run_dir / "battery/manifest.json").is_file()


class LearningEditor:
    """Fails the battery on attempt 1, then fixes it after reading feedback."""

    def __init__(self) -> None:
        self.attempts = 0
        self.saw_feedback = False

    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        assert role == "candidate_editor"
        self.attempts += 1
        if self.attempts == 1:
            (workspace / "source/behavior.txt").write_text(
                "still-bad", encoding="utf-8"
            )
        else:
            assert "continuing your own previous attempt" in prompt
            feedback = read_json(workspace / "unit_feedback.json")
            self.saw_feedback = bool(feedback["target_instances"])
            assert feedback["target_improved"] is False
            assert feedback["target_capability"] == "behavior-repair"
            failing = [
                item
                for item in feedback["target_instances"]
                if not item["candidate_passed"]
            ]
            # The editor sees opaque codes, never probe ids or descriptions.
            assert failing and failing[0]["instance"].startswith("instance_")
            assert "description" not in failing[0]
            assert "instance_id" not in failing[0]
            assert feedback["damaged_capabilities"] == {}
            (workspace / "source/behavior.txt").write_text("good", encoding="utf-8")
        write_json(
            workspace / "proposal.json",
            {
                "candidate_id": workspace.name,
                "parent_id": "baseline",
                "mechanism_claim": "change behavior from bad to good",
                "predicted_effect": "battery and search improve",
                "regression_risks": [],
            },
        )
        return _agent_result(role, log_dir)


def test_inner_unit_loop_feeds_failures_back_before_search(tmp_path: Path) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    editor = LearningEditor()
    summary = OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": BatteryAuthorAgent(), "search": editor},
    ).run()

    assert summary["incumbent_id"] == "iter001_candidate"
    assert editor.attempts == 2
    assert editor.saw_feedback
    # The search pool was evaluated once for the baseline and once for the
    # final candidate; the failed inner attempt never reached search.
    assert benchmark.calls == ["search", "search"]
    evidence = read_json(config.loop.run_dir / "iterations/iter_001/evidence.json")
    assert evidence["metadata"]["unit_attempts"] == 2
    assert evidence["target_improved"] is True


def test_exhausted_inner_loop_still_reaches_search(tmp_path: Path) -> None:
    from dataclasses import replace as dc_replace

    config = _config(tmp_path, ExperimentCondition.FULL)
    config = dc_replace(
        config, loop=dc_replace(config.loop, max_inner_retries=1)
    )
    benchmark = FakeBenchmark(tmp_path, natural_gain=False)
    OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": BatteryAuthorAgent(), "search": BadEditor()},
    ).run()

    assert benchmark.calls == ["search", "search"]
    decision = read_json(config.loop.run_dir / "iterations/iter_001/decision.json")
    assert decision["decision"] == "reject"
    assert decision["evidence"]["metadata"]["unit_attempts"] == 2
    assert decision["evidence"]["search_delta"] is not None


def test_proposer_sees_target_traces_score_and_world_model(tmp_path: Path) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL)
    editor = SearchAgent()
    OptimizationLoop(
        config,
        benchmark=FakeBenchmark(tmp_path, natural_gain=True),
        agents={"test_author": BatteryAuthorAgent(), "search": editor},
    ).run()

    prompt = editor.prompts[0]
    assert "target capability" in prompt
    assert "failing search traces of the incumbent" in prompt
    assert "current aggregate search score" in prompt
    assert "UT-design world model" in prompt
    candidate = config.loop.run_dir / "candidates/iter001_candidate"
    assert (candidate / "trace_evidence/manifest.json").is_file()
    assert (candidate / "ut_design_world_model.md").is_file()
    target = read_json(candidate / "target_capability.json")
    assert target["target_capability"] == "behavior-repair"
    # Anonymous instance results only: the mechanism description is the
    # group's, and the probes' ids/descriptions never reach the editor.
    assert target["instances"]
    assert target["instances"][0]["instance"] == "instance_01"
    assert "description" not in target["instances"][0]
    assert "instance_id" not in target["instances"][0]


class FailingAuthor:
    """An author whose agent always fails, e.g. an exhausted quota."""

    def run(self, *, role: str, prompt: str, workspace: Path, log_dir: Path):
        result = _agent_result(role, log_dir)
        return type(result)(
            role=role,
            returncode=1,
            duration_s=0.0,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            final_message_path=result.final_message_path,
        )


def test_consecutive_agent_failures_halt_and_hand_back_iterations(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ExperimentCondition.FULL, iterations=10)
    benchmark = FakeBenchmark(tmp_path, natural_gain=True)
    summary = OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": FailingAuthor(), "search": SearchAgent()},
    ).run()

    assert summary["status"] == "halted"
    state = read_json(config.loop.run_dir / "run_state.json")
    assert state["next_iteration"] == 1
    events = (config.loop.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"run_halted_after_consecutive_skips"' in events
    assert benchmark.calls == ["search"]

    resumed = OptimizationLoop(
        config,
        benchmark=benchmark,
        agents={"test_author": BatteryAuthorAgent(), "search": SearchAgent()},
    ).run()
    assert resumed["status"] in {"completed", "converged"}
    assert resumed["incumbent_id"] == "iter001_candidate"
