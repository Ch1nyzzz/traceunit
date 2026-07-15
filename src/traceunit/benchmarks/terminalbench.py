"""Terminal-Bench 2.0 adapter (self-contained; no WorldCalib dependency).

Terminal-Bench 2.0 is hosted through Harbor: each task ships a Docker image,
instruction, tests, and an oracle solution. TraceUnit optimizes an *editable*
Harbor agent -- a vendored copy of Terminus 2 under
``scaffolds/terminus_baseline/editable_terminus`` -- against those tasks. The
Harbor job itself runs out-of-process via :mod:`traceunit.benchmarks.harbor_worker`
so each candidate's edited scaffold is imported in a clean interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import traceunit
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.common import load_cached_evaluation
from traceunit.benchmarks.native_eval import normalize_trial_results
from traceunit.benchmarks.pools import (
    freeze_benchmark_plan,
    load_benchmark_plan,
    load_pool_items,
    pool_identity,
)
from traceunit.config import BenchmarkConfig
from traceunit.io import sha256_tree, write_json
from traceunit.models import BenchmarkEvaluation, BenchmarkPlan, PoolSliceRef

ADAPTER_CACHE_VERSION = 1
_AGENT_IMPORT_PATH = "editable_terminus.terminus_2:Terminus2"


def _packaged_baseline_source() -> Path:
    """Path to the vendored editable Terminus scaffold shipped with TraceUnit."""

    return (
        Path(traceunit.__file__).parent / "scaffolds" / "terminus_baseline"
    ).resolve()


class TerminalBenchAdapter(BenchmarkAdapter):
    name = "terminalbench"
    supports_agent_probe = False

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._plan: BenchmarkPlan | None = None
        self._dataset_dir: Path | None = None

    # -- preparation ---------------------------------------------------------

    def prepare(self, work_dir: Path) -> BenchmarkPlan:
        if not self.baseline_source().is_dir():
            raise FileNotFoundError(
                f"editable Terminus scaffold is missing: {self.baseline_source()}"
            )
        pool_dir = work_dir / "benchmark_data" / "terminalbench"
        pool_dir.mkdir(parents=True, exist_ok=True)
        self._dataset_dir = self._ensure_dataset(pool_dir)

        frozen_plan = pool_dir / "plan.json"
        if frozen_plan.is_file():
            plan = load_benchmark_plan(frozen_plan)
            self.bind_plan(plan)
            for pool in (plan.search, plan.final):
                load_pool_items(pool)
            return plan

        tasks = sorted(
            p.parent.name for p in self._dataset_dir.glob("*/task.toml")
        )
        if not tasks:
            raise FileNotFoundError(
                f"no Terminal-Bench tasks found under {self._dataset_dir}"
            )
        search_tasks, final_tasks = _split_tasks(
            tasks,
            seed=self.config.benchmark_seed,
            search_fraction=self.config.search_fraction,
        )
        search_tasks = _seeded_order(
            search_tasks, seed=self.config.benchmark_seed, namespace="search"
        )
        final_tasks = _seeded_order(
            final_tasks, seed=self.config.benchmark_seed, namespace="final"
        )
        if self.config.search_limit > 0:
            search_tasks = search_tasks[: self.config.search_limit]
        if self.config.final_limit > 0:
            final_tasks = final_tasks[: self.config.final_limit]
        if not search_tasks or not final_tasks:
            raise ValueError("Terminal-Bench search and final pools must be non-empty")

        self._plan = freeze_benchmark_plan(
            root=pool_dir,
            benchmark=self.name,
            search_items=[{"task_id": name, "split": "search"} for name in search_tasks],
            final_items=[{"task_id": name, "split": "final"} for name in final_tasks],
            cluster_key=lambda item: str(item["task_id"]),
        )
        return self._plan

    def preflight(self) -> None:
        if not shutil.which("docker"):
            raise RuntimeError("Terminal-Bench runtime is missing: docker")
        if not shutil.which("harbor"):
            raise RuntimeError("Terminal-Bench runtime is missing: harbor")
        try:
            import harbor  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "Terminal-Bench requires the harbor package to be importable"
            ) from exc
        if not self.config.dry_run and not os.environ.get(self.config.api_key_env):
            raise RuntimeError(
                f"Terminal-Bench solver key is missing: {self.config.api_key_env}"
            )

    def baseline_source(self) -> Path:
        return (
            self.config.scaffold_source_path or _packaged_baseline_source()
        ).resolve()

    def context(self) -> str:
        return """Terminal-Bench 2.0. The editable artifact is the Terminus terminal agent under
editable_terminus/: its control loop, prompt templates, output parser, tmux/tool execution,
context-summarization policy, and termination logic. The solver model is frozen. Each task runs
in its own Docker container with hidden tests. Never read, import, or encode the task tests,
verifier, oracle/reference solution, reward files, or any task-specific answer. Improve general
terminal-agent behavior only."""

    # -- evaluation ----------------------------------------------------------

    def evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        pool: PoolSliceRef,
        out_dir: Path,
    ) -> BenchmarkEvaluation:
        if self._plan is None:
            raise RuntimeError("prepare() must be called before evaluate()")
        if pool not in (self._plan.search, self._plan.final):
            raise ValueError(f"pool is not part of the prepared plan: {pool.slice_id}")
        if self._dataset_dir is None:
            raise RuntimeError("dataset directory was not resolved during prepare()")

        items = load_pool_items(pool)
        task_names = [str(item["task_id"]) for item in items]
        source = source.resolve()
        source_hash = sha256_tree(source / "editable_terminus")
        cache_fingerprint, cache_payload = self._cache_fingerprint(
            source_hash=source_hash, pool=pool, task_names=task_names
        )
        cached = load_cached_evaluation(out_dir)
        if (
            cached is not None
            and not self.config.force
            and cached.metadata.get("cache_fingerprint") == cache_fingerprint
        ):
            return cached

        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = out_dir / "harbor_spec.json"
        result_path = out_dir / "harbor_result.json"
        spec = self._build_spec(
            candidate_id=candidate_id,
            source=source,
            task_names=task_names,
            pool=pool,
            jobs_dir=out_dir / "jobs",
        )
        write_json(spec_path, spec)

        proc = subprocess.run(
            [
                str(self.config.unit_python or Path(sys.executable)),
                "-m",
                "traceunit.benchmarks.harbor_worker",
                "run",
                "--spec",
                str(spec_path),
                "--out",
                str(result_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (out_dir / "harbor_worker.log").write_text(proc.stdout, encoding="utf-8")
        if not result_path.is_file():
            raise RuntimeError(
                "Terminal-Bench worker produced no result envelope; see "
                f"{out_dir / 'harbor_worker.log'} (tail: {proc.stdout[-2000:]})"
            )

        evaluation = normalize_trial_results(
            result_path=result_path,
            benchmark=self.name,
            split=pool.slice_id,
            candidate_id=candidate_id,
            out_dir=out_dir,
        )
        evaluation = _attach_cache_metadata(
            evaluation, cache_fingerprint=cache_fingerprint, cache_payload=cache_payload
        )
        write_json(out_dir / "evaluation.json", evaluation.to_dict())
        return evaluation

    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = source / "editable_terminus"
        proc = subprocess.run(
            [
                str(self.config.unit_python or Path(sys.executable)),
                "-m",
                "compileall",
                "-q",
                str(target),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (out_dir / "smoke.txt").write_text(proc.stdout, encoding="utf-8")
        return proc.returncode == 0, proc.stdout[-4000:]

    def policy_violations(self, source: Path, diff_text: str) -> list[str]:
        added = "\n".join(
            line[1:]
            for line in diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        banned = {
            "tests/test_": "candidate refers to hidden task tests",
            "run-tests.sh": "candidate refers to the task verifier script",
            "solution.sh": "candidate refers to the oracle solution",
            "solution.yaml": "candidate refers to the oracle solution",
            "oracle": "candidate refers to the oracle solution",
            "reward.txt": "candidate refers to the verifier reward file",
            "task.toml": "candidate reads the task definition/verifier config",
            "harbor.agents.oracle": "candidate imports Harbor's oracle agent",
        }
        return [message for token, message in banned.items() if token in added]

    # -- helpers -------------------------------------------------------------

    def _ensure_dataset(self, pool_dir: Path) -> Path:
        if self.config.harbor_dataset_path is not None:
            dataset_dir = self.config.harbor_dataset_path.resolve()
            if not any(dataset_dir.glob("*/task.toml")):
                raise FileNotFoundError(
                    f"configured harbor_dataset_path has no tasks: {dataset_dir}"
                )
            return dataset_dir
        dataset_dir = (pool_dir / "dataset").resolve()
        if any(dataset_dir.glob("*/task.toml")):
            return dataset_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        ref = f"{self.config.harbor_dataset_name}@{self.config.harbor_dataset_version}"
        proc = subprocess.run(
            [
                "harbor",
                "datasets",
                "download",
                ref,
                "--output-dir",
                str(dataset_dir),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if not any(dataset_dir.glob("*/task.toml")):
            raise RuntimeError(
                f"failed to download Terminal-Bench dataset {ref}: {proc.stdout[-2000:]}"
            )
        return dataset_dir

    def _build_spec(
        self,
        *,
        candidate_id: str,
        source: Path,
        task_names: list[str],
        pool: PoolSliceRef,
        jobs_dir: Path,
    ) -> dict[str, Any]:
        jobs_dir_path = (self.config.harbor_jobs_dir or jobs_dir).resolve()
        run: dict[str, Any] = {
            "n_concurrent": self.config.concurrency,
            "agent_timeout_sec": self.config.timeout_s,
            "jobs_dir": str(jobs_dir_path),
            "delete_env": True,
            "quiet": True,
        }
        if self.config.dry_run:
            # No solver tokens: a nop agent exercises dataset/env/verifier plumbing.
            agent = {"name": "nop"}
            model: dict[str, Any] = {}
        else:
            agent = {"import_path": _AGENT_IMPORT_PATH}
            model = {
                "model_name": f"openai/{self.config.model}",
                "api_base": self.config.base_url,
                "api_key_env": self.config.api_key_env,
                "parser_name": self.config.parser_name,
            }
            if self.config.harbor_max_turns > 0:
                model["max_turns"] = self.config.harbor_max_turns
        return {
            "job_name": f"terminalbench-{pool.slice_id}-{candidate_id}",
            "candidate_id": candidate_id,
            "source": str(source),
            "dataset": {
                "path": str(self._dataset_dir),
                "task_names": task_names,
            },
            "agent": agent,
            "model": model,
            "run": run,
        }

    def _cache_fingerprint(
        self, *, source_hash: str, pool: PoolSliceRef, task_names: list[str]
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "adapter_cache_version": ADAPTER_CACHE_VERSION,
            "source_sha256": source_hash,
            "pool": pool_identity(pool),
            "task_names": task_names,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "parser_name": self.config.parser_name,
            "timeout_s": self.config.timeout_s,
            "concurrency": self.config.concurrency,
            "harbor_max_turns": self.config.harbor_max_turns,
            "dataset_name": self.config.harbor_dataset_name,
            "dataset_version": self.config.harbor_dataset_version,
            "dry_run": self.config.dry_run,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest(), payload


def _split_tasks(
    tasks: list[str], *, seed: int, search_fraction: float
) -> tuple[list[str], list[str]]:
    """Assign each independent task to the search or final pool by seeded hash."""

    if not 0 < search_fraction < 1:
        raise ValueError("search_fraction must be between 0 and 1")
    search: list[str] = []
    final: list[str] = []
    for task in tasks:
        value = (
            int.from_bytes(
                hashlib.sha256(f"{seed}:{task}".encode()).digest()[:8], "big"
            )
            / 2**64
        )
        (search if value < search_fraction else final).append(task)
    if tasks and (not search or not final):
        ordered = _seeded_order(tasks, seed=seed, namespace="fallback")
        cut = max(1, min(len(ordered) - 1, round(len(ordered) * search_fraction)))
        search, final = ordered[:cut], ordered[cut:]
    return search, final


def _seeded_order(tasks: list[str], *, seed: int, namespace: str) -> list[str]:
    return sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            f"{seed}:{namespace}:{task}".encode()
        ).hexdigest(),
    )


def _attach_cache_metadata(
    evaluation: BenchmarkEvaluation,
    *,
    cache_fingerprint: str,
    cache_payload: Mapping[str, Any],
) -> BenchmarkEvaluation:
    from dataclasses import replace

    return replace(
        evaluation,
        metadata={
            **evaluation.metadata,
            "cache_fingerprint": cache_fingerprint,
            "cache_identity": dict(cache_payload),
        },
    )
