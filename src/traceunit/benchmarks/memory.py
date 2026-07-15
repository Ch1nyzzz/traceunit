"""Adapters for the LoCoMo and LongMemEval memory benchmarks.

The data and evaluator stay host-side. Candidate editors receive a compact copy
of the memory scaffold implementation, while this adapter loads the candidate
inside the vendored memory substrate's dynamic loader and scores it against
redacted task views. The evaluation substrate is vendored in-repo under
``vendor/worldcalib_memory`` (Python package name kept as ``worldcalib`` so its
internal imports and the candidate seed package structure need no rewrite), so
the adapter carries no dependency on an external WorldCalib checkout.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import traceunit
from traceunit.agent_probe import run_declarative_probe
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.common import (
    load_cached_evaluation,
    normalize_worldcalib_result,
    worldcalib_import,
)
from traceunit.benchmarks.pools import (
    freeze_benchmark_plan,
    load_benchmark_plan,
    load_pool_items,
    pool_identity,
)
from traceunit.config import BenchmarkConfig
from traceunit.io import sha256_file, sha256_tree, write_json
from traceunit.models import BenchmarkEvaluation, BenchmarkPlan, PoolSliceRef


_ADAPTER_VERSION = 1
_MINIMAL_PACKAGE_INIT = '"""TraceUnit editable memory-scaffold package."""\n'


def _vendored_worldcalib_root() -> Path:
    """Root of the vendored memory evaluation substrate (holds src/worldcalib)."""

    return (
        Path(traceunit.__file__).parent / "vendor" / "worldcalib_memory"
    ).resolve()

_MEMORY_SOURCE_FILES = (
    "metrics.py",
    "model.py",
    "schemas.py",
    "scaffolds/__init__.py",
    "scaffolds/base.py",
    "memory/__init__.py",
    "memory/scaffolds/__init__.py",
    "memory/scaffolds/bm25_scaffold.py",
    "memory/scaffolds/memgpt_scaffold.py",
    "utils/__init__.py",
    "utils/text.py",
)
_LME_FILENAMES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}
_REDACTED_METADATA_KEYS = frozenset(
    {
        "benchmark",
        "variant",
        "question_type",
        "question_date",
        "category",
        "abstention",
    }
)


def _longmemeval_variant(value: str) -> str:
    raw = str(value or "s").strip().lower()
    aliases = {
        "small": "s",
        "longmemeval_s": "s",
        "longmemeval_s_cleaned": "s",
        "medium": "m",
        "longmemeval_m": "m",
        "longmemeval_m_cleaned": "m",
        "longmemeval_oracle": "oracle",
    }
    variant = aliases.get(raw, raw)
    if variant not in _LME_FILENAMES:
        allowed = ", ".join(sorted(_LME_FILENAMES))
        raise ValueError(
            f"unknown LongMemEval variant {value!r}; expected one of: {allowed}"
        )
    return variant


def _stable_rank(value: str, *, seed: int, namespace: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def _split_clustered_task_ids(
    task_to_cluster: Mapping[str, str], *, seed: int, search_fraction: float
) -> tuple[list[str], list[str]]:
    """Partition all task ids by cluster without opening a cluster in both pools."""

    clusters = sorted(
        set(task_to_cluster.values()),
        key=lambda item: _stable_rank(item, seed=seed, namespace="clusters"),
    )
    if len(clusters) < 2:
        raise ValueError("memory benchmark needs at least two independent clusters")
    search_cluster_count = round(len(clusters) * search_fraction)
    search_cluster_count = min(max(1, search_cluster_count), len(clusters) - 1)
    search_clusters = set(clusters[:search_cluster_count])
    search = [
        task_id
        for task_id in sorted(
            task_to_cluster,
            key=lambda item: _stable_rank(item, seed=seed, namespace="tasks"),
        )
        if task_to_cluster[task_id] in search_clusters
    ]
    final = [
        task_id
        for task_id in sorted(
            task_to_cluster,
            key=lambda item: _stable_rank(item, seed=seed, namespace="tasks"),
        )
        if task_to_cluster[task_id] not in search_clusters
    ]
    return search, final


def _limit_memory_pool(
    task_ids: list[str],
    *,
    task_to_cluster: Mapping[str, str],
    limit: int,
    seed: int,
    namespace: str,
) -> list[str]:
    """Take a deterministic, cluster-balanced prefix without changing pools."""

    if limit <= 0 or len(task_ids) <= limit:
        return list(task_ids)
    groups: dict[str, list[str]] = {}
    for task_id in task_ids:
        groups.setdefault(task_to_cluster[task_id], []).append(task_id)
    ordered_clusters = sorted(
        groups,
        key=lambda item: _stable_rank(item, seed=seed, namespace=f"{namespace}:clusters"),
    )
    for cluster in ordered_clusters:
        groups[cluster].sort(
            key=lambda item: _stable_rank(item, seed=seed, namespace=f"{namespace}:tasks")
        )
    selected: list[str] = []
    cursor = 0
    while len(selected) < limit:
        added = False
        for cluster in ordered_clusters:
            values = groups[cluster]
            if cursor < len(values):
                selected.append(values[cursor])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        cursor += 1
    return selected


def _validate_memory_pools(
    search: list[str], final: list[str], task_to_cluster: Mapping[str, str]
) -> None:
    if len(set(search)) != len(search):
        raise ValueError("memory search pool contains duplicate task ids")
    if len(set(final)) != len(final):
        raise ValueError("memory final pool contains duplicate task ids")
    unknown = (set(search) | set(final)) - set(task_to_cluster)
    if unknown:
        raise ValueError(f"memory pool contains unknown task ids: {sorted(unknown)[:3]}")
    overlap = set(search) & set(final)
    if overlap:
        raise ValueError(f"memory search/final pools overlap: {sorted(overlap)[:3]}")
    search_clusters = {task_to_cluster[item] for item in search}
    final_clusters = {task_to_cluster[item] for item in final}
    cluster_overlap = search_clusters & final_clusters
    if cluster_overlap:
        raise ValueError(
            "memory search/final conversation clusters overlap: "
            f"{sorted(cluster_overlap)[:3]}"
        )


def _load_pool_ids(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        values: Any = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        values = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(values, Mapping):
        values = next(
            (
                values[key]
                for key in ("task_ids", "tasks", "instances", "data")
                if isinstance(values.get(key), list)
            ),
            None,
        )
    if not isinstance(values, list):
        raise ValueError(f"unsupported memory pool shape: {path}")
    task_ids: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("task_id") or value.get("id")
        task_id = str(value or "").strip()
        if not task_id:
            raise ValueError(f"memory pool contains an empty task id: {path}")
        task_ids.append(task_id)
    return task_ids


def _file_digest(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "missing"


def _public_memory_example(example: Any) -> Any:
    """Return the task view supplied to editable candidate code.

    WorldCalib's shared schema carries the gold answer because its evaluator
    needs it.  Passing that object straight into candidate code makes the
    benchmark trivially reward-hackable, so all answer-bearing and identity
    fields are removed before build/retrieve/answer execute.
    """

    metadata = {
        str(key): value
        for key, value in dict(getattr(example, "metadata", {}) or {}).items()
        if str(key) in _REDACTED_METADATA_KEYS
    }
    return replace(
        example,
        task_id="traceunit-redacted",
        sample_id="traceunit-redacted",
        answer="",
        evidence=(),
        metadata=metadata,
    )


def _make_redacted_memory_runner(**kwargs: Any) -> Any:
    """Build a WorldCalib runner that never exposes gold data to a scaffold."""

    from worldcalib.evaluation import EvaluationRunner  # type: ignore
    from worldcalib.metrics import retrieval_oracle_prediction  # type: ignore
    from worldcalib.scaffolds.base import ScaffoldRun  # type: ignore
    from worldcalib.schemas import TaskResult  # type: ignore
    from worldcalib.utils.text import estimate_tokens  # type: ignore

    class RedactedMemoryEvaluationRunner(EvaluationRunner):
        def _build_sample_cache(self, scaffold, scaffold_name, config):
            # Cache construction in the parent runner receives the full example.
            # Disable it rather than allow a future cache setting to reintroduce
            # a gold-answer path into candidate ``build`` methods.
            return None

        def _evaluate_example(self, scaffold, config, example, *, build_cache=None):
            public_example = _public_memory_example(example)
            try:
                retrieve = getattr(scaffold, "retrieve", None)
                if self.dry_run and callable(retrieve):
                    state = scaffold.build(public_example, config)
                    hits = list(retrieve(state, public_example.question, config) or [])
                    retrieved_text = "\n\n".join(
                        str(getattr(hit, "text", "")) for hit in hits
                    )
                    prediction = retrieval_oracle_prediction(
                        retrieved_text, example.answer
                    )
                    run = ScaffoldRun(
                        prediction=prediction,
                        prompt_tokens=estimate_tokens(
                            public_example.question + "\n" + retrieved_text
                        ),
                        completion_tokens=estimate_tokens(prediction),
                        retrieved=hits,
                    )
                else:
                    run = scaffold.run(
                        public_example,
                        self.client,
                        config,
                        max_context_chars=self.max_context_chars,
                        dry_run=self.dry_run,
                    )
                score_result = self.score_run(example, run)
                metadata = dict(score_result.metadata or {})
                if not metadata.get("question_type"):
                    question_type = (getattr(example, "metadata", {}) or {}).get(
                        "question_type"
                    )
                    if question_type:
                        metadata["question_type"] = question_type
                return TaskResult(
                    task_id=example.task_id,
                    question=example.question,
                    gold_answer=example.answer,
                    prediction=run.prediction,
                    score=score_result.score,
                    passed=score_result.passed,
                    prompt_tokens=run.prompt_tokens,
                    completion_tokens=run.completion_tokens,
                    retrieved=[asdict(hit) for hit in run.retrieved],
                    metadata=metadata,
                )
            except Exception as exc:  # candidate code is intentionally untrusted
                return TaskResult(
                    task_id=example.task_id,
                    question=example.question,
                    gold_answer=example.answer,
                    prediction="",
                    score=0.0,
                    passed=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    retrieved=[],
                    metadata={
                        "run_status": "infra_error",
                        "evaluation_error": f"{type(exc).__name__}: {exc}",
                    },
                )

    return RedactedMemoryEvaluationRunner(**kwargs)


class _MemoryQAAdapter(BenchmarkAdapter):
    """Common host-side implementation for WorldCalib's memory QA datasets."""

    supports_agent_probe = True

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._plan: BenchmarkPlan | None = None
        self._seed_root: Path | None = None
        self._data_path: Path | None = None
        self._data_sha256 = ""
        self._examples_by_id: dict[str, Any] = {}

    @property
    def _is_longmemeval(self) -> bool:
        return self.name == "longmemeval"

    @property
    def _variant(self) -> str:
        return _longmemeval_variant(self.config.dataset_variant)

    def run_agent_probe(self, case, bundle, source, subject, output_dir):
        return run_declarative_probe(
            case=case,
            bundle=bundle,
            source=source,
            subject=subject,
            output_dir=output_dir,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key_env=self.config.api_key_env,
        )

    def prepare(self, work_dir: Path) -> BenchmarkPlan:
        root = _vendored_worldcalib_root()
        if not (root / "src/worldcalib/evaluation.py").is_file():
            raise FileNotFoundError(
                f"vendored memory evaluator is missing under {root}"
            )
        examples = self._ensure_dataset_loaded()
        task_to_cluster = self._task_to_cluster(examples)

        pool_dir = work_dir / "benchmark_data" / self.name
        frozen_plan = pool_dir / "plan.json"
        if frozen_plan.is_file():
            plan = load_benchmark_plan(frozen_plan)
            self.bind_plan(plan)
            self._seed_root = self._materialize_seed(pool_dir)
            return plan

        configured = {
            "search": self.config.search_data_path,
            "final": self.config.final_data_path,
        }
        if any(path is not None for path in configured.values()):
            missing = [
                name
                for name, path in configured.items()
                if path is None or not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "explicit memory pools require readable search and final files; "
                    f"missing={missing}"
                )
            search = _load_pool_ids(configured["search"])
            final = _load_pool_ids(configured["final"])
        else:
            search, final = _split_clustered_task_ids(
                task_to_cluster,
                seed=self.config.benchmark_seed,
                search_fraction=self.config.search_fraction,
            )
        search = _limit_memory_pool(
            search,
            task_to_cluster=task_to_cluster,
            limit=self.config.search_limit,
            seed=self.config.benchmark_seed,
            namespace="search",
        )
        final = _limit_memory_pool(
            final,
            task_to_cluster=task_to_cluster,
            limit=self.config.final_limit,
            seed=self.config.benchmark_seed,
            namespace="final",
        )
        if not search or not final:
            raise ValueError(f"{self.name} search and final pools must be non-empty")
        _validate_memory_pools(search, final, task_to_cluster)
        self._plan = freeze_benchmark_plan(
            root=pool_dir,
            benchmark=self.name,
            search_items=search,
            final_items=final,
            cluster_key=lambda task_id: task_to_cluster[str(task_id)],
        )
        self._seed_root = self._materialize_seed(pool_dir)
        return self._plan

    def bind_plan(self, plan: BenchmarkPlan) -> None:
        super().bind_plan(plan)
        examples = self._ensure_dataset_loaded()
        _validate_memory_pools(
            load_pool_items(plan.search),
            load_pool_items(plan.final),
            self._task_to_cluster(examples),
        )

    def _ensure_dataset_loaded(self) -> list[Any]:
        if self._data_path is None:
            self._data_path = self._resolve_data_path()
        if not self._data_sha256:
            self._data_sha256 = sha256_file(self._data_path)
        if not self._examples_by_id:
            examples = self._load_examples()
            self._examples_by_id = {str(item.task_id): item for item in examples}
            if len(self._examples_by_id) != len(examples):
                raise ValueError(f"{self.name} data has duplicate task ids")
            if len(examples) < 2:
                raise ValueError(f"{self.name} data must contain at least two tasks")
        return list(self._examples_by_id.values())

    @staticmethod
    def _task_to_cluster(examples: list[Any]) -> dict[str, str]:
        return {
            str(item.task_id): str(item.sample_id) or str(item.task_id)
            for item in examples
        }

    def preflight(self) -> None:
        if self.config.repeats != 1:
            raise RuntimeError(
                f"{self.name} currently supports repeats=1; repeated memory "
                "evaluations must not be requested silently"
            )
        if self.config.dry_run:
            return
        if not os.environ.get(self.config.api_key_env):
            raise RuntimeError(
                f"{self.name} target-model key is missing: {self.config.api_key_env}"
            )
        if self._is_longmemeval and self.config.use_llm_judge:
            if not os.environ.get(self.config.judge_api_key_env):
                raise RuntimeError(
                    "LongMemEval judge key is missing: "
                    f"{self.config.judge_api_key_env}"
                )

    def baseline_source(self) -> Path:
        if self._seed_root is None:
            raise RuntimeError("prepare() must be called before baseline_source()")
        return self._seed_root

    def context(self) -> str:
        evaluator = (
            "LongMemEval's official-style LLM yes/no judge"
            if self._is_longmemeval
            else "the LoCoMo token-F1 scorer"
        )
        return f"""{self.name} conversational-memory QA. The editable source is a minimal
WorldCalib memory-scaffold package under src/worldcalib/. The host owns raw data,
conversation split manifests, answer scoring, model credentials, and {evaluator}.
Candidate code receives only a redacted question/conversation view: no gold answer,
evidence annotations, task id, or sample id. Modify general retrieval, packing,
compression, prompting, or verification behavior. Never read data files, access
scorers/evaluators, inspect result artifacts, branch on task identity, or encode
dataset-specific answers. Proxy tests must use synthetic conversations or mocks,
not benchmark records."""

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
        task_ids = [str(item) for item in load_pool_items(pool)]
        try:
            examples = [self._examples_by_id[task_id] for task_id in task_ids]
        except KeyError as exc:
            raise ValueError(
                f"frozen {self.name} pool references an unavailable task: {exc.args[0]}"
            ) from exc
        source_hash = sha256_tree(source)
        fingerprint, identity = self._evaluation_cache_fingerprint(
            source_hash=source_hash, pool=pool
        )
        cached = load_cached_evaluation(out_dir)
        if (
            cached is not None
            and not self.config.force
            and cached.metadata.get("cache_fingerprint") == fingerprint
        ):
            return cached
        out_dir.mkdir(parents=True, exist_ok=True)

        with worldcalib_import(_vendored_worldcalib_root()):
            from worldcalib.dynamic import load_candidate_scaffold  # type: ignore
            from worldcalib.scaffolds.base import ScaffoldConfig  # type: ignore

            score_run = self._score_run()
            runner = _make_redacted_memory_runner(
                examples=examples,
                out_dir=out_dir,
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=os.environ.get(self.config.api_key_env, ""),
                timeout_s=self.config.timeout_s,
                dry_run=self.config.dry_run,
                max_context_chars=self.config.max_context_chars,
                max_eval_workers=self.config.concurrency,
                # This adapter owns a stronger source+data+pool fingerprint. Do
                # not let WorldCalib's candidate-id-only cache return stale work.
                force=True,
                score_run=score_run,
            )
            scaffold = load_candidate_scaffold(
                {
                    "scaffold_name": "memgpt_source",
                    "source_project_path": str(source.resolve()),
                },
                project_root=_vendored_worldcalib_root(),
            )
            config = ScaffoldConfig(
                top_k=self.config.memory_top_k,
                window=self.config.memory_window,
                extra={
                    "benchmark": self.name,
                    "source_family": "memgpt",
                    "dataset_variant": self._variant if self._is_longmemeval else None,
                    "scoring_method": self._scoring_method(),
                },
            )
            result = runner.evaluate_scaffold(
                scaffold=scaffold,
                scaffold_name="memgpt_source",
                config=config,
                candidate_id=candidate_id,
            )

        evaluation = normalize_worldcalib_result(
            result_path=Path(result.result_path),
            benchmark=self.name,
            split=pool.slice_id,
            candidate_id=candidate_id,
            out_dir=out_dir,
        )
        judge_tokens = _judge_token_total(Path(result.result_path))
        evaluation = replace(
            evaluation,
            cost=evaluation.cost + judge_tokens,
            metadata={
                **evaluation.metadata,
                "cache_fingerprint": fingerprint,
                "cache_identity": identity,
                "judge_tokens": judge_tokens,
                "source_sha256": source_hash,
            },
        )
        write_json(out_dir / "evaluation.json", evaluation.to_dict())
        return evaluation

    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        package = source / "src/worldcalib"
        proc = subprocess.run(
            [
                str(self.config.unit_python or Path(sys.executable)),
                "-m",
                "compileall",
                "-q",
                str(package),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        message = proc.stdout
        if proc.returncode == 0:
            try:
                with worldcalib_import(_vendored_worldcalib_root()):
                    from worldcalib.dynamic import load_candidate_scaffold  # type: ignore

                    load_candidate_scaffold(
                        {
                            "scaffold_name": "memgpt_source",
                            "source_project_path": str(source.resolve()),
                        },
                        project_root=_vendored_worldcalib_root(),
                    )
            except Exception as exc:  # import failures are smoke failures
                message += f"\n{type(exc).__name__}: {exc}\n"
                proc_returncode = 1
            else:
                proc_returncode = 0
        else:
            proc_returncode = proc.returncode
        (out_dir / "smoke.txt").write_text(message, encoding="utf-8")
        return proc_returncode == 0, message[-4000:]

    def policy_violations(self, source: Path, diff_text: str) -> list[str]:
        del source
        added = "\n".join(
            line[1:]
            for line in diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ).lower()
        banned = {
            "load_locomo_examples": "candidate loads the raw LoCoMo dataset",
            "load_longmemeval_examples": "candidate loads the raw LongMemEval dataset",
            "locomo10.json": "candidate refers to raw LoCoMo data",
            "longmemeval_": "candidate refers to raw LongMemEval data",
            "data/locomo": "candidate refers to raw LoCoMo data",
            "data/longmemeval": "candidate refers to raw LongMemEval data",
            "example.answer": "candidate reads the gold answer field",
            "gold_answer": "candidate reads or stores gold answers",
            "score_prediction": "candidate calls a host scoring helper",
            "worldcalib.metrics": "candidate imports a host scoring helper",
            "open(": "candidate performs host filesystem access",
            "pathlib": "candidate performs host filesystem access",
            "subprocess": "candidate launches a host subprocess",
        }
        return [message for token, message in banned.items() if token in added]

    def _resolve_data_path(self) -> Path:
        if self.config.data_path is not None:
            path = self.config.data_path.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"configured {self.name} data is missing: {path}")
            return path
        if self._is_longmemeval:
            filename = _LME_FILENAMES[self._variant]
            subdir = "longmemeval"
        else:
            filename = "locomo10.json"
            subdir = "locomo"
        candidates = (
            Path("/data/home/yuhan/Optimizer1/data") / subdir / filename,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        searched = ", ".join(str(item) for item in candidates)
        raise FileNotFoundError(
            f"{self.name} data is missing. Set benchmark.data_path; searched: {searched}. "
            "Downloads are deliberately disabled during a TraceUnit run."
        )

    def _load_examples(self) -> list[Any]:
        if self._data_path is None:
            raise RuntimeError("memory data path has not been resolved")
        with worldcalib_import(_vendored_worldcalib_root()):
            if self._is_longmemeval:
                from worldcalib.memory.longmemeval import (  # type: ignore
                    load_longmemeval_examples,
                )

                return load_longmemeval_examples(
                    data_path=self._data_path,
                    variant=self._variant,
                    question_types=self.config.memory_question_types,
                )
            from worldcalib.memory.locomo import load_locomo_examples  # type: ignore

            return load_locomo_examples(data_path=self._data_path)

    def _materialize_seed(self, pool_dir: Path) -> Path:
        seed_root = pool_dir / "memory_scaffold_seed"
        package_root = seed_root / "src/worldcalib"
        expected = [package_root / relative for relative in _MEMORY_SOURCE_FILES]
        if package_root.is_dir() and all(path.is_file() for path in expected):
            return seed_root
        if seed_root.exists():
            shutil.rmtree(seed_root)
        source_root = _vendored_worldcalib_root() / "src/worldcalib"
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text(
            _MINIMAL_PACKAGE_INIT, encoding="utf-8"
        )
        for relative in _MEMORY_SOURCE_FILES:
            src = source_root / relative
            if not src.is_file():
                raise FileNotFoundError(
                    f"WorldCalib memory source is missing: {src}"
                )
            dest = package_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return seed_root

    def _score_run(self) -> Any:
        if not self._is_longmemeval:
            return None
        from worldcalib.memory.longmemeval import (  # type: ignore
            LongMemEvalJudge,
            _fallback_score_run,
        )

        if self.config.dry_run or not self.config.use_llm_judge:
            return _fallback_score_run
        return LongMemEvalJudge(
            model=self.config.judge_model,
            base_url=self.config.judge_base_url,
            api_key=os.environ.get(self.config.judge_api_key_env, ""),
            timeout_s=self.config.judge_timeout_s,
        ).score_run

    def _scoring_method(self) -> str:
        if not self._is_longmemeval:
            return "token_f1"
        return (
            "longmemeval_llm_judge"
            if self.config.use_llm_judge and not self.config.dry_run
            else "token_f1"
        )

    def _evaluation_cache_fingerprint(
        self, *, source_hash: str, pool: PoolSliceRef
    ) -> tuple[str, dict[str, Any]]:
        root = _vendored_worldcalib_root() / "src/worldcalib"
        harness_files = [
            root / "evaluation.py",
            root / "dynamic.py",
            root / "scaffolds/base.py",
            root / "memory/scaffolds/memgpt_scaffold.py",
            root / ("memory/longmemeval.py" if self._is_longmemeval else "memory/locomo.py"),
        ]
        payload = {
            "adapter_version": _ADAPTER_VERSION,
            "benchmark": self.name,
            "harness_sha256": {
                str(path.relative_to(root)): _file_digest(path) for path in harness_files
            },
            "data_sha256": self._data_sha256,
            "source_sha256": source_hash,
            "pool": pool_identity(pool),
            "model": self.config.model,
            "base_url": self.config.base_url,
            "api_key_env": self.config.api_key_env,
            "timeout_s": self.config.timeout_s,
            "concurrency": self.config.concurrency,
            "dry_run": self.config.dry_run,
            "memory_top_k": self.config.memory_top_k,
            "memory_window": self.config.memory_window,
            "max_context_chars": self.config.max_context_chars,
            "dataset_variant": self._variant if self._is_longmemeval else None,
            "memory_question_types": list(self.config.memory_question_types),
            "use_llm_judge": self.config.use_llm_judge if self._is_longmemeval else False,
            "judge_model": self.config.judge_model if self._is_longmemeval else None,
            "judge_base_url": self.config.judge_base_url if self._is_longmemeval else None,
            "judge_api_key_env": self.config.judge_api_key_env if self._is_longmemeval else None,
            "judge_timeout_s": self.config.judge_timeout_s if self._is_longmemeval else None,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return fingerprint, payload


class LocomoAdapter(_MemoryQAAdapter):
    name = "locomo"


class LongMemEvalAdapter(_MemoryQAAdapter):
    name = "longmemeval"


def _judge_token_total(result_path: Path) -> float:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    total = 0
    for task in payload.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        metadata = task.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        total += int(metadata.get("judge_prompt_tokens") or 0)
        total += int(metadata.get("judge_completion_tokens") or 0)
    return float(total)


__all__ = [
    "LocomoAdapter",
    "LongMemEvalAdapter",
    "_split_clustered_task_ids",
    "_validate_memory_pools",
]
