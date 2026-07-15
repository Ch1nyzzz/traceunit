"""Memory scaffold evaluation runner."""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from worldcalib.memory.locomo import (
    default_data_path,
    load_locomo_examples,
    prepare_locomo,
    select_split,
)
from worldcalib.metrics import passed, score_prediction
from worldcalib.model import DEFAULT_BASE_URL, DEFAULT_MODEL, LocalModelClient
from worldcalib.pareto import ParetoPoint, save_frontier
from worldcalib.schemas import CandidateResult, LocomoExample, TaskResult
from worldcalib.memory.scaffolds import (
    DEFAULT_MEMORY_EVOLUTION_SEED_SCAFFOLDS as DEFAULT_EVOLUTION_SEED_SCAFFOLDS,
    DEFAULT_MEMORY_SCAFFOLD_TOP_KS as DEFAULT_SCAFFOLD_TOP_KS,
    build_memory_scaffold as build_scaffold,
)
from worldcalib.scaffolds.base import (
    MemoryScaffold,
    ScaffoldConfig,
    ScaffoldRun,
)


BUILD_CACHE_SOURCE_FAMILIES = frozenset()
BUILD_CACHE_SCAFFOLDS = frozenset()


@dataclass
class _CachedState:
    state: Any
    lock: threading.Lock


@dataclass
class _BuildCache:
    build_key: str
    states: dict[str, _CachedState]
    built_samples: list[str]
    reused_samples: list[str]


@dataclass(frozen=True)
class ScoreResult:
    """Score assigned to one scaffold answer."""

    score: float
    passed: bool
    metadata: dict[str, Any] | None = None


ScoreRunFn = Callable[[LocomoExample, ScaffoldRun], ScoreResult]


class EvaluationRunner:
    """Evaluate memory scaffold candidates and write a Pareto frontier."""

    def __init__(
        self,
        *,
        examples: list[LocomoExample],
        out_dir: Path,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        timeout_s: int = 300,
        dry_run: bool = False,
        max_context_chars: int = 6000,
        max_eval_workers: int = 1,
        force: bool = False,
        score_run: ScoreRunFn | None = None,
    ) -> None:
        self.examples = examples
        self.out_dir = out_dir
        self.dry_run = dry_run
        self.max_context_chars = max_context_chars
        self.max_eval_workers = max(1, int(max_eval_workers))
        self.force = force
        self.score_run = score_run or _default_score_run
        self._build_state_cache: dict[str, dict[str, _CachedState]] = {}
        self.client = LocalModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    def evaluate_candidate(
        self,
        *,
        scaffold_name: str,
        config: ScaffoldConfig,
        candidate_id: str,
    ) -> CandidateResult:
        scaffold = build_scaffold(scaffold_name)
        return self.evaluate_scaffold(
            scaffold=scaffold,
            scaffold_name=scaffold_name,
            config=config,
            candidate_id=candidate_id,
        )

    def evaluate_scaffold(
        self,
        *,
        scaffold: MemoryScaffold,
        scaffold_name: str,
        config: ScaffoldConfig,
        candidate_id: str,
    ) -> CandidateResult:
        """Evaluate a built-in or dynamically proposed memory scaffold."""

        candidate_dir = self.out_dir / "candidate_results"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        result_path = candidate_dir / f"{candidate_id}.json"
        if not self.force:
            existing = _load_candidate_result(
                result_path,
                candidate_id=candidate_id,
                scaffold_name=scaffold_name,
                config=config,
            )
            if existing is not None:
                return existing

        build_cache = self._build_sample_cache(scaffold, scaffold_name, config)
        if self.max_eval_workers == 1 or len(self.examples) <= 1:
            task_results = [
                self._evaluate_example(
                    scaffold,
                    config,
                    example,
                    build_cache=build_cache.states if build_cache is not None else None,
                )
                for example in self.examples
            ]
        else:
            with ThreadPoolExecutor(max_workers=self.max_eval_workers) as pool:
                task_results = list(
                    pool.map(
                        lambda example: self._evaluate_example(
                            scaffold,
                            config,
                            example,
                            build_cache=build_cache.states if build_cache is not None else None,
                        ),
                        self.examples,
                    )
                )

        count = len(task_results)
        passrate = sum(1 for item in task_results if item.passed) / count if count else 0.0
        average_score = sum(item.score for item in task_results) / count if count else 0.0
        prompt_tokens = sum(item.prompt_tokens for item in task_results)
        completion_tokens = sum(item.completion_tokens for item in task_results)
        token_consuming = prompt_tokens + completion_tokens
        candidate = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=scaffold_name,
            passrate=passrate,
            average_score=average_score,
            token_consuming=token_consuming,
            avg_token_consuming=(token_consuming / count if count else 0.0),
            avg_prompt_tokens=(prompt_tokens / count if count else 0.0),
            avg_completion_tokens=(completion_tokens / count if count else 0.0),
            count=count,
            config=config.to_dict(),
            result_path=str(result_path),
        )
        payload = {
            "candidate": candidate.to_dict(),
            "tasks": [item.to_dict() for item in task_results],
            "score_breakdown": _score_breakdown(task_results),
            "build_cache": {
                "enabled": build_cache is not None,
                "build_key": build_cache.build_key if build_cache is not None else None,
                "sample_count": len(build_cache.states) if build_cache is not None else 0,
                "samples": sorted(build_cache.states) if build_cache is not None else [],
                "built_samples": build_cache.built_samples if build_cache is not None else [],
                "reused_samples": build_cache.reused_samples if build_cache is not None else [],
            },
        }
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return candidate

    def _build_sample_cache(
        self,
        scaffold: MemoryScaffold,
        scaffold_name: str,
        config: ScaffoldConfig,
    ) -> _BuildCache | None:
        build_key = _build_cache_key(scaffold, scaffold_name, config)
        if build_key is None:
            return None

        examples_by_sample: dict[str, LocomoExample] = {}
        for example in self.examples:
            examples_by_sample.setdefault(example.sample_id, example)
        if len(examples_by_sample) >= len(self.examples):
            return None

        states = self._build_state_cache.setdefault(build_key, {})
        missing_items = [
            item for item in examples_by_sample.items() if item[0] not in states
        ]
        reused_samples = sorted(set(examples_by_sample) - {sample_id for sample_id, _ in missing_items})

        def build_one(item: tuple[str, LocomoExample]) -> tuple[str, _CachedState]:
            sample_id, example = item
            return sample_id, _CachedState(
                state=scaffold.build(example, config),
                lock=threading.Lock(),
            )

        if self.max_eval_workers == 1 or len(missing_items) <= 1:
            built = dict(build_one(item) for item in missing_items)
        else:
            workers = min(self.max_eval_workers, len(missing_items))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                built = dict(pool.map(build_one, missing_items))
        states.update(built)
        candidate_states = {
            sample_id: states[sample_id]
            for sample_id in examples_by_sample
        }
        return _BuildCache(
            build_key=build_key,
            states=candidate_states,
            built_samples=sorted(built),
            reused_samples=reused_samples,
        )

    def _evaluate_example(
        self,
        scaffold: MemoryScaffold,
        config: ScaffoldConfig,
        example: LocomoExample,
        *,
        build_cache: dict[str, _CachedState] | None = None,
    ) -> TaskResult:
        cached = (build_cache or {}).get(example.sample_id)
        try:
            if cached is None:
                run = scaffold.run(
                    example,
                    self.client,
                    config,
                    max_context_chars=self.max_context_chars,
                    dry_run=self.dry_run,
                )
            else:
                with cached.lock:
                    run = scaffold.answer(
                        cached.state,
                        example,
                        self.client,
                        config,
                        max_context_chars=self.max_context_chars,
                        dry_run=self.dry_run,
                    )
            score_result = self.score_run(example, run)
            task_metadata = dict(score_result.metadata or {})
            # Per-question-type score_breakdown groups on metadata["question_type"].
            # The longmemeval judge echoes it, but locomo's default scorer does
            # not — fall back to the example's declared type/category so locomo
            # also gets a per-category breakdown instead of a single "all" bucket.
            if not task_metadata.get("question_type"):
                qt = (getattr(example, "metadata", None) or {}).get("question_type")
                if qt:
                    task_metadata["question_type"] = qt
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
                metadata=task_metadata,
            )
        except Exception as exc:  # noqa: BLE001 - candidate code can raise anything
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
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                },
            )


def _default_score_run(example: LocomoExample, run: ScaffoldRun) -> ScoreResult:
    score = score_prediction(run.prediction, example.answer)
    return ScoreResult(score=score, passed=passed(score))


def _score_breakdown(task_results: list[TaskResult]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[TaskResult]] = {}
    for item in task_results:
        key = str(item.metadata.get("question_type") or "all")
        grouped.setdefault(key, []).append(item)
    return {
        key: {
            "count": len(items),
            "passrate": (
                sum(1 for item in items if item.passed) / len(items)
                if items
                else 0.0
            ),
            "average_score": (
                sum(item.score for item in items) / len(items)
                if items
                else 0.0
            ),
        }
        for key, items in sorted(grouped.items())
    }


def make_initial_candidate_grid(
    *,
    scaffolds: Iterable[str] | None = None,
    top_k_variants: Iterable[int] | None = None,
    scaffold_extra: Mapping[str, Mapping[str, object]] | None = None,
) -> list[tuple[str, ScaffoldConfig, str]]:
    """Build the initial scaffold/config grid for evaluation."""

    selected = list(scaffolds or DEFAULT_EVOLUTION_SEED_SCAFFOLDS)
    extras = scaffold_extra or {}
    out: list[tuple[str, ScaffoldConfig, str]] = []
    for scaffold_name in selected:
        top_k_values = (
            [DEFAULT_SCAFFOLD_TOP_KS.get(scaffold_name, 8)]
            if top_k_variants is None
            else [int(item) for item in top_k_variants]
        )
        for top_k in top_k_values:
            config = ScaffoldConfig(
                top_k=int(top_k),
                window=1,
                extra=dict(extras.get(scaffold_name, {})),
            )
            out.append((scaffold_name, config, f"{scaffold_name}_top{top_k}"))
    return out


def _build_cache_key(
    scaffold: MemoryScaffold,
    scaffold_name: str,
    config: ScaffoldConfig,
) -> str | None:
    source_family = _source_family_for_build_cache(scaffold_name, config)
    if source_family is None:
        return None

    payload = {
        "source_family": source_family,
        "build_tag": _build_tag(scaffold, scaffold_name, config),
        "build_config": _build_relevant_config(config.extra),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return f"{source_family}:{digest}"


def _source_family_for_build_cache(scaffold_name: str, config: ScaffoldConfig) -> str | None:
    source_family = str(config.extra.get("source_family") or "").lower()
    if source_family in BUILD_CACHE_SOURCE_FAMILIES:
        return source_family
    return None


def _build_tag(scaffold: MemoryScaffold, scaffold_name: str, config: ScaffoldConfig) -> str:
    explicit = str(
        config.extra.get("build_tag")
        or config.extra.get("build_cache_tag")
        or ""
    ).strip()
    if explicit:
        return explicit

    if scaffold_name in BUILD_CACHE_SCAFFOLDS:
        return f"builtin:{scaffold_name}:{_source_file_digest(scaffold)}"

    module_name = str(config.extra.get("module") or scaffold.__class__.__module__)
    class_name = str(config.extra.get("class") or scaffold.__class__.__qualname__)
    module_path = str(config.extra.get("module_path") or "")
    return f"candidate:{scaffold_name}:{module_name}:{class_name}:{module_path}:{_source_file_digest(scaffold)}"


def _source_file_digest(scaffold: MemoryScaffold) -> str:
    try:
        source_path = inspect.getsourcefile(scaffold.__class__)
    except TypeError:
        source_path = None
    if not source_path:
        return "unknown-source"
    path = Path(source_path)
    if not path.exists():
        return str(path)
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:20]


def _build_relevant_config(extra: Mapping[str, object]) -> dict[str, object]:
    ignored = {
        "build_cache_tag",
        "build_tag",
        "changes",
        "class",
        "cost_level",
        "factory",
        "hypothesis",
        "module",
        "module_path",
        "optimization_cell",
        "optimization_target",
        "rerank",
        "source_family",
        "threshold",
        "touched_cells",
    }
    return {
        str(key): value
        for key, value in extra.items()
        if str(key) not in ignored
    }


def run_initial_frontier(
    *,
    split: str = "train",
    limit: int = 0,
    out_dir: Path,
    scaffolds: Iterable[str] | None = None,
    top_k_variants: Iterable[int] | None = None,
    scaffold_extra: Mapping[str, Mapping[str, object]] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    timeout_s: int = 300,
    dry_run: bool = False,
    max_context_chars: int = 6000,
    candidate_id_prefix: str = "",
    max_eval_workers: int = 1,
    force: bool = False,
    pareto_quality_threshold: float = 0.125,
) -> dict[str, object]:
    """Evaluate initial scaffolds and write summary + Pareto frontier."""

    if not default_data_path().exists():
        prepare_locomo()
    all_examples = load_locomo_examples()
    examples = select_split(all_examples, split=split)
    if limit:
        examples = examples[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    runner = EvaluationRunner(
        examples=examples,
        out_dir=out_dir,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
        dry_run=dry_run,
        max_context_chars=max_context_chars,
        max_eval_workers=max_eval_workers,
        force=force,
    )

    selected_scaffolds = list(scaffolds or DEFAULT_EVOLUTION_SEED_SCAFFOLDS)
    selected_top_k = None if top_k_variants is None else [int(item) for item in top_k_variants]
    grid = make_initial_candidate_grid(
        scaffolds=selected_scaffolds,
        top_k_variants=selected_top_k,
        scaffold_extra=scaffold_extra,
    )
    started = time.time()
    summary_candidates: list[CandidateResult] = []
    for scaffold_name, config, candidate_id in grid:
        candidate_id = f"{candidate_id_prefix}{candidate_id}"
        summary_candidates.append(
            runner.evaluate_candidate(
                scaffold_name=scaffold_name,
                config=config,
                candidate_id=candidate_id,
            )
        )

    summary_candidates = sorted(
        summary_candidates,
        key=lambda item: (item.scaffold_name, int(item.config.get("top_k", 0)), item.candidate_id),
    )
    frontier_path = out_dir / "pareto_frontier.json"
    save_frontier(
        frontier_path,
        [
            ParetoPoint(
                candidate_id=item.candidate_id,
                scaffold_name=item.scaffold_name,
                passrate=item.passrate,
                token_consuming=item.token_consuming,
                avg_token_consuming=item.avg_token_consuming,
                average_score=item.average_score,
                result_path=item.result_path,
                config=item.config,
            )
            for item in summary_candidates
        ],
        quality_gap_threshold=pareto_quality_threshold,
    )
    summary = {
        "split": split,
        "limit": limit,
        "count": len(examples),
        "dry_run": dry_run,
        "model": model,
        "base_url": base_url,
        "max_context_chars": max_context_chars,
        "max_eval_workers": max_eval_workers,
        "force": force,
        "pareto_quality_threshold": pareto_quality_threshold,
        "scaffolds": selected_scaffolds,
        "top_k_variants": selected_top_k,
        "scaffold_top_k": {
            item.scaffold_name: int(item.config.get("top_k", 0))
            for item in summary_candidates
        },
        "duration_s": time.time() - started,
        "candidate_count": len(summary_candidates),
        "candidates": [candidate.to_dict() for candidate in summary_candidates],
        "pareto_frontier_path": str(frontier_path),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _load_candidate_result(
    result_path: Path,
    *,
    candidate_id: str,
    scaffold_name: str,
    config: ScaffoldConfig,
) -> CandidateResult | None:
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        candidate = CandidateResult.from_dict(payload["candidate"])
    except Exception:
        return None
    if (
        candidate.candidate_id != candidate_id
        or candidate.scaffold_name != scaffold_name
        or candidate.config != config.to_dict()
    ):
        return None
    return candidate
