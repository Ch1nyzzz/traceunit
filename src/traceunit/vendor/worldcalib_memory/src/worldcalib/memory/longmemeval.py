"""LongMemEval data import and memory-scaffold benchmark runner."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

from worldcalib.evaluation import EvaluationRunner, ScoreResult, make_initial_candidate_grid
from worldcalib.memory.locomo import project_root
from worldcalib.metrics import passed, score_prediction
from worldcalib.model import DEFAULT_BASE_URL, DEFAULT_MODEL, LocalModelClient
from worldcalib.pareto import ParetoPoint, save_frontier
from worldcalib.schemas import CandidateResult, ConversationTurn, LocomoExample
from worldcalib.scaffolds.base import ScaffoldRun


LONGMEMEVAL_REMOTE_URLS = {
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
}
LONGMEMEVAL_FILENAMES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}
DEFAULT_LONGMEMEVAL_SCAFFOLDS = ("memgpt_source",)
DEFAULT_LONGMEMEVAL_TRAIN_SIZE = 100
DEFAULT_LONGMEMEVAL_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV = "TOGETHER_API_KEY"

_QUESTION_TYPE_TO_CATEGORY = {
    "single-session-user": 1,
    "single-session-assistant": 1,
    "single-session-preference": 3,
    "temporal-reasoning": 2,
    "knowledge-update": 3,
    "multi-session": 3,
}


def default_data_path(variant: str = "s") -> Path:
    key = _normalize_variant(variant)
    return project_root() / "data" / "longmemeval" / LONGMEMEVAL_FILENAMES[key]


def default_split_path(variant: str = "s") -> Path:
    key = _normalize_variant(variant)
    return project_root() / "data" / "longmemeval" / f"splits_{key}.json"


def prepare_longmemeval(
    *,
    variant: str = "s",
    dest: Path | None = None,
    source: Path | None = None,
    allow_download: bool = False,
    warmup_size: int = 0,
    train_size: int = DEFAULT_LONGMEMEVAL_TRAIN_SIZE,
    seed: int = 13,
) -> dict[str, Any]:
    """Materialize one LongMemEval JSON file and write deterministic splits."""

    key = _normalize_variant(variant)
    dest = dest or default_data_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source is not None and source.exists():
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    elif not dest.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"{dest} is missing. Pass --source with a downloaded LongMemEval "
                "JSON file or --allow-download to fetch it from Hugging Face."
            )
        _download_longmemeval(key, dest)

    examples = load_longmemeval_examples(data_path=dest, variant=key)
    split_payload = build_splits(
        examples,
        variant=key,
        warmup_size=warmup_size,
        train_size=train_size,
        seed=seed,
    )
    split_path = default_split_path(key)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "benchmark": "longmemeval",
        "variant": key,
        "data_path": str(dest),
        "split_path": str(split_path),
        "count": len(examples),
        "warmup_size": len(split_payload["splits"]["warmup"]),
        "train_size": len(split_payload["splits"]["train"]),
        "test_size": len(split_payload["splits"]["test"]),
    }


def load_longmemeval_examples(
    *,
    data_path: Path | None = None,
    variant: str = "s",
    limit: int = 0,
    question_types: Iterable[str] | None = None,
) -> list[LocomoExample]:
    """Load LongMemEval rows as the shared memory-QA example schema."""

    key = _normalize_variant(variant)
    path = data_path or default_data_path(key)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected top-level list in {path}")

    allowed_types = {item.strip() for item in (question_types or ()) if item.strip()}
    examples: list[LocomoExample] = []
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        question_type = str(sample.get("question_type") or "").strip()
        if allowed_types and question_type not in allowed_types:
            continue
        question = str(sample.get("question") or "").strip()
        answer = sample.get("answer")
        if not question or answer is None:
            continue
        gold = str(answer).strip()
        if not gold:
            continue
        question_id = str(sample.get("question_id") or f"{key}_{idx}").strip()
        conversation = tuple(flatten_haystack_sessions(sample))
        if not conversation:
            continue
        category = _category_for_question(question_id=question_id, question_type=question_type)
        evidence = tuple(str(item) for item in (sample.get("answer_session_ids") or []))
        examples.append(
            LocomoExample(
                task_id=f"LONGMEMEVAL::{key}::{question_id}",
                sample_id=question_id,
                question=question,
                answer=gold,
                category=category,
                evidence=evidence,
                conversation=conversation,
                metadata={
                    "benchmark": "longmemeval",
                    "variant": key,
                    "question_id": question_id,
                    "question_type": question_type,
                    "question_date": str(sample.get("question_date") or ""),
                    "abstention": question_id.endswith("_abs"),
                },
            )
        )
        if limit and len(examples) >= limit:
            break
    return examples


def flatten_haystack_sessions(sample: Mapping[str, Any]) -> list[ConversationTurn]:
    """Flatten LongMemEval haystack sessions into ordered conversation turns."""

    session_ids = list(sample.get("haystack_session_ids") or [])
    session_dates = list(sample.get("haystack_dates") or [])
    sessions = list(sample.get("haystack_sessions") or [])
    turns: list[ConversationTurn] = []
    for session_index, raw_session in enumerate(sessions):
        if not isinstance(raw_session, list):
            continue
        session_id = (
            str(session_ids[session_index])
            if session_index < len(session_ids)
            else f"session_{session_index + 1}"
        )
        session_date = (
            str(session_dates[session_index])
            if session_index < len(session_dates)
            else str(sample.get("question_date") or "")
        )
        for turn_index, raw_turn in enumerate(raw_session):
            if not isinstance(raw_turn, dict):
                continue
            text = str(
                raw_turn.get("content")
                or raw_turn.get("message")
                or raw_turn.get("text")
                or ""
            ).strip()
            if not text:
                continue
            role = str(raw_turn.get("role") or raw_turn.get("speaker") or "unknown").strip()
            turns.append(
                ConversationTurn(
                    session=session_id,
                    session_date=session_date,
                    dia_id=f"{session_id}:turn_{turn_index}",
                    speaker=role,
                    text=text,
                    global_index=len(turns),
                )
            )
    return turns


def build_splits(
    examples: list[LocomoExample],
    *,
    variant: str = "s",
    warmup_size: int = 0,
    train_size: int = DEFAULT_LONGMEMEVAL_TRAIN_SIZE,
    seed: int = 13,
) -> dict[str, Any]:
    """Build deterministic LongMemEval warmup/train/test splits."""

    task_ids = [example.task_id for example in examples]
    shuffled = list(task_ids)
    random.Random(seed).shuffle(shuffled)
    warmup = shuffled[:warmup_size]
    train = shuffled[warmup_size : warmup_size + train_size]
    used = set(warmup) | set(train)
    test = [task_id for task_id in shuffled if task_id not in used]
    return {
        "benchmark": "longmemeval",
        "variant": _normalize_variant(variant),
        "seed": seed,
        "total": len(task_ids),
        "splits": {
            "warmup": warmup,
            "train": train,
            "test": test,
        },
    }


def select_split(
    examples: list[LocomoExample],
    *,
    split: str,
    variant: str = "s",
    split_path: Path | None = None,
) -> list[LocomoExample]:
    """Select LongMemEval examples by the saved split file."""

    key = _normalize_variant(variant)
    split_path = split_path or default_split_path(key)
    if not split_path.exists():
        payload = build_splits(examples, variant=key)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    ids = payload["splits"][split]
    by_id = {example.task_id: example for example in examples}
    return [by_id[item] for item in ids if item in by_id]


def run_longmemeval_frontier(
    *,
    split: str = "train",
    limit: int = 0,
    out_dir: Path,
    variant: str = "s",
    data_path: Path | None = None,
    split_path: Path | None = None,
    question_types: Iterable[str] | None = None,
    scaffolds: Iterable[str] | None = DEFAULT_LONGMEMEVAL_SCAFFOLDS,
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
    judge_model: str = DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
    judge_base_url: str = DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL,
    judge_api_key: str | None = None,
    judge_timeout_s: int = 300,
    use_llm_judge: bool = True,
) -> dict[str, object]:
    """Evaluate memory scaffolds on LongMemEval and write summary/frontier."""

    key = _normalize_variant(variant)
    resolved_data_path = data_path or default_data_path(key)
    if not resolved_data_path.exists():
        prepare_longmemeval(variant=key, dest=resolved_data_path)
    all_examples = load_longmemeval_examples(
        data_path=data_path,
        variant=key,
        question_types=question_types,
    )
    examples = select_split(all_examples, split=split, variant=key, split_path=split_path)
    if limit:
        examples = examples[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    judge = None
    if use_llm_judge and not dry_run:
        env_key_order = (
            ("DEEPSEEK_API_KEY", DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV)
            if "api.deepseek.com" in judge_base_url
            else (DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV, "DEEPSEEK_API_KEY")
        )
        resolved_judge_api_key = judge_api_key or next(
            (os.environ.get(env_key, "") for env_key in env_key_order if os.environ.get(env_key, "")),
            "",
        )
        if not resolved_judge_api_key:
            raise ValueError(
                "LongMemEval LLM-as-judge requires a judge API key. "
                f"Set {DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV}, DEEPSEEK_API_KEY, or pass judge_api_key."
            )
        judge = LongMemEvalJudge(
            model=judge_model,
            base_url=judge_base_url,
            api_key=resolved_judge_api_key,
            timeout_s=judge_timeout_s,
        )
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
        score_run=judge.score_run if judge is not None else _fallback_score_run,
    )
    selected_scaffolds = list(scaffolds or DEFAULT_LONGMEMEVAL_SCAFFOLDS)
    selected_top_k = None if top_k_variants is None else [int(item) for item in top_k_variants]
    scoring_method = "longmemeval_llm_judge" if use_llm_judge and not dry_run else "token_f1"
    extras = _with_longmemeval_extra(
        scaffold_extra,
        scaffolds=selected_scaffolds,
        scoring_method=scoring_method,
        judge_model=judge_model if scoring_method == "longmemeval_llm_judge" else None,
    )
    grid = make_initial_candidate_grid(
        scaffolds=selected_scaffolds,
        top_k_variants=selected_top_k,
        scaffold_extra=extras,
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
        "benchmark": "longmemeval",
        "variant": key,
        "split": split,
        "limit": limit,
        "count": len(examples),
        "question_types": list(question_types or ()),
        "dry_run": dry_run,
        "scoring_method": (
            "longmemeval_llm_judge"
            if judge is not None
            else "token_f1"
        ),
        "judge": (
            {
                "model": judge_model,
                "base_url": judge_base_url,
                "api_key_env": DEFAULT_LONGMEMEVAL_JUDGE_API_KEY_ENV,
            }
            if judge is not None
            else None
        ),
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


def _with_longmemeval_extra(
    scaffold_extra: Mapping[str, Mapping[str, object]] | None,
    *,
    scaffolds: Iterable[str] = DEFAULT_LONGMEMEVAL_SCAFFOLDS,
    scoring_method: str = "longmemeval_llm_judge",
    judge_model: str | None = DEFAULT_LONGMEMEVAL_JUDGE_MODEL,
) -> dict[str, dict[str, object]]:
    out = {str(name): dict(extra) for name, extra in (scaffold_extra or {}).items()}
    for name in scaffolds:
        extra = out.setdefault(name, {})
        extra.setdefault("benchmark", "longmemeval")
        extra.setdefault("source_family", "memgpt")
        extra.setdefault("scoring_method", scoring_method)
        if judge_model:
            extra.setdefault("judge_model", judge_model)
    return out


@dataclass(frozen=True)
class LongMemEvalJudge:
    """Official-style LongMemEval yes/no LLM judge."""

    model: str = DEFAULT_LONGMEMEVAL_JUDGE_MODEL
    base_url: str = DEFAULT_LONGMEMEVAL_JUDGE_BASE_URL
    api_key: str = ""
    timeout_s: int = 300

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_client",
            LocalModelClient(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout_s=self.timeout_s,
                chat_template_kwargs={},
            ),
        )

    def score_run(self, example: LocomoExample, run: ScaffoldRun) -> ScoreResult:
        prompt = build_judge_prompt(
            question_type=str(example.metadata.get("question_type") or ""),
            question=example.question,
            answer=example.answer,
            response=run.prediction,
            abstention=bool(example.metadata.get("abstention")),
        )
        response = self._client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        label = "yes" in response.content.strip().lower()
        return ScoreResult(
            score=1.0 if label else 0.0,
            passed=label,
            metadata={
                "scoring_method": "longmemeval_llm_judge",
                "judge_model": self.model,
                "judge_base_url": self.base_url,
                "judge_response": response.content,
                "judge_label": label,
                "judge_prompt_tokens": response.prompt_tokens,
                "judge_completion_tokens": response.completion_tokens,
                "question_type": example.metadata.get("question_type"),
                "question_id": example.metadata.get("question_id"),
            },
        )


def build_judge_prompt(
    *,
    question_type: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    """Build LongMemEval's official yes/no answer-checking prompt."""

    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is "
            "given but the asked information is not.\n\n"
            f"Question: {question}\n\n"
            f"Explanation: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
    if question_type == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. "
            "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal "
            "information correctly.\n\n"
            f"Question: {question}\n\n"
            f"Rubric: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if question_type == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. In addition, "
            "do not penalize off-by-one errors for the number of days. If the question asks for "
            "the number of days/weeks/months, etc., and the model makes off-by-one errors, the "
            "model's response is still correct.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    if question_type == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, the "
            "response should be considered as correct as long as the updated answer is the required "
            "answer.\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    return (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no.\n\n"
        f"Question: {question}\n\n"
        f"Correct Answer: {answer}\n\n"
        f"Model Response: {response}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


def _fallback_score_run(example: LocomoExample, run: ScaffoldRun) -> ScoreResult:
    score = score_prediction(run.prediction, example.answer)
    return ScoreResult(
        score=score,
        passed=passed(score),
        metadata={
            "scoring_method": "token_f1",
            "question_type": example.metadata.get("question_type"),
            "question_id": example.metadata.get("question_id"),
        },
    )


def _category_for_question(*, question_id: str, question_type: str) -> int:
    if question_id.endswith("_abs"):
        return 5
    return _QUESTION_TYPE_TO_CATEGORY.get(question_type, 1)


def _download_longmemeval(variant: str, dest: Path) -> None:
    url = LONGMEMEVAL_REMOTE_URLS[_normalize_variant(variant)]
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def _normalize_variant(variant: str) -> str:
    raw = str(variant or "s").strip().lower()
    aliases = {
        "small": "s",
        "longmemeval_s": "s",
        "longmemeval_s_cleaned": "s",
        "medium": "m",
        "longmemeval_m": "m",
        "longmemeval_m_cleaned": "m",
        "longmemeval_oracle": "oracle",
    }
    key = aliases.get(raw, raw)
    if key not in LONGMEMEVAL_REMOTE_URLS:
        choices = ", ".join(sorted(LONGMEMEVAL_REMOTE_URLS))
        raise ValueError(f"unknown LongMemEval variant {variant!r}; expected one of: {choices}")
    return key
