"""Humanity's Last Exam (HLE) adapter -- native, self-contained QA runner.

HLE is a text-only expert QA benchmark. Unlike the Harbor/Docker benchmarks it
runs entirely in-process under host control: the editable ``hle_qa`` scaffold
builds prompts and extracts answers, a frozen solver model answers, and a hidden
LLM judge grades each answer against the gold. The gold answer is held only by
the host and is never written into a pool manifest or handed to the scaffold, so
a candidate can never read or encode it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import traceunit
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.common import load_cached_evaluation
from traceunit.benchmarks.native_eval import normalize_trial_results
from traceunit.benchmarks.openai_chat import ChatError, chat_completion
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


def _packaged_baseline_source() -> Path:
    return (Path(traceunit.__file__).parent / "scaffolds" / "hle_baseline").resolve()


class HLEAdapter(BenchmarkAdapter):
    name = "hle"
    supports_agent_probe = False

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._plan: BenchmarkPlan | None = None
        self._gold_path: Path | None = None

    # -- preparation ---------------------------------------------------------

    def prepare(self, work_dir: Path) -> BenchmarkPlan:
        if not self.baseline_source().is_dir():
            raise FileNotFoundError(
                f"editable HLE scaffold is missing: {self.baseline_source()}"
            )
        pool_dir = work_dir / "benchmark_data" / "hle"
        pool_dir.mkdir(parents=True, exist_ok=True)
        self._gold_path = pool_dir / "gold.jsonl"

        frozen_plan = pool_dir / "plan.json"
        if frozen_plan.is_file() and self._gold_path.is_file():
            plan = load_benchmark_plan(frozen_plan)
            self.bind_plan(plan)
            for pool in (plan.search, plan.final):
                load_pool_items(pool)
            return plan

        rows = self._load_rows()
        search_rows, final_rows = _split_rows(
            rows,
            seed=self.config.benchmark_seed,
            search_fraction=self.config.search_fraction,
            cluster_key=_cluster_key,
        )
        search_rows = _seeded_order(search_rows, seed=self.config.benchmark_seed, ns="search")
        final_rows = _seeded_order(final_rows, seed=self.config.benchmark_seed, ns="final")
        if self.config.search_limit > 0:
            search_rows = search_rows[: self.config.search_limit]
        if self.config.final_limit > 0:
            final_rows = final_rows[: self.config.final_limit]
        if not search_rows or not final_rows:
            raise ValueError("HLE search and final pools must be non-empty")

        self._write_gold(self._gold_path, search_rows + final_rows)
        self._plan = freeze_benchmark_plan(
            root=pool_dir,
            benchmark=self.name,
            search_items=[_public_row(row, "search") for row in search_rows],
            final_items=[_public_row(row, "final") for row in final_rows],
            cluster_key=lambda item: str(item["cluster"]),
        )
        return self._plan

    def preflight(self) -> None:
        if not self.config.dry_run and not os.environ.get(self.config.api_key_env):
            raise RuntimeError(f"HLE solver key is missing: {self.config.api_key_env}")
        if (
            not self.config.dry_run
            and self.config.use_llm_judge
            and not os.environ.get(self.config.judge_api_key_env)
        ):
            raise RuntimeError(
                f"HLE judge key is missing: {self.config.judge_api_key_env}"
            )

    def baseline_source(self) -> Path:
        return (
            self.config.scaffold_source_path or _packaged_baseline_source()
        ).resolve()

    def context(self) -> str:
        return """Humanity's Last Exam (text-only). The editable artifact is the hle_qa scaffold:
the system prompt, answer-format instruction, answer extraction, and answering strategy
(single pass or self-consistency voting). The solver model is frozen. Never open the HLE
dataset, load gold answers, import the datasets library, or encode question-specific answers;
the scaffold only ever receives the question text. Grading is done by a sealed host LLM judge."""

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
        if self._gold_path is None or not self._gold_path.is_file():
            raise RuntimeError("gold answers were not materialized during prepare()")

        items = load_pool_items(pool)
        source = source.resolve()
        source_hash = sha256_tree(source / "hle_qa")
        cache_fingerprint, cache_payload = self._cache_fingerprint(
            source_hash=source_hash, pool=pool, item_count=len(items)
        )
        cached = load_cached_evaluation(out_dir)
        if (
            cached is not None
            and not self.config.force
            and cached.metadata.get("cache_fingerprint") == cache_fingerprint
        ):
            return cached

        out_dir.mkdir(parents=True, exist_ok=True)
        questions_path = out_dir / "questions.json"
        # Only public fields reach the worker: id, question, answer_type.
        public_questions = [
            {
                "id": item["id"],
                "question": item["question"],
                "answer_type": item.get("answer_type") or "exactMatch",
            }
            for item in items
        ]
        write_json(questions_path, public_questions)

        spec_path = out_dir / "hle_spec.json"
        predictions_path = out_dir / "hle_predictions.json"
        write_json(
            spec_path,
            {
                "source": str(source),
                "questions_path": str(questions_path),
                "dry_run": self.config.dry_run,
                "model": {
                    "model": self.config.model,
                    "base_url": self.config.base_url,
                    "api_key_env": self.config.api_key_env,
                    "max_output_tokens": self.config.hle_max_output_tokens,
                    "timeout_s": self.config.timeout_s,
                },
            },
        )
        proc = subprocess.run(
            [
                str(self.config.unit_python or Path(sys.executable)),
                "-m",
                "traceunit.benchmarks.hle_worker",
                "run",
                "--spec",
                str(spec_path),
                "--out",
                str(predictions_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (out_dir / "hle_worker.log").write_text(proc.stdout, encoding="utf-8")
        if not predictions_path.is_file():
            raise RuntimeError(
                "HLE worker produced no predictions; see "
                f"{out_dir / 'hle_worker.log'} (tail: {proc.stdout[-2000:]})"
            )

        predictions = {
            str(record.get("id")): record
            for record in json.loads(predictions_path.read_text(encoding="utf-8")).get(
                "predictions", []
            )
        }
        gold = self._load_gold()
        result_path = out_dir / "hle_result.json"
        judge_tokens = self._grade(
            items=items,
            predictions=predictions,
            gold=gold,
            candidate_id=candidate_id,
            result_path=result_path,
        )

        evaluation = normalize_trial_results(
            result_path=result_path,
            benchmark=self.name,
            split=pool.slice_id,
            candidate_id=candidate_id,
            out_dir=out_dir,
        )
        evaluation = replace(
            evaluation,
            metadata={
                **evaluation.metadata,
                "cache_fingerprint": cache_fingerprint,
                "cache_identity": dict(cache_payload),
                "judge_tokens": judge_tokens,
                "judge_model": self.config.judge_model if self.config.use_llm_judge else None,
            },
        )
        write_json(out_dir / "evaluation.json", evaluation.to_dict())
        return evaluation

    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = source / "hle_qa"
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
            "datasets": "candidate imports the datasets library to open HLE directly",
            "load_dataset": "candidate loads the HLE dataset directly",
            "cais/hle": "candidate opens the HLE dataset",
            "gold": "candidate references gold answers",
            "gold.jsonl": "candidate reads the host-only gold file",
            "answer_key": "candidate references an answer key",
        }
        return [message for token, message in banned.items() if token in added]

    # -- grading -------------------------------------------------------------

    def _grade(
        self,
        *,
        items: list[Mapping[str, Any]],
        predictions: Mapping[str, Mapping[str, Any]],
        gold: Mapping[str, Mapping[str, Any]],
        candidate_id: str,
        result_path: Path,
    ) -> int:
        tasks: list[dict[str, Any]] = []
        passed_count = 0
        score_sum = 0.0
        total_tokens = 0
        judge_tokens_total = 0
        for item in items:
            qid = str(item["id"])
            record = predictions.get(qid, {})
            prediction = str(record.get("prediction") or "")
            prompt_tokens = int(record.get("prompt_tokens") or 0)
            completion_tokens = int(record.get("completion_tokens") or 0)
            task_tokens = prompt_tokens + completion_tokens
            total_tokens += task_tokens
            error = str(record.get("error") or "")
            gold_row = gold.get(qid, {})
            gold_answer = str(gold_row.get("answer") or "")

            if error:
                correct, status, judge_tokens = False, "solver_error", 0
            elif self.config.dry_run:
                correct, status, judge_tokens = False, "dry_run", 0
            elif not prediction:
                correct, status, judge_tokens = False, "empty_prediction", 0
            else:
                correct, judge_tokens = self._judge(
                    question=str(item.get("question") or ""),
                    gold_answer=gold_answer,
                    prediction=prediction,
                    answer_type=str(item.get("answer_type") or "exactMatch"),
                )
                status = "correct" if correct else "incorrect"
            judge_tokens_total += judge_tokens
            score = 1.0 if correct else 0.0
            passed_count += 1 if correct else 0
            score_sum += score
            tasks.append(
                {
                    "task_id": qid,
                    "score": score,
                    "passed": correct,
                    "status": status,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": task_tokens,
                    "question": str(item.get("question") or ""),
                    "prediction": prediction,
                    "error": error,
                }
            )

        count = len(tasks)
        envelope = {
            "candidate": {
                "candidate_id": candidate_id,
                "count": count,
                "passrate": (passed_count / count) if count else 0.0,
                "average_score": (score_sum / count) if count else 0.0,
                "token_consuming": total_tokens,
                "config": {"model": self.config.model, "judge_model": self.config.judge_model},
            },
            "tasks": tasks,
            "job_stats": {"n_trials": count, "n_errors": 0},
        }
        write_json(result_path, envelope)
        return judge_tokens_total

    def _judge(
        self, *, question: str, gold_answer: str, prediction: str, answer_type: str
    ) -> tuple[bool, int]:
        """Grade one answer. Exact-match short-circuits; otherwise the LLM judge."""

        if _normalize(prediction) and _normalize(prediction) == _normalize(gold_answer):
            return True, 0
        if not self.config.use_llm_judge:
            return False, 0
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict grader. Decide whether the response's final "
                    "answer matches the gold answer. Respond with a JSON object "
                    '{"correct": true|false, "confidence": 0-100} and nothing else.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Gold answer:\n{gold_answer}\n\n"
                    f"Response's answer:\n{prediction}\n\n"
                    "Is the response's answer correct?"
                ),
            },
        ]
        try:
            response = chat_completion(
                base_url=self.config.judge_base_url,
                api_key=os.environ.get(self.config.judge_api_key_env, ""),
                model=self.config.judge_model,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
                timeout_s=self.config.judge_timeout_s,
            )
        except ChatError:
            return False, 0
        tokens = int(response.get("prompt_tokens") or 0) + int(
            response.get("completion_tokens") or 0
        )
        return _parse_judge_verdict(str(response.get("content") or "")), tokens

    # -- data ----------------------------------------------------------------

    def _load_rows(self) -> list[dict[str, Any]]:
        # A local host-only copy (data_path) wins over the gated Hub dataset. This
        # lets HLE run offline or against a custom HLE-shaped .jsonl/.json when
        # cais/hle access has not been granted.
        if self.config.data_path is not None:
            source: Any = _read_local_rows(self.config.data_path)
        else:
            try:
                from datasets import load_dataset  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "HLE requires the datasets library; install traceunit[hle]."
                ) from exc
            # cais/hle is gated; token=True uses the locally stored HF login token.
            source = load_dataset(
                self.config.hle_dataset_name,
                split=self.config.hle_dataset_split,
                token=True,
            )
        categories = {c.lower() for c in self.config.hle_categories}
        rows: list[dict[str, Any]] = []
        for raw in source:
            if self.config.hle_text_only and raw.get("image"):
                continue
            category = str(raw.get("category") or "")
            raw_subject = str(raw.get("raw_subject") or "")
            if categories and not (
                category.lower() in categories or raw_subject.lower() in categories
            ):
                continue
            rows.append(
                {
                    "id": str(raw.get("id") or ""),
                    "question": str(raw.get("question") or ""),
                    "answer": str(raw.get("answer") or ""),
                    "answer_type": str(raw.get("answer_type") or "exactMatch"),
                    "category": category,
                    "raw_subject": raw_subject,
                }
            )
        if not rows:
            raise ValueError("no HLE rows matched the configured filters")
        return rows

    def _write_gold(self, path: Path, rows: list[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "answer": row["answer"],
                            "answer_type": row["answer_type"],
                        },
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")

    def _load_gold(self) -> dict[str, dict[str, Any]]:
        assert self._gold_path is not None
        gold: dict[str, dict[str, Any]] = {}
        for line in self._gold_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gold[str(row["id"])] = row
        return gold

    def _cache_fingerprint(
        self, *, source_hash: str, pool: PoolSliceRef, item_count: int
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "adapter_cache_version": ADAPTER_CACHE_VERSION,
            "source_sha256": source_hash,
            "pool": pool_identity(pool),
            "item_count": item_count,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "max_output_tokens": self.config.hle_max_output_tokens,
            "use_llm_judge": self.config.use_llm_judge,
            "judge_model": self.config.judge_model,
            "dry_run": self.config.dry_run,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest(), payload


def _read_local_rows(path: Path) -> list[dict[str, Any]]:
    """Load HLE-shaped rows from a local .jsonl or .json file."""

    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"HLE data_path is not a readable file: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("rows", "data", "test"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    raise ValueError(f"unsupported HLE data shape: {path}")


def _public_row(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Pool-manifest row: question is public, the gold answer is excluded."""

    return {
        "id": row["id"],
        "question": row["question"],
        "answer_type": row["answer_type"],
        "category": row["category"],
        "raw_subject": row["raw_subject"],
        "cluster": _cluster_key(row),
        "split": split,
    }


def _cluster_key(row: Mapping[str, Any]) -> str:
    return str(row.get("category") or row.get("raw_subject") or "unknown")


def _split_rows(rows, *, seed, search_fraction, cluster_key):
    if not 0 < search_fraction < 1:
        raise ValueError("search_fraction must be between 0 and 1")
    search: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    for row in rows:
        cluster = cluster_key(row)
        value = (
            int.from_bytes(
                hashlib.sha256(f"{seed}:{cluster}".encode()).digest()[:8], "big"
            )
            / 2**64
        )
        (search if value < search_fraction else final).append(row)
    if rows and (not search or not final):
        ordered = _seeded_order(list(rows), seed=seed, ns="fallback")
        cut = max(1, min(len(ordered) - 1, round(len(ordered) * search_fraction)))
        search, final = ordered[:cut], ordered[cut:]
    return search, final


def _seeded_order(rows, *, seed, ns):
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{ns}:{row['id']}".encode()
        ).hexdigest(),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(" .")


def _parse_judge_verdict(content: str) -> bool:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, Mapping) and "correct" in payload:
                return bool(payload["correct"])
        except json.JSONDecodeError:
            pass
    lowered = content.lower()
    if re.search(r"\b(correct|yes|true)\b", lowered) and not re.search(
        r"\b(incorrect|not correct|no|false)\b", lowered
    ):
        return True
    return False
