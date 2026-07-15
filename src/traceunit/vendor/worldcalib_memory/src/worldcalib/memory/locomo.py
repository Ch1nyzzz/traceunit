"""LOCOMO data import, cache, and split helpers."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from worldcalib.schemas import ConversationTurn, LocomoExample


LOCOMO_REMOTE_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
ANSWERABLE_CATEGORIES = frozenset({1, 2, 3, 4})
# LoCoMo question-category integers → stable human labels, used as the
# per-category score_breakdown keys (so the calib proposer can predict
# Upside/Downside per category). Category 5 (adversarial/unanswerable) is
# excluded by ANSWERABLE_CATEGORIES.
LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
}


def project_root() -> Path:
    """Return the OptiHarness repository root."""

    return Path(__file__).resolve().parents[3]


def default_data_path() -> Path:
    return project_root() / "data" / "locomo" / "locomo10.json"


def default_split_path() -> Path:
    return project_root() / "data" / "locomo" / "splits.json"


def prepare_locomo(
    *,
    dest: Path | None = None,
    source: Path | None = None,
    allow_download: bool = False,
    warmup_size: int = 0,
    train_size: int = 80,
    train_sample_id: str | None = "auto",
    seed: int = 13,
) -> dict[str, Any]:
    """Materialize LOCOMO under OptiHarness and write deterministic splits."""

    dest = dest or default_data_path()
    source = source or _local_skillevolve_cache()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source is not None and source.exists():
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    elif not dest.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"{dest} is missing and no local cache was found. "
                "Pass --allow-download to fetch LOCOMO."
            )
        _download_locomo(dest)

    examples = load_locomo_examples(data_path=dest)
    split_payload = build_splits(
        examples,
        warmup_size=warmup_size,
        train_size=train_size,
        train_sample_id=train_sample_id,
        seed=seed,
    )
    split_path = default_split_path()
    split_path.write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "data_path": str(dest),
        "split_path": str(split_path),
        "count": len(examples),
        "warmup_size": len(split_payload["splits"]["warmup"]),
        "train_size": len(split_payload["splits"]["train"]),
        "test_size": len(split_payload["splits"]["test"]),
        "train_sample_id": split_payload.get("train_sample_id"),
    }


def load_locomo_examples(
    *,
    data_path: Path | None = None,
    limit: int = 0,
    categories: frozenset[int] = ANSWERABLE_CATEGORIES,
) -> list[LocomoExample]:
    """Load answerable LOCOMO QA examples."""

    path = data_path or default_data_path()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected top-level list in {path}")

    examples: list[LocomoExample] = []
    seen: set[tuple[str, str, tuple[str, ...], int]] = set()
    for sample in data:
        sample_id = str(sample.get("sample_id") or f"sample{len(examples)}")
        conversation = tuple(flatten_conversation(sample.get("conversation") or {}))
        if not conversation:
            continue
        for qa_index, qa in enumerate(sample.get("qa") or []):
            if limit and len(examples) >= limit:
                return examples
            try:
                category = int(qa.get("category", -1))
            except (TypeError, ValueError):
                continue
            if category not in categories:
                continue
            question = str(qa.get("question", "")).strip()
            answer = qa.get("answer")
            if not question or answer is None:
                continue
            gold = str(answer).strip()
            if not gold:
                continue
            evidence = tuple(str(item) for item in (qa.get("evidence") or []))
            key = (_norm(question), _norm(gold), evidence, category)
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                LocomoExample(
                    task_id=f"LOCOMO::{sample_id}::qa::{qa_index}",
                    sample_id=sample_id,
                    question=question,
                    answer=gold,
                    category=category,
                    evidence=evidence,
                    conversation=conversation,
                    metadata={
                        "question_type": LOCOMO_CATEGORY_NAMES.get(
                            category, f"category-{category}"
                        ),
                        "category": category,
                    },
                )
            )
    return examples


def flatten_conversation(conversation: dict[str, Any]) -> list[ConversationTurn]:
    """Flatten one raw LOCOMO conversation dict into ordered turns."""

    session_nums: list[int] = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                session_nums.append(int(key.split("_", 1)[1]))
            except ValueError:
                continue
    session_nums.sort()

    turns: list[ConversationTurn] = []
    for num in session_nums:
        session = f"session_{num}"
        date = str(conversation.get(f"{session}_date_time", "") or "")
        for raw in conversation.get(session, []) or []:
            text = str(raw.get("text", "") or "")
            caption = str(raw.get("blip_caption", "") or "")
            if caption and raw.get("img_url"):
                image_text = f"[Image: {caption}]"
                text = f"{image_text} {text}" if text else image_text
            if not text:
                continue
            turns.append(
                ConversationTurn(
                    session=session,
                    session_date=date,
                    dia_id=str(raw.get("dia_id", "") or ""),
                    speaker=str(raw.get("speaker", "") or ""),
                    text=text,
                    global_index=len(turns),
                )
            )
    return turns


def build_splits(
    examples: list[LocomoExample],
    *,
    warmup_size: int = 0,
    train_size: int = 80,
    train_sample_id: str | None = "auto",
    seed: int = 13,
) -> dict[str, Any]:
    """Build deterministic warmup/train/test splits."""

    task_ids = [example.task_id for example in examples]
    task_to_sample = {example.task_id: example.sample_id for example in examples}
    shuffled = list(task_ids)
    random.Random(seed).shuffle(shuffled)
    warmup = shuffled[:warmup_size]
    selected_train_sample = _resolve_train_sample_id(
        examples,
        train_sample_id=train_sample_id,
        excluded_task_ids=set(warmup),
        train_size=train_size,
    )
    if selected_train_sample is None:
        train = shuffled[warmup_size : warmup_size + train_size]
    else:
        warmup_ids = set(warmup)
        train = [
            task_id
            for task_id in shuffled
            if task_to_sample[task_id] == selected_train_sample and task_id not in warmup_ids
        ][:train_size]
    used = set(warmup) | set(train)
    test = [task_id for task_id in shuffled if task_id not in used]
    return {
        "benchmark": "locomo",
        "seed": seed,
        "total": len(task_ids),
        "train_sample_id": selected_train_sample,
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
    split_path: Path | None = None,
) -> list[LocomoExample]:
    """Select examples by the saved split file."""

    split_path = split_path or default_split_path()
    if not split_path.exists():
        prepare_locomo()
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    ids = payload["splits"][split]
    by_id = {example.task_id: example for example in examples}
    return [by_id[item] for item in ids if item in by_id]


def _resolve_train_sample_id(
    examples: list[LocomoExample],
    *,
    train_sample_id: str | None,
    excluded_task_ids: set[str],
    train_size: int,
) -> str | None:
    if not train_sample_id:
        return None

    available = Counter(
        example.sample_id
        for example in examples
        if example.task_id not in excluded_task_ids
    )
    if train_sample_id == "auto":
        enough = [
            (count, sample_id)
            for sample_id, count in available.items()
            if count >= train_size
        ]
        if not enough:
            raise ValueError(f"no LOCOMO sample has {train_size} train examples after warmup")
        return max(enough, key=lambda item: (item[0], item[1]))[1]

    if available[train_sample_id] < train_size:
        raise ValueError(
            f"LOCOMO sample {train_sample_id!r} has only {available[train_sample_id]} "
            f"available train examples after warmup; need {train_size}"
        )
    return train_sample_id


def _local_skillevolve_cache() -> Path | None:
    candidates = [
        Path.home() / ".cache" / "skillevolve" / "locomo10.json",
        Path("/data/home/yuhan/.cache/skillevolve/locomo10.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _download_locomo(dest: Path) -> None:
    with httpx.stream("GET", LOCOMO_REMOTE_URL, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())
