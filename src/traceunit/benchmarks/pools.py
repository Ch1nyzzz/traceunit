from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from traceunit.io import read_json, sha256_file, write_json
from traceunit.models import BenchmarkPlan, PoolRole, PoolSliceRef
from traceunit.ontology import ontology_ref

T = TypeVar("T")


def freeze_benchmark_plan(
    *,
    root: Path,
    benchmark: str,
    search_items: Sequence[Any],
    final_items: Sequence[Any],
    cluster_key: Callable[[Any], str],
) -> BenchmarkPlan:
    """Persist immutable pool files and bind every slice to its content hash."""

    root.mkdir(parents=True, exist_ok=True)

    def freeze(
        *,
        slice_id: str,
        role: PoolRole,
        items: Sequence[Any],
    ) -> PoolSliceRef:
        path = root / f"{slice_id}.json"
        path.write_text(
            json.dumps(list(items), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return PoolSliceRef(
            slice_id=slice_id,
            role=role,
            manifest_path=str(path.resolve()),
            manifest_sha256=sha256_file(path),
            cluster_ids=tuple(dict.fromkeys(cluster_key(item) for item in items)),
        )

    search = freeze(slice_id="search", role=PoolRole.SEARCH, items=search_items)
    final = freeze(slice_id="final", role=PoolRole.FINAL, items=final_items)
    identity = {
        "benchmark": benchmark,
        "search": pool_identity(search),
        "final": pool_identity(final),
        "ontology": ontology_ref(),
    }
    plan_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = BenchmarkPlan(
        benchmark=benchmark,
        search=search,
        final=final,
        plan_sha256=plan_sha256,
        ontology=ontology_ref(),
    )
    write_json(root / "plan.json", plan.to_dict())
    return plan


def load_pool_items(pool: PoolSliceRef) -> list[Any]:
    path = Path(pool.manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"pool manifest is missing: {path}")
    actual = sha256_file(path)
    if actual != pool.manifest_sha256:
        raise RuntimeError(
            f"pool manifest hash mismatch for {pool.slice_id}: "
            f"expected {pool.manifest_sha256}, got {actual}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"pool manifest must contain a list: {path}")
    return raw


def load_benchmark_plan(path: Path) -> BenchmarkPlan:
    return BenchmarkPlan.from_dict(read_json(path))


def take_cluster_groups(
    items: Sequence[T],
    limit: int,
    *,
    cluster_key: Callable[[T], str],
) -> list[T]:
    """Take whole clusters in first-appearance order until the limit is reached."""

    if limit <= 0 or len(items) <= limit:
        return list(items)
    groups: dict[str, list[T]] = {}
    for item in items:
        groups.setdefault(cluster_key(item), []).append(item)
    selected: list[T] = []
    for group in groups.values():
        if selected and len(selected) + len(group) > limit:
            break
        selected.extend(group)
    return selected


def pool_identity(pool: PoolSliceRef) -> dict[str, Any]:
    """Portable slice identity shared by plan hashing and evaluation fingerprints."""

    return {
        "slice_id": pool.slice_id,
        "role": pool.role.value,
        "manifest_sha256": pool.manifest_sha256,
        "cluster_ids": pool.cluster_ids,
    }
