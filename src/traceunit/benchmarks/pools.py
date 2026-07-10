from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from traceunit.io import read_json, sha256_file, write_json
from traceunit.models import BenchmarkPlan, PoolRole, PoolSliceRef

T = TypeVar("T")


def partition_by_cluster(
    items: Sequence[T],
    *,
    cluster_key: Callable[[T], str],
    shard_size: int,
) -> list[list[T]]:
    """Partition ordered items without splitting a correlated cluster."""

    if not items:
        return []
    if shard_size <= 0:
        return [list(items)]
    groups: list[list[T]] = []
    positions: dict[str, int] = {}
    for item in items:
        key = cluster_key(item)
        if key not in positions:
            positions[key] = len(groups)
            groups.append([])
        groups[positions[key]].append(item)
    shards: list[list[T]] = []
    current: list[T] = []
    for group in groups:
        if current and len(current) + len(group) > shard_size:
            shards.append(current)
            current = []
        current.extend(group)
    if current:
        shards.append(current)
    return shards


def freeze_benchmark_plan(
    *,
    root: Path,
    benchmark: str,
    search_items: Sequence[Any],
    calibration_shards: Sequence[Sequence[Any]],
    final_items: Sequence[Any],
    cluster_key: Callable[[Any], str],
) -> BenchmarkPlan:
    """Persist immutable pool files and bind every slice to its content hash."""

    root.mkdir(parents=True, exist_ok=True)

    def freeze(
        *,
        slice_id: str,
        role: PoolRole,
        ordinal: int,
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
            ordinal=ordinal,
        )

    search = freeze(
        slice_id="search",
        role=PoolRole.SEARCH,
        ordinal=0,
        items=search_items,
    )
    calibration = tuple(
        freeze(
            slice_id=f"calibration_{index:03d}",
            role=PoolRole.CALIBRATION,
            ordinal=index,
            items=items,
        )
        for index, items in enumerate(calibration_shards)
    )
    final = freeze(
        slice_id="final",
        role=PoolRole.FINAL,
        ordinal=0,
        items=final_items,
    )
    identity = {
        "benchmark": benchmark,
        "search": _portable_identity(search),
        "calibration": [_portable_identity(item) for item in calibration],
        "final": _portable_identity(final),
    }
    plan_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = BenchmarkPlan(
        benchmark=benchmark,
        search=search,
        calibration=calibration,
        final=final,
        plan_sha256=plan_sha256,
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


def _portable_identity(pool: PoolSliceRef) -> dict[str, Any]:
    return {
        "slice_id": pool.slice_id,
        "role": pool.role.value,
        "manifest_sha256": pool.manifest_sha256,
        "cluster_ids": pool.cluster_ids,
        "ordinal": pool.ordinal,
    }
