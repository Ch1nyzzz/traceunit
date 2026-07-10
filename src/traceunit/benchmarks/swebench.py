from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

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
    partition_by_cluster,
)
from traceunit.config import BenchmarkConfig
from traceunit.io import sha256_file, sha256_tree, write_json
from traceunit.models import BenchmarkEvaluation, BenchmarkPlan, PoolSliceRef


LOCAL_VERIFIED_DATA = Path(
    "/data/home/yuhan/Optimizer1/data/swebench_verified_all500_test.json"
)

ADAPTER_CACHE_VERSION = 4
SWEBENCH_HARNESS_SPEC = "swebench==4.1.0"
_SAFE_TRACE_METRIC_KEYS = {
    "repo",
    "base_commit",
    "duration_s",
    "returncode",
    "evaluator_returncode",
    "exit_status",
    "timed_out",
    "timeout_s",
    "patch_bytes",
    "error_tail",
}


class SwebenchVerifiedAdapter(BenchmarkAdapter):
    name = "swebench_verified"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._plan: BenchmarkPlan | None = None

    def prepare(self, work_dir: Path) -> BenchmarkPlan:
        root = self.config.worldcalib_root
        if not (root / "src/worldcalib/coding/swebench.py").is_file():
            raise FileNotFoundError(
                f"WorldCalib SWE-bench runner is missing under {root}"
            )
        if not self.baseline_source().is_dir():
            raise FileNotFoundError(
                f"mini-SWE-agent source is missing: {self.baseline_source()}"
            )
        pool_dir = work_dir / "benchmark_data" / "swebench_verified"
        pool_dir.mkdir(parents=True, exist_ok=True)
        frozen_plan = pool_dir / "plan.json"
        if frozen_plan.is_file():
            plan = load_benchmark_plan(frozen_plan)
            self.bind_plan(plan)
            for pool in (plan.search, *plan.calibration, plan.final):
                load_pool_items(pool)
            return plan
        configured = {
            "search": self.config.search_data_path,
            "calibration": self.config.calibration_data_path,
            "final": self.config.final_data_path,
        }
        has_explicit_heldout = any(
            configured[name] is not None for name in ("calibration", "final")
        )
        if has_explicit_heldout:
            missing = [
                name
                for name, path in configured.items()
                if path is None or not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "explicit SWE-bench pools require readable search, calibration, "
                    f"and final files; missing={missing}"
                )
            pools = {
                name: _representative_order(
                    _load_rows(configured_path),
                    seed=self.config.benchmark_seed,
                    namespace=name,
                )
                for name, configured_path in configured.items()
                if configured_path is not None
            }
        else:
            source_path = self.config.search_data_path
            if source_path is None and LOCAL_VERIFIED_DATA.is_file():
                source_path = LOCAL_VERIFIED_DATA
            if source_path is not None and source_path.is_file():
                rows = _load_rows(source_path)
            else:
                rows = _download_verified(
                    dataset_name=self.config.dataset_name,
                    split=self.config.dataset_split,
                )
            pools = _split_rows(
                rows,
                seed=self.config.benchmark_seed,
                search_fraction=self.config.search_fraction,
                calibration_fraction=self.config.calibration_fraction,
            )

        limits = {
            "search": self.config.search_limit,
            "calibration": self.config.calibration_limit,
            "final": self.config.final_limit,
        }
        pools = {
            name: _take_cluster_groups(items, limits[name])
            for name, items in pools.items()
        }
        if not pools["search"] or not pools["final"]:
            raise ValueError("SWE-bench search and final pools must be non-empty")
        if self.config.calibration_fraction > 0 and not pools["calibration"]:
            raise ValueError(
                "SWE-bench calibration pool must be non-empty when calibration is enabled"
            )
        _validate_disjoint_pools(pools)
        calibration_shards = partition_by_cluster(
            pools["calibration"],
            cluster_key=_repo_cluster,
            shard_size=self.config.calibration_shard_size,
        )
        public_search = [_public_row(item, split="search") for item in pools["search"]]
        public_calibration = [
            [_public_row(item, split="calibration") for item in shard]
            for shard in calibration_shards
        ]
        public_final = [_public_row(item, split="final") for item in pools["final"]]
        self._plan = freeze_benchmark_plan(
            root=pool_dir,
            benchmark=self.name,
            search_items=public_search,
            calibration_shards=public_calibration,
            final_items=public_final,
            cluster_key=_repo_cluster,
        )
        return self._plan

    def preflight(self) -> None:
        if self.config.dry_run:
            return
        if self.config.evaluator_command:
            raise RuntimeError(
                "custom SWE evaluator_command is disabled because it cannot guarantee "
                "patch-specific official report identity"
            )
        if self.config.repeats != 1:
            raise RuntimeError(
                "SWE-bench adapter currently supports repeats=1; repeated evaluations "
                "must not be requested silently"
            )
        if not os.environ.get(self.config.api_key_env):
            raise RuntimeError(
                f"SWE-bench target-model key is missing: {self.config.api_key_env}"
            )
        for executable in ("docker", "uvx"):
            if not shutil.which(executable):
                raise RuntimeError(f"SWE-bench runtime is missing: {executable}")

    def baseline_source(self) -> Path:
        return (
            self.config.baseline_source_path
            or self.config.worldcalib_root / "references/vendor/mini-swe-agent"
        ).resolve()

    def context(self) -> str:
        return """SWE-bench Verified. The editable artifact is the mini-SWE-agent control loop,
prompts, tool execution, context policy, and patch submission logic. The solver model is frozen.
Never access or encode gold patches, test patches, evaluator output, instance ids, or
repository-specific solutions. Proxy tests must exercise general agent behavior with mocks,
scripted trajectories, or micro repositories. Natural scoring is performed only by the sealed
official SWE-bench harness."""

    def evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        pool: PoolSliceRef,
        out_dir: Path,
    ) -> BenchmarkEvaluation:
        if self.config.evaluator_command:
            raise RuntimeError(
                "custom SWE evaluator_command is disabled; use the adapter-owned "
                "patch-specific official evaluator"
            )
        if self._plan is None:
            raise RuntimeError("prepare() must be called before evaluate()")
        known = (self._plan.search, *self._plan.calibration, self._plan.final)
        if pool not in known:
            raise ValueError(f"pool is not part of the prepared plan: {pool.slice_id}")
        load_pool_items(pool)
        pool_path = Path(pool.manifest_path)
        source_hash = sha256_tree(source)
        cache_fingerprint, cache_payload = _evaluation_cache_fingerprint(
            source_hash=source_hash,
            pool=pool,
            config=self.config,
        )
        cached = load_cached_evaluation(out_dir)
        if (
            cached is not None
            and not self.config.force
            and cached.metadata.get("cache_fingerprint") == cache_fingerprint
        ):
            return cached
        command = self.config.agent_command or self._default_agent_command()
        attempt_id = f"{candidate_id}-{cache_fingerprint[:16]}"
        eval_command = self._default_eval_command(attempt_id=attempt_id)
        with worldcalib_import(self.config.worldcalib_root):
            from worldcalib.coding.swebench import (  # type: ignore
                DEFAULT_MINI_SWE_AGENT_NAME,
                MiniSweAgentSourceRunner,
                load_swebench_instances,
            )

            instances = load_swebench_instances(pool_path, split=pool.slice_id, limit=0)
            runner = MiniSweAgentSourceRunner(
                instances=instances,
                out_dir=out_dir,
                timeout_s=self.config.timeout_s,
                max_eval_workers=self.config.concurrency,
                dry_run=self.config.dry_run,
                force=self.config.force,
                project_root=self.config.worldcalib_root,
            )
            candidate = {
                "name": DEFAULT_MINI_SWE_AGENT_NAME,
                "agent_name": DEFAULT_MINI_SWE_AGENT_NAME,
                "source_project_path": str(source.resolve()),
                "source_sha256": source_hash,
                "evaluation_fingerprint": cache_fingerprint,
                "command": command,
                "eval_command": eval_command,
            }
            result = runner.evaluate_candidate(
                candidate=candidate,
                candidate_id=candidate_id,
                agent_name=DEFAULT_MINI_SWE_AGENT_NAME,
            )
        evaluation = normalize_worldcalib_result(
            result_path=Path(result.result_path),
            benchmark=self.name,
            split=pool.slice_id,
            candidate_id=candidate_id,
            out_dir=out_dir,
        )
        trajectory_stats = _augment_miniswe_trajectories(Path(evaluation.trace_path))
        evaluation = _finalize_evaluation(
            evaluation,
            trajectory_stats=trajectory_stats,
            cache_fingerprint=cache_fingerprint,
            cache_payload=cache_payload,
        )
        write_json(out_dir / "evaluation.json", evaluation.to_dict())
        return evaluation

    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = source / "src/minisweagent"
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
            "gold_patch": "candidate refers to SWE-bench gold patches",
            "test_patch": "candidate refers to SWE-bench hidden test patches",
            "run_miniswe_swebench_single.py": "candidate refers to the sealed eval entry",
            "swebench.harness": "candidate imports the SWE-bench evaluator",
            "SWE-bench_Verified": "candidate directly opens the benchmark dataset",
            "SWE-Bench_Verified": "candidate directly opens the benchmark dataset",
        }
        return [message for token, message in banned.items() if token in added]

    def _default_agent_command(self) -> str:
        script = self.config.worldcalib_root / "scripts/run_miniswe_swebench_single.py"
        return (
            f"python {script} run --source-path {{source_path}} "
            "--instance-path {instance_path} --patch-path {patch_path} "
            f"--task-dir {{task_dir}} --model openai/{self.config.model} "
            f"--base-url {self.config.base_url} --max-tokens 4096 "
            f"--api-key-env {self.config.api_key_env}"
        )

    def _default_eval_command(self, *, attempt_id: str) -> str:
        python = shlex.quote(str(Path(sys.executable).resolve()))
        script = shlex.quote(str(Path(__file__).resolve()))
        return (
            f"{python} {script} _eval-patch "
            "--instance-path '{instance_path}' --patch-path '{patch_path}' "
            "--task-dir '{task_dir}' "
            f"--attempt-id {shlex.quote(attempt_id)} "
            f"--dataset-name {shlex.quote(self.config.dataset_name)} "
            f"--dataset-split {shlex.quote(self.config.dataset_split)} "
            f"--timeout-s {max(1, self.config.timeout_s - 30)}"
        )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("instances", "tasks", "data"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    raise ValueError(f"unsupported SWE-bench data shape: {path}")


def _download_verified(*, dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SWE-bench Verified data is not configured. Install traceunit[swebench] "
            "or provide benchmark.search_data_path."
        ) from exc
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def _evaluation_cache_fingerprint(
    *,
    source_hash: str,
    pool: PoolSliceRef,
    config: BenchmarkConfig,
) -> tuple[str, dict[str, Any]]:
    worldcalib_runner = config.worldcalib_root / "src/worldcalib/coding/swebench.py"
    mini_entry = config.worldcalib_root / "scripts/run_miniswe_swebench_single.py"
    payload: dict[str, Any] = {
        "adapter_cache_version": ADAPTER_CACHE_VERSION,
        "swebench_harness_spec": SWEBENCH_HARNESS_SPEC,
        "source_sha256": source_hash,
        "pool": {
            "slice_id": pool.slice_id,
            "role": pool.role.value,
            "manifest_sha256": pool.manifest_sha256,
            "cluster_ids": pool.cluster_ids,
            "ordinal": pool.ordinal,
        },
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "timeout_s": config.timeout_s,
        "concurrency": config.concurrency,
        "repeats": config.repeats,
        "max_interactions": config.max_interactions,
        "dry_run": config.dry_run,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "agent_command": config.agent_command,
        "evaluator_command": config.evaluator_command,
        "worldcalib_runner_sha256": (
            sha256_file(worldcalib_runner) if worldcalib_runner.is_file() else "missing"
        ),
        "miniswe_entry_sha256": (
            sha256_file(mini_entry) if mini_entry.is_file() else "missing"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _public_row(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    return {
        "instance_id": str(row.get("instance_id") or row.get("task_id") or ""),
        "problem_statement": str(
            row.get("problem_statement") or row.get("issue") or ""
        ),
        "repo": str(row.get("repo") or ""),
        "base_commit": str(row.get("base_commit") or ""),
        "split": split,
    }


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("instance_id") or row.get("task_id") or "").strip()


def _repo_cluster(row: Mapping[str, Any]) -> str:
    repository = str(row.get("repo") or "").strip()
    if repository:
        return f"repo:{repository}"
    instance_id = _row_id(row)
    if not instance_id:
        raise ValueError("SWE-bench row is missing instance_id/task_id")
    return f"instance:{instance_id}"


def _take_cluster_groups(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_repo_cluster(row), []).append(row)
    selected: list[dict[str, Any]] = []
    for group in groups.values():
        if selected and len(selected) + len(group) > limit:
            break
        selected.extend(group)
    return selected


def _representative_order(
    rows: list[dict[str, Any]], *, seed: int, namespace: str
) -> list[dict[str, Any]]:
    """Return an input-order-independent, repository-interleaved ordering."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        instance_id = _row_id(row)
        if not instance_id:
            raise ValueError("SWE-bench row is missing instance_id/task_id")
        repository = str(row.get("repo") or instance_id.split("__", 1)[0] or "unknown")
        groups.setdefault(repository, []).append(row)

    def digest(value: str) -> str:
        return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()

    for repository, items in groups.items():
        items.sort(key=lambda row: digest(f"{repository}:{_row_id(row)}"))
    repository_order = sorted(
        groups, key=lambda repository: digest(f"repo:{repository}")
    )
    ordered: list[dict[str, Any]] = []
    offset = 0
    while True:
        added = False
        for repository in repository_order:
            items = groups[repository]
            if offset < len(items):
                ordered.append(items[offset])
                added = True
        if not added:
            return ordered
        offset += 1


def _validate_disjoint_pools(pools: Mapping[str, list[dict[str, Any]]]) -> None:
    item_owner: dict[str, str] = {}
    cluster_owner: dict[str, str] = {}
    for name, rows in pools.items():
        for row in rows:
            instance_id = _row_id(row)
            if not instance_id:
                raise ValueError(f"{name} pool contains a row without an instance id")
            previous = item_owner.get(instance_id)
            if previous is not None:
                raise ValueError(
                    f"SWE-bench instance {instance_id!r} appears in both "
                    f"{previous!r} and {name!r} pools"
                )
            item_owner[instance_id] = name
            cluster = _repo_cluster(row)
            cluster_previous = cluster_owner.get(cluster)
            if cluster_previous is not None and cluster_previous != name:
                raise ValueError(
                    f"SWE-bench cluster {cluster!r} appears in both "
                    f"{cluster_previous!r} and {name!r} pools"
                )
            cluster_owner[cluster] = name


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    search_fraction: float,
    calibration_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < search_fraction < 1:
        raise ValueError("search_fraction must be between 0 and 1")
    if not 0 <= calibration_fraction < 1 or search_fraction + calibration_fraction >= 1:
        raise ValueError("search_fraction + calibration_fraction must be less than 1")
    pools: dict[str, list[dict[str, Any]]] = {
        "search": [],
        "calibration": [],
        "final": [],
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_repo_cluster(row), []).append(row)
    for cluster, group in groups.items():
        value = (
            int.from_bytes(
                hashlib.sha256(f"{seed}:{cluster}".encode()).digest()[:8], "big"
            )
            / 2**64
        )
        if value < search_fraction:
            pool = "search"
        elif value < search_fraction + calibration_fraction:
            pool = "calibration"
        else:
            pool = "final"
        pools[pool].extend(group)

    required = ["search", "final"]
    if calibration_fraction > 0:
        required.append("calibration")
    if len(groups) >= len(required) and any(not pools[name] for name in required):
        ordered_clusters = sorted(
            groups,
            key=lambda cluster: hashlib.sha256(
                f"{seed}:fallback:{cluster}".encode()
            ).hexdigest(),
        )
        cluster_count = len(ordered_clusters)
        search_count = max(
            1,
            min(
                cluster_count - len(required) + 1,
                round(cluster_count * search_fraction),
            ),
        )
        calibration_count = 0
        if calibration_fraction > 0:
            calibration_count = max(
                1,
                min(
                    cluster_count - search_count - 1,
                    round(cluster_count * calibration_fraction),
                ),
            )
        assigned = {
            "search": ordered_clusters[:search_count],
            "calibration": ordered_clusters[
                search_count : search_count + calibration_count
            ],
            "final": ordered_clusters[search_count + calibration_count :],
        }
        pools = {
            name: [row for cluster in cluster_ids for row in groups[cluster]]
            for name, cluster_ids in assigned.items()
        }
    return {
        name: _representative_order(items, seed=seed, namespace=name)
        for name, items in pools.items()
    }


def _safe_identity(value: str, *, max_length: int = 80) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._-")
    normalized = normalized or "attempt"
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    if max_length <= len(digest):
        return digest[:max_length]
    prefix_length = max_length - len(digest) - 1
    return f"{normalized[:prefix_length]}_{digest}"


def _official_eval_identity(
    *, attempt_id: str, instance_id: str, patch_text: str
) -> tuple[str, str, str]:
    patch_sha256 = hashlib.sha256(patch_text.encode()).hexdigest()
    attempt = _safe_identity(attempt_id, max_length=48)
    instance = _safe_identity(instance_id, max_length=64)
    suffix = patch_sha256[:16]
    return (
        f"traceunit_{attempt}_{instance}_{suffix}",
        f"traceunit_{attempt}_{suffix}",
        patch_sha256,
    )


def _run_official_patch_evaluation(
    *,
    instance_path: Path,
    patch_path: Path,
    task_dir: Path,
    attempt_id: str,
    dataset_name: str,
    dataset_split: str,
    timeout_s: int,
) -> int:
    task_dir.mkdir(parents=True, exist_ok=True)
    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    instance_id = str(
        instance.get("task_id") or instance.get("instance_id") or ""
    ).strip()
    if not instance_id:
        raise ValueError("instance file is missing task_id/instance_id")
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    run_id, model_name, patch_sha256 = _official_eval_identity(
        attempt_id=attempt_id,
        instance_id=instance_id,
        patch_text=patch_text,
    )
    verdict_path = task_dir / "official_verdict.json"
    if not patch_text.strip():
        write_json(
            verdict_path,
            {
                "instance_id": instance_id,
                "run_id": run_id,
                "model_name": model_name,
                "patch_sha256": patch_sha256,
                "status": "empty_patch",
                "resolved": False,
            },
        )
        return 1

    prediction_path = task_dir / "single_pred.json"
    write_json(
        prediction_path,
        {
            instance_id: {
                "model_name_or_path": model_name,
                "instance_id": instance_id,
                "model_patch": patch_text,
            }
        },
    )
    harness_dir = task_dir / "official_harness" / run_id
    harness_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "uvx",
        "--from",
        SWEBENCH_HARNESS_SPEC,
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        dataset_name,
        "-s",
        dataset_split,
        "-i",
        instance_id,
        "-p",
        str(prediction_path.resolve()),
        "--max_workers",
        "1",
        "-t",
        str(max(1, timeout_s)),
        "--cache_level",
        "instance",
        "--clean",
        "False",
        "-id",
        run_id,
        "--report_dir",
        str((harness_dir / "reports").resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=harness_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(60, timeout_s + 60),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        (task_dir / "official_eval_stdout.txt").write_text(stdout, encoding="utf-8")
        (task_dir / "official_eval_stderr.txt").write_text(
            stderr + f"\nTIMEOUT after {max(60, timeout_s + 60)}s\n",
            encoding="utf-8",
        )
        write_json(
            verdict_path,
            {
                "instance_id": instance_id,
                "run_id": run_id,
                "model_name": model_name,
                "patch_sha256": patch_sha256,
                "status": "evaluator_timeout",
                "resolved": False,
            },
        )
        return 2
    (task_dir / "official_eval_stdout.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    (task_dir / "official_eval_stderr.txt").write_text(
        completed.stderr, encoding="utf-8"
    )
    report_path = (
        harness_dir
        / "logs"
        / "run_evaluation"
        / run_id
        / model_name.replace("/", "__")
        / instance_id
        / "report.json"
    )
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    instance_report = report.get(instance_id) if isinstance(report, Mapping) else None
    resolved = bool(
        isinstance(instance_report, Mapping) and instance_report.get("resolved")
    )
    status = "resolved" if resolved else "unresolved"
    returncode = 0 if resolved else 1
    if completed.returncode != 0 or not isinstance(instance_report, Mapping):
        status = "evaluator_error"
        returncode = 2
    write_json(
        verdict_path,
        {
            "instance_id": instance_id,
            "run_id": run_id,
            "model_name": model_name,
            "patch_sha256": patch_sha256,
            "status": status,
            "resolved": resolved,
            "harness_returncode": completed.returncode,
            "report_path": str(report_path.resolve()),
            "report": instance_report if isinstance(instance_report, Mapping) else {},
        },
    )
    return returncode


def _as_number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _trajectory_usage(payload: Mapping[str, Any]) -> dict[str, float | int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    message_cost = 0.0
    for message in payload.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        extra = (
            message.get("extra") if isinstance(message.get("extra"), Mapping) else {}
        )
        response = extra.get("response") if isinstance(extra, Mapping) else None
        usage = response.get("usage") if isinstance(response, Mapping) else None
        if isinstance(usage, Mapping):
            prompt = int(
                _as_number(usage.get("prompt_tokens") or usage.get("input_tokens"))
            )
            completion = int(
                _as_number(usage.get("completion_tokens") or usage.get("output_tokens"))
            )
            prompt_tokens += prompt
            completion_tokens += completion
            total_tokens += (
                int(_as_number(usage.get("total_tokens"))) or prompt + completion
            )
        message_cost += _as_number(
            extra.get("cost") if isinstance(extra, Mapping) else 0
        )
    info = payload.get("info") if isinstance(payload.get("info"), Mapping) else {}
    model_stats = (
        info.get("model_stats") if isinstance(info.get("model_stats"), Mapping) else {}
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        "monetary_cost": _as_number(model_stats.get("instance_cost")) or message_cost,
        "api_calls": int(_as_number(model_stats.get("api_calls"))),
    }


def _task_status(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    if bool(row.get("passed")):
        return "resolved"
    if bool(metrics.get("dry_run")):
        return "dry_run"
    if bool(metrics.get("timed_out")):
        return "agent_timeout"
    agent_returncode = metrics.get("returncode")
    if agent_returncode not in (None, 0):
        return "agent_error"
    exit_status = str(metrics.get("exit_status") or "")
    if exit_status and not exit_status.startswith("Submit"):
        if exit_status == "LimitsExceeded":
            return "agent_limit"
        return "agent_error"
    patch_bytes = metrics.get("patch_bytes")
    if patch_bytes is not None and int(_as_number(patch_bytes)) <= 0:
        return "empty_patch"
    evaluator_returncode = metrics.get("evaluator_returncode")
    if evaluator_returncode is None:
        return "evaluator_missing"
    if evaluator_returncode == 1:
        return "unresolved"
    if evaluator_returncode != 0:
        return "evaluator_error"
    return "unresolved"


def _augment_miniswe_trajectories(
    trace_path: Path,
) -> dict[str, dict[str, Any]]:
    """Parse mini-SWE messages while removing pointers to sealed eval artifacts."""

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        row["events"] = [
            event
            for event in row.get("events") or []
            if not (
                isinstance(event, Mapping)
                and event.get("kind") == "artifact"
                and any(
                    marker in str((event.get("input") or {}).get("name") or "")
                    for marker in ("official_eval", "eval_stdout", "eval_stderr")
                )
            )
        ]
        row["artifact_paths"] = [
            path
            for path in row.get("artifact_paths") or []
            if not any(
                marker in Path(str(path)).name
                for marker in ("official_eval", "eval_stdout", "eval_stderr")
            )
        ]
        raw_metrics = dict(row.get("metrics") or {})
        dump_text = str(raw_metrics.get("task_dump") or "")
        dump = Path(dump_text) if dump_text else None
        payload: dict[str, Any] | None = None
        trajectory_path: Path | None = None
        if dump is not None and dump.is_dir():
            task_id = str(row.get("task_id") or "")
            expected = dump / "miniswe_run" / task_id / f"{task_id}.traj.json"
            trajectories = (
                [expected]
                if expected.is_file()
                else list(dump.glob("miniswe_run/**/*.traj.json"))
            )
            if trajectories:
                trajectory_path = max(
                    trajectories, key=lambda path: path.stat().st_mtime
                )
                try:
                    loaded = json.loads(trajectory_path.read_text(encoding="utf-8"))
                    payload = loaded if isinstance(loaded, dict) else None
                except (OSError, json.JSONDecodeError):
                    payload = None

        usage = _trajectory_usage(payload or {})
        if payload is not None:
            events = list(row.get("events") or [])
            for index, message in enumerate(payload.get("messages") or []):
                if not isinstance(message, Mapping):
                    continue
                extra = (
                    message.get("extra")
                    if isinstance(message.get("extra"), Mapping)
                    else {}
                )
                actions = extra.get("actions") if isinstance(extra, Mapping) else None
                if actions:
                    for action_index, action in enumerate(actions):
                        if not isinstance(action, Mapping):
                            continue
                        events.append(
                            {
                                "event_id": f"trajectory-{index}-action-{action_index}",
                                "kind": "action",
                                "input": action.get("command") or action,
                                "output": None,
                                "metadata": {
                                    "role": message.get("role"),
                                    "message_index": index,
                                },
                            }
                        )
                if isinstance(extra, Mapping) and (
                    "raw_output" in extra
                    or "returncode" in extra
                    or "exception_info" in extra
                ):
                    events.append(
                        {
                            "event_id": f"trajectory-{index}-observation",
                            "kind": "observation",
                            "input": None,
                            "output": extra.get("raw_output"),
                            "metadata": {
                                "returncode": extra.get("returncode"),
                                "exception_info": extra.get("exception_info"),
                                "message_index": index,
                            },
                        }
                    )
            row["events"] = events
            if trajectory_path is not None:
                row.setdefault("artifact_paths", []).append(
                    str(trajectory_path.resolve())
                )

        status = _task_status(row, raw_metrics)
        safe_metrics = {
            key: raw_metrics[key]
            for key in _SAFE_TRACE_METRIC_KEYS
            if key in raw_metrics
        }
        safe_metrics.update(usage)
        safe_metrics["status_detail"] = status
        if payload is not None:
            info = (
                payload.get("info") if isinstance(payload.get("info"), Mapping) else {}
            )
            safe_metrics["trajectory_info"] = {
                "exit_status": info.get("exit_status"),
                "model_stats": info.get("model_stats"),
            }
        row["status"] = status
        row["metrics"] = safe_metrics
        summaries[str(row.get("task_id") or "")] = {"status": status, **usage}

    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return summaries


def _finalize_evaluation(
    evaluation: BenchmarkEvaluation,
    *,
    trajectory_stats: Mapping[str, Mapping[str, Any]],
    cache_fingerprint: str,
    cache_payload: Mapping[str, Any],
) -> BenchmarkEvaluation:
    outcomes = tuple(
        replace(
            outcome,
            metadata={
                **outcome.metadata,
                **dict(trajectory_stats.get(outcome.task_id) or {}),
            },
        )
        for outcome in evaluation.outcomes
    )
    total_tokens = sum(
        int(_as_number(item.get("total_tokens"))) for item in trajectory_stats.values()
    )
    monetary_cost = sum(
        _as_number(item.get("monetary_cost")) for item in trajectory_stats.values()
    )
    status_counts: dict[str, int] = {}
    for item in trajectory_stats.values():
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return replace(
        evaluation,
        cost=float(total_tokens) if total_tokens else evaluation.cost,
        outcomes=outcomes,
        metadata={
            **evaluation.metadata,
            "cache_fingerprint": cache_fingerprint,
            "cache_identity": dict(cache_payload),
            "total_tokens": total_tokens,
            "monetary_cost": monetary_cost,
            "task_status_counts": status_counts,
        },
    )


def _module_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("_eval-patch")
    evaluate.add_argument("--instance-path", required=True, type=Path)
    evaluate.add_argument("--patch-path", required=True, type=Path)
    evaluate.add_argument("--task-dir", required=True, type=Path)
    evaluate.add_argument("--attempt-id", required=True)
    evaluate.add_argument("--dataset-name", required=True)
    evaluate.add_argument("--dataset-split", default="test")
    evaluate.add_argument("--timeout-s", type=int, default=870)
    args = parser.parse_args(argv)
    if args.command == "_eval-patch":
        return _run_official_patch_evaluation(
            instance_path=args.instance_path,
            patch_path=args.patch_path,
            task_dir=args.task_dir,
            attempt_id=args.attempt_id,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            timeout_s=args.timeout_s,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(_module_main())
