from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from traceunit.agent_probe import run_declarative_probe
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.common import (
    load_cached_evaluation,
    normalize_worldcalib_result,
)
from traceunit.benchmarks.pools import (
    freeze_benchmark_plan,
    load_benchmark_plan,
    load_pool_items,
    pool_identity,
    take_cluster_groups,
)
from traceunit.config import BenchmarkConfig
from traceunit.io import sha256_file, sha256_tree, write_json
from traceunit.models import BenchmarkPlan, PoolSliceRef


_ADAPTER_VERSION = 3


class AppWorldAdapter(BenchmarkAdapter):
    name = "appworld"
    supports_agent_probe = True

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._plan: BenchmarkPlan | None = None

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
        root = self.config.worldcalib_root
        if not (root / ".venv-appworld/bin/python").exists():
            raise FileNotFoundError(f"AppWorld evaluation venv is missing under {root}")
        pool_dir = work_dir / "benchmark_data" / "appworld"
        frozen_plan = pool_dir / "plan.json"
        if frozen_plan.is_file():
            plan = load_benchmark_plan(frozen_plan)
            self.bind_plan(plan)
            for pool in (plan.search, plan.final):
                load_pool_items(pool)
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
                    "explicit AppWorld pools require readable search and final files; "
                    f"missing={missing}"
                )
            search = _load_task_ids(configured["search"])
            final = _load_task_ids(configured["final"])
        else:
            default_challenge = root / "data/appworld/split_challenge.json"
            default_normal = root / "data/appworld/split.json"
            source_manifest = (
                self.config.split_manifest_path
                or (
                    default_challenge if default_challenge.is_file() else default_normal
                )
            ).resolve()
            if not source_manifest.is_file():
                raise FileNotFoundError(
                    f"AppWorld split manifest is missing: {source_manifest}"
                )
            raw = json.loads(source_manifest.read_text(encoding="utf-8"))
            search = list(raw.get(self.config.search_split) or raw.get("train") or [])
            final = list(raw.get(self.config.heldout_split) or raw.get("test") or [])
        search = take_cluster_groups(
            search, self.config.search_limit, cluster_key=_scenario_id
        )
        final = take_cluster_groups(
            final, self.config.final_limit, cluster_key=_scenario_id
        )
        if not search or not final:
            raise ValueError("AppWorld search and final pools must be non-empty")
        _validate_disjoint_pools(search, final)
        self._plan = freeze_benchmark_plan(
            root=pool_dir,
            benchmark=self.name,
            search_items=search,
            final_items=final,
            cluster_key=_scenario_id,
        )
        return self._plan

    def preflight(self) -> None:
        if self.config.dry_run:
            return
        if not os.environ.get(self.config.api_key_env):
            raise RuntimeError(
                f"AppWorld target-model key is missing: {self.config.api_key_env}"
            )
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("AppWorld candidate isolation requires Docker")
        try:
            inspected = subprocess.run(
                [docker, "image", "inspect", "python:3.12-slim"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("cannot inspect AppWorld sandbox image") from exc
        if inspected.returncode != 0:
            raise RuntimeError("AppWorld sandbox image is not cached: python:3.12-slim")

    def baseline_source(self) -> Path:
        return (
            self.config.baseline_source_path
            or self.config.worldcalib_root / "src/worldcalib/agentic/backends/appworld"
        ).resolve()

    def context(self) -> str:
        return """AppWorld interactive code agent. The editable artifact is agent.py and helper
modules used by solve(world). The solver model and endpoint are frozen. Candidate execution gets a
world explicitly created with load_ground_truth=False. It may use only the public task instruction,
supervisor data, world.execute, world.task_completed, and documented app APIs. It must never call an
evaluator, inspect ground truth, branch on task ids, or encode specific task answers. A separate
sealed process scores the persisted environment state after candidate execution exits."""

    def evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        pool: PoolSliceRef,
        out_dir: Path,
    ):
        source_hash = sha256_tree(source)
        if self._plan is None:
            raise RuntimeError("prepare() must be called before evaluate()")
        known = (self._plan.search, self._plan.final)
        if pool not in known:
            raise ValueError(f"pool is not part of the prepared plan: {pool.slice_id}")
        raw_task_ids = load_pool_items(pool)
        if not all(isinstance(item, str) and item for item in raw_task_ids):
            raise ValueError(
                f"AppWorld pool contains an invalid task id: {pool.slice_id}"
            )
        task_ids = list(raw_task_ids)
        fingerprint = self._evaluation_fingerprint(
            source_hash=source_hash,
            pool=pool,
        )
        cached = load_cached_evaluation(out_dir)
        if (
            cached is not None
            and not self.config.force
            and (cached.metadata.get("candidate_config") or {}).get(
                "evaluation_fingerprint"
            )
            == fingerprint
        ):
            return cached
        out_dir.mkdir(parents=True, exist_ok=True)

        jobs = [
            (task_id, rep) for task_id in task_ids for rep in range(self.config.repeats)
        ]
        if self.config.dry_run:
            rows = [self._dry_run_row(task_id, rep, out_dir) for task_id, rep in jobs]
        else:
            workers = min(max(1, self.config.concurrency), max(1, len(jobs)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(
                    executor.map(
                        lambda item: self._run_one(
                            source=source,
                            candidate_id=candidate_id,
                            task_id=item[0],
                            rep=item[1],
                            out_dir=out_dir,
                            source_hash=source_hash,
                        ),
                        jobs,
                    )
                )
        task_rows = _aggregate_repetitions(task_ids, rows)
        passrate = (
            sum(float(row["score"]) for row in task_rows) / len(task_rows)
            if task_rows
            else 0.0
        )
        total_prompt = sum(int(row.get("prompt_tokens") or 0) for row in task_rows)
        total_completion = sum(
            int(row.get("completion_tokens") or 0) for row in task_rows
        )
        result_path = out_dir / "candidate_result.json"
        payload = {
            "candidate": {
                "candidate_id": candidate_id,
                "scaffold_name": "appworld_agent",
                "passrate": passrate,
                "token_consuming": total_prompt + total_completion,
                "count": len(task_rows),
                "config": {
                    "source_project_path": str(source.resolve()),
                    "source_sha256": source_hash,
                    "evaluation_fingerprint": fingerprint,
                    "adapter_version": _ADAPTER_VERSION,
                    "ground_truth_isolated": True,
                },
            },
            "tasks": task_rows,
        }
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return normalize_worldcalib_result(
            result_path=result_path,
            benchmark=self.name,
            split=pool.slice_id,
            candidate_id=candidate_id,
            out_dir=out_dir,
        )

    def _evaluation_fingerprint(
        self,
        *,
        source_hash: str,
        pool: PoolSliceRef,
    ) -> str:
        worker = Path(__file__).with_name("appworld_worker.py")
        appworld_init = (
            self.config.worldcalib_root
            / ".venv-appworld/lib/python3.12/site-packages/appworld/__init__.py"
        )
        payload = {
            "adapter_version": _ADAPTER_VERSION,
            "worker_sha256": sha256_file(worker),
            "appworld_sha256": (
                sha256_file(appworld_init) if appworld_init.is_file() else "missing"
            ),
            "source_sha256": source_hash,
            "pool": pool_identity(pool),
            "model": self.config.model,
            "base_url": self.config.base_url,
            "api_key_env": self.config.api_key_env,
            "dry_run": self.config.dry_run,
            "repeats": self.config.repeats,
            "max_interactions": self.config.max_interactions,
            "seed": self.config.benchmark_seed,
            "timeout_s": self.config.timeout_s,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        agent = source / "agent.py"
        proc = subprocess.run(
            [
                str(self.config.unit_python or Path(sys.executable)),
                "-m",
                "py_compile",
                str(agent),
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
            "world.evaluate(": "candidate calls the sealed AppWorld evaluator",
            ".ground_truth": "candidate reads AppWorld ground truth",
            "evaluate_task(": "candidate calls the sealed AppWorld evaluator",
            "task_id ==": "candidate branches on an AppWorld task id",
            "task_id in": "candidate branches on AppWorld task ids",
        }
        return [message for token, message in banned.items() if token in added]

    def _run_one(
        self,
        *,
        source: Path,
        candidate_id: str,
        task_id: str,
        rep: int,
        out_dir: Path,
        source_hash: str,
    ) -> dict[str, Any]:
        safe_candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        run_hash = hashlib.sha256(str(out_dir.resolve()).encode()).hexdigest()[:10]
        experiment = (
            f"traceunit_{run_hash}_{source_hash[:10]}_"
            f"{safe_candidate}_{safe_task}_r{rep}"
        )
        task_out = out_dir / "agent_runs" / safe_task / f"rep{rep}"
        if task_out.exists():
            shutil.rmtree(task_out)
        task_out.mkdir(parents=True, exist_ok=True)
        worker = Path(__file__).with_name("appworld_worker.py").resolve()
        python = self.config.worldcalib_root / ".venv-appworld/bin/python"
        real_root = Path(
            os.environ.get("APPWORLD_ROOT", "/data/home/yuhan/appworld_home")
        ).resolve()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        }
        env["APPWORLD_MODEL"] = self.config.model
        env["APPWORLD_OPENAI_BASE_URL"] = self.config.base_url
        key = os.environ.get(self.config.api_key_env)
        if key:
            env["APPWORLD_OPENAI_API_KEY"] = key
        common = [
            "--task-id",
            task_id,
            "--experiment-name",
            experiment,
            "--out",
            str(task_out),
        ]
        run_cmd = [
            str(python),
            str(worker),
            "run",
            *common,
            "--agent-path",
            str((source / "agent.py").resolve()),
            "--max-interactions",
            str(self.config.max_interactions),
            "--seed",
            str(self.config.benchmark_seed + rep),
        ]
        sandbox_root = task_out / "candidate_appworld"
        sandbox_outputs = sandbox_root / "experiments" / "outputs"
        sandbox_outputs.mkdir(parents=True, exist_ok=True)
        run_cmd, run_env = _sandboxed_appworld_command(
            argv=run_cmd,
            env=env,
            site_packages=python.parent.parent / "lib" / "python3.12" / "site-packages",
            worker=worker,
            source=source,
            task_out=task_out,
            sandbox_outputs=sandbox_outputs,
            real_root=real_root,
            task_id=task_id,
        )
        run_log = task_out / "worker_run.log"
        run_started = time.monotonic()
        run = _run_process(
            run_cmd,
            env=run_env,
            timeout=self.config.timeout_s,
            log_path=run_log,
        )
        run_wall_seconds = time.monotonic() - run_started
        eval_log = task_out / "worker_evaluate.log"
        runtime = _read_json(task_out / "runtime.json")
        worker_seconds = runtime.get("seconds")
        if worker_seconds is not None:
            runtime["worker_reported_seconds"] = worker_seconds
        runtime["seconds"] = round(run_wall_seconds, 3)
        write_json(task_out / "runtime.json", runtime)
        real_experiment = real_root / "experiments" / "outputs" / experiment
        sandbox_experiment = sandbox_outputs / experiment
        sealed_path = task_out / "sealed_evaluation.json"
        if sealed_path.exists():
            sealed_path.unlink()
        evaluated: int | None = None
        if run == 0 and not runtime.get("error") and sandbox_experiment.is_dir():
            if real_experiment.exists():
                shutil.rmtree(real_experiment)
            real_experiment.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(sandbox_experiment, real_experiment)
            eval_env = dict(env)
            eval_env["APPWORLD_ROOT"] = str(real_root)
            eval_cmd = [str(python), str(worker), "evaluate", *common]
            try:
                evaluated = _run_process(
                    eval_cmd,
                    env=eval_env,
                    timeout=max(60, self.config.timeout_s // 2),
                    log_path=eval_log,
                )
            finally:
                shutil.rmtree(real_experiment, ignore_errors=True)
        else:
            reason = runtime.get("error") or (
                "candidate output state is missing" if run == 0 else f"run exited {run}"
            )
            write_json(
                sealed_path,
                {
                    "task_id": task_id,
                    "experiment_name": experiment,
                    "success": False,
                    "error": f"not evaluated: {reason}",
                },
            )
            eval_log.write_text(f"SKIPPED: {reason}\n", encoding="utf-8")
        sealed = _read_json(task_out / "sealed_evaluation.json")
        return {
            "task_id": task_id,
            "rep": rep,
            "success": bool(sealed.get("success")),
            "runtime": runtime,
            "sealed": sealed,
            "task_dump": str(task_out.resolve()),
            "run_returncode": run,
            "eval_returncode": evaluated,
        }

    def _dry_run_row(self, task_id: str, rep: int, out_dir: Path) -> dict[str, Any]:
        task_out = out_dir / "agent_runs" / task_id / f"rep{rep}"
        task_out.mkdir(parents=True, exist_ok=True)
        (task_out / "runtime.json").write_text(
            json.dumps({"task_id": task_id, "dry_run": True}), encoding="utf-8"
        )
        return {
            "task_id": task_id,
            "rep": rep,
            "success": False,
            "runtime": {"dry_run": True},
            "sealed": {"success": False},
            "task_dump": str(task_out.resolve()),
            "run_returncode": 0,
            "eval_returncode": 0,
        }


def _run_process(
    argv: list[str], *, env: dict[str, str], timeout: int, log_path: Path
) -> int:
    proc = subprocess.Popen(
        argv,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
        log_path.write_text(output or "", encoding="utf-8")
        return int(proc.returncode or 0)
    except subprocess.TimeoutExpired as exc:
        _force_remove_docker_container(argv, env)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            output, _ = proc.communicate(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = proc.communicate()
        prefix = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        combined = output or prefix
        log_path.write_text(
            combined + f"\nTIMEOUT after {timeout}s\n", encoding="utf-8"
        )
        return 124


def _force_remove_docker_container(argv: list[str], env: dict[str, str]) -> None:
    if not argv or Path(argv[0]).name != "docker" or "--cidfile" not in argv:
        return
    try:
        cidfile = Path(argv[argv.index("--cidfile") + 1])
        container_id = cidfile.read_text(encoding="utf-8").strip()
    except (OSError, ValueError, IndexError):
        return
    if not container_id:
        return
    try:
        subprocess.run(
            [argv[0], "rm", "--force", container_id],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _sandboxed_appworld_command(
    *,
    argv: list[str],
    env: dict[str, str],
    site_packages: Path,
    worker: Path,
    source: Path,
    task_out: Path,
    sandbox_outputs: Path,
    real_root: Path,
    task_id: str,
) -> tuple[list[str], dict[str, str]]:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("AppWorld candidate isolation requires Docker")
    venv = site_packages.resolve()
    task_root = real_root / "data" / "tasks" / task_id
    specs = task_root / "specs.json"
    dbs = task_root / "dbs"
    for required in (venv, specs, dbs):
        if not required.exists():
            raise FileNotFoundError(f"AppWorld sandbox input is missing: {required}")
    cidfile = task_out / "container.cid"
    cidfile.unlink(missing_ok=True)
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    command = [
        docker,
        "run",
        "--rm",
        "--init",
        "--network",
        "bridge",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "2g",
        "--cpus",
        "2",
        "--user",
        uid_gid,
        "--cidfile",
        str(cidfile.resolve()),
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "--workdir",
        "/trace",
        "--mount",
        f"type=bind,src={venv},dst=/opt/appworld-site,readonly",
        "--mount",
        f"type=bind,src={source.resolve()},dst=/candidate,readonly",
        "--mount",
        f"type=bind,src={worker},dst=/worker.py,readonly",
        "--mount",
        f"type=bind,src={task_out.resolve()},dst=/trace",
        "--mount",
        "type=bind,src=/etc/ssl/certs,dst=/etc/ssl/certs,readonly",
        "--mount",
        f"type=bind,src={specs},dst=/appworld/data/tasks/{task_id}/specs.json,readonly",
        "--mount",
        f"type=bind,src={dbs},dst=/appworld/data/tasks/{task_id}/dbs,readonly",
        "--mount",
        (
            f"type=bind,src={sandbox_outputs.resolve()},"
            "dst=/appworld/experiments/outputs"
        ),
    ]
    for source_path, target in (
        (real_root / "data" / "api_docs", "/appworld/data/api_docs"),
        (real_root / "data" / "base_dbs", "/appworld/data/base_dbs"),
    ):
        if source_path.exists():
            command.extend(
                [
                    "--mount",
                    f"type=bind,src={source_path},dst={target},readonly",
                ]
            )
    sandbox_env = {
        **env,
        "HOME": "/tmp",
        "APPWORLD_ROOT": "/appworld",
        "APPWORLD_CACHE": "/tmp/.appworld",
        "PYTHONPATH": "/opt/appworld-site",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    }
    for key, value in sandbox_env.items():
        if key == "APPWORLD_OPENAI_API_KEY":
            command.extend(["--env", key])
        else:
            command.extend(["--env", f"{key}={value}"])
    sandbox_argv = [
        "python",
        "/worker.py",
        *argv[2:],
    ]
    sandbox_argv = [
        "/candidate/agent.py" if item == str((source / "agent.py").resolve()) else item
        for item in sandbox_argv
    ]
    sandbox_argv = [
        "/trace" if item == str(task_out.resolve()) else item for item in sandbox_argv
    ]
    command.extend(["python:3.12-slim", *sandbox_argv])
    outer_env = {
        "PATH": env["PATH"],
        "LANG": env["LANG"],
        "LC_ALL": env["LC_ALL"],
    }
    if "APPWORLD_OPENAI_API_KEY" in env:
        outer_env["APPWORLD_OPENAI_API_KEY"] = env["APPWORLD_OPENAI_API_KEY"]
    return command, outer_env


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _aggregate_repetitions(
    task_ids: list[str], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for task_id in task_ids:
        repetitions = grouped.get(task_id) or []
        successes = [bool(row.get("success")) for row in repetitions]
        score = sum(successes) / len(successes) if successes else 0.0
        prompt_tokens = sum(
            int((row.get("runtime") or {}).get("prompt_tokens") or 0)
            for row in repetitions
        )
        completion_tokens = sum(
            int((row.get("runtime") or {}).get("completion_tokens") or 0)
            for row in repetitions
        )
        evidence = next(
            (row for row in repetitions if not row.get("success")),
            repetitions[0] if repetitions else {},
        )
        run_codes = [row.get("run_returncode") for row in repetitions]
        eval_codes = [row.get("eval_returncode") for row in repetitions]
        has_infra_error = any(code != 0 for code in run_codes) or any(
            code != 0 for code in eval_codes
        )
        run_status = (
            "infra_error"
            if has_infra_error
            else ("resolved" if score >= 0.5 else "unresolved")
        )
        public_task = _read_json(
            Path(str(evidence.get("task_dump") or "")) / "public_task.json"
        )
        output.append(
            {
                "task_id": task_id,
                "question": str(public_task.get("instruction") or task_id),
                "prediction": f"{sum(successes)}/{len(successes)} pass",
                "score": score,
                "passed": bool(successes) and sum(successes) * 2 >= len(successes),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "metadata": {
                    "benchmark": "appworld",
                    "ground_truth_isolated": True,
                    "run_status": run_status,
                    "rep_successes": successes,
                    "task_dump": evidence.get("task_dump", ""),
                    "task_dumps": [row.get("task_dump", "") for row in repetitions],
                    "dump_evidence_files": [
                        "public_task.json",
                        "runtime.json",
                        "transcript.json",
                        "transcript.txt",
                        "environment_io.md",
                        "api_calls.jsonl",
                    ],
                    "run_returncodes": run_codes,
                    "eval_returncodes": eval_codes,
                },
            }
        )
    return output


def _scenario_id(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def _load_task_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = next(
            (
                payload[key]
                for key in ("tasks", "instances", "data")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if values is None:
            raise ValueError(f"unsupported AppWorld pool shape: {path}")
    else:
        raise ValueError(f"unsupported AppWorld pool shape: {path}")
    task_ids = [str(item).strip() for item in values]
    if any(not item for item in task_ids):
        raise ValueError(f"AppWorld pool contains an empty task id: {path}")
    return task_ids


def _validate_disjoint_pools(search: list[str], final: list[str]) -> None:
    if len(set(search)) != len(search):
        raise ValueError("AppWorld search pool contains duplicate task ids")
    if len(set(final)) != len(final):
        raise ValueError("AppWorld final pool contains duplicate task ids")
    overlap = set(search) & set(final)
    if overlap:
        raise ValueError(f"AppWorld search/final pools overlap: {sorted(overlap)[:3]}")
    scenario_overlap = {_scenario_id(task_id) for task_id in search} & {
        _scenario_id(task_id) for task_id in final
    }
    if scenario_overlap:
        raise ValueError(
            f"AppWorld search/final scenarios overlap: {sorted(scenario_overlap)[:3]}"
        )
