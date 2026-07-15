"""Source-backed mini-SWE-agent runner (in-repo; no WorldCalib dependency).

Ported from WorldCalib's ``worldcalib.coding.swebench`` so the SWE-bench adapter
is self-contained. It drives a candidate's mini-SWE-agent source through the
vendored ``scripts/run_miniswe_swebench_single.py`` entry, captures the patch and
agent exit status, runs the caller-provided (sealed, in-repo) evaluation command,
and writes the same result envelope that ``common.normalize_worldcalib_result``
consumes. The ``CandidateResult``/``TaskResult`` shapes are inlined here (they
were WorldCalib's shared schema) so nothing imports ``worldcalib``.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_MINI_SWE_AGENT_SOURCE_PATH = Path("src/traceunit/scaffolds/mini_swe_agent_baseline")
DEFAULT_MINI_SWE_AGENT_NAME = "mini_swe_agent_source"

_SWEBENCH_DUMP_EVIDENCE_FILES = [
    "miniswe_stdout.txt",
    "official_eval_stdout.txt",
    "official_eval_stderr.txt",
    "stdout.txt",
    "stderr.txt",
    "eval_stdout.txt",
    "eval_stderr.txt",
]

# Eval-gate entry script. The trusted copy lives at ``<project_root>/scripts``;
# the same filename inside a candidate snapshot is a read-only reference and must
# never be invoked. The list covers every shape seen in launcher command strings.
EVAL_ENTRY_SCRIPT_FILENAME = "run_miniswe_swebench_single.py"
EVAL_ENTRY_RELATIVE_FORMS = (
    "scripts/run_miniswe_swebench_single.py",
    "./scripts/run_miniswe_swebench_single.py",
    "run_miniswe_swebench_single.py",
)


@dataclass(frozen=True)
class TaskResult:
    """Evaluation result for one task (inlined from WorldCalib's shared schema)."""

    task_id: str
    question: str
    gold_answer: str
    prediction: str
    score: float
    passed: bool
    prompt_tokens: int
    completion_tokens: int
    retrieved: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    """Evaluation result for one scaffold/config candidate."""

    candidate_id: str
    scaffold_name: str
    passrate: float
    average_score: float
    token_consuming: int
    avg_token_consuming: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    count: int
    config: dict[str, Any]
    result_path: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateResult":
        data = dict(payload)
        if "scaffold_name" not in data and "seed_name" in data:
            data["scaffold_name"] = data.pop("seed_name")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SwebenchInstance:
    """One software-engineering issue in SWE-bench-compatible shape."""

    task_id: str
    problem_statement: str
    repo: str = ""
    base_commit: str = ""
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SwebenchInstance":
        task_id = str(
            payload.get("instance_id")
            or payload.get("task_id")
            or payload.get("id")
            or ""
        ).strip()
        if not task_id:
            raise ValueError("SWE-bench instance must include instance_id/task_id/id")
        problem = str(
            payload.get("problem_statement")
            or payload.get("issue")
            or payload.get("prompt")
            or payload.get("task")
            or ""
        )
        if not problem:
            raise ValueError(f"SWE-bench instance {task_id!r} is missing problem text")
        metadata = {
            str(key): value
            for key, value in payload.items()
            if key
            not in {
                "instance_id",
                "task_id",
                "id",
                "problem_statement",
                "issue",
                "prompt",
                "task",
                "repo",
                "base_commit",
                "split",
            }
        }
        return cls(
            task_id=task_id,
            problem_statement=problem,
            repo=str(payload.get("repo") or ""),
            base_commit=str(payload.get("base_commit") or ""),
            split=str(payload.get("split") or "train"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["instance_id"] = self.task_id
        return payload


@dataclass(frozen=True)
class CodingAgentRun:
    """One coding-agent attempt on a SWE-bench instance."""

    prediction: str
    passed: bool
    score: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MiniSweAgentSourceRunner:
    """Evaluate a source-backed mini-SWE-agent candidate on local tasks."""

    def __init__(
        self,
        *,
        instances: list[SwebenchInstance],
        out_dir: Path,
        timeout_s: int = 300,
        max_eval_workers: int = 1,
        dry_run: bool = False,
        force: bool = False,
        project_root: Path | None = None,
    ) -> None:
        self.instances = instances
        self.out_dir = out_dir
        self.timeout_s = timeout_s
        self.max_eval_workers = max(1, int(max_eval_workers))
        self.dry_run = dry_run
        self.force = force
        # Used to rewrite the eval-gate entry script into an absolute path so
        # proposer edits to the in-candidate copy can never affect grading.
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )

    def evaluate_candidate(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_id: str,
        agent_name: str = DEFAULT_MINI_SWE_AGENT_NAME,
    ) -> CandidateResult:
        candidate_dir = self.out_dir / "candidate_results"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        result_path = candidate_dir / f"{candidate_id}.json"
        if not self.force:
            existing = _load_candidate_result(
                result_path,
                candidate_id=candidate_id,
                agent_name=agent_name,
                config=dict(candidate),
            )
            if existing is not None:
                return existing

        if self.max_eval_workers == 1 or len(self.instances) <= 1:
            task_results = [
                self._evaluate_instance(candidate, candidate_id=candidate_id, instance=instance)
                for instance in self.instances
            ]
        else:
            workers = min(self.max_eval_workers, len(self.instances))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                task_results = list(
                    pool.map(
                        lambda instance: self._evaluate_instance(
                            candidate,
                            candidate_id=candidate_id,
                            instance=instance,
                        ),
                        self.instances,
                    )
                )
        count = len(task_results)
        passrate = sum(1 for item in task_results if item.passed) / count if count else 0.0
        average_score = sum(item.score for item in task_results) / count if count else 0.0
        prompt_tokens = sum(item.prompt_tokens for item in task_results)
        completion_tokens = sum(item.completion_tokens for item in task_results)
        token_consuming = prompt_tokens + completion_tokens
        result = CandidateResult(
            candidate_id=candidate_id,
            scaffold_name=agent_name,
            passrate=passrate,
            average_score=average_score,
            token_consuming=token_consuming,
            avg_token_consuming=(token_consuming / count if count else 0.0),
            avg_prompt_tokens=(prompt_tokens / count if count else 0.0),
            avg_completion_tokens=(completion_tokens / count if count else 0.0),
            count=count,
            config=dict(candidate),
            result_path=str(result_path),
        )
        payload = {
            "candidate": result.to_dict(),
            "tasks": [item.to_dict() for item in task_results],
            "score_breakdown": _score_breakdown(task_results),
        }
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def _evaluate_instance(
        self,
        candidate: Mapping[str, Any],
        *,
        candidate_id: str,
        instance: SwebenchInstance,
    ) -> TaskResult:
        task_dir = self.out_dir / "agent_runs" / candidate_id / instance.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        instance_path = task_dir / "instance.json"
        patch_path = task_dir / "patch.diff"
        instance_path.write_text(
            json.dumps(instance.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        run = self._run_agent(
            candidate,
            instance=instance,
            task_dir=task_dir,
            instance_path=instance_path,
            patch_path=patch_path,
        )
        return TaskResult(
            task_id=instance.task_id,
            question=instance.problem_statement,
            gold_answer="",
            prediction=run.prediction,
            score=run.score,
            passed=run.passed,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            retrieved=[],
            metadata=run.metadata,
        )

    def _run_agent(
        self,
        candidate: Mapping[str, Any],
        *,
        instance: SwebenchInstance,
        task_dir: Path,
        instance_path: Path,
        patch_path: Path,
    ) -> CodingAgentRun:
        if self.dry_run:
            patch_path.write_text("", encoding="utf-8")
            return CodingAgentRun(
                prediction="",
                passed=False,
                score=0.0,
                metadata={
                    "benchmark": "swebench",
                    "agent": DEFAULT_MINI_SWE_AGENT_NAME,
                    "dry_run": True,
                    "repo": instance.repo,
                    "base_commit": instance.base_commit,
                    "patch_path": str(patch_path),
                    "patch_bytes": 0,
                },
            )

        source_path = _candidate_source_path(candidate)
        if source_path is None:
            source_path = DEFAULT_MINI_SWE_AGENT_SOURCE_PATH
        if not source_path.exists():
            raise FileNotFoundError(
                "mini-SWE-agent source path does not exist. Set source_project_path "
                f"or vendor the baseline under {DEFAULT_MINI_SWE_AGENT_SOURCE_PATH}."
            )

        command = _format_command(
            candidate.get("command") or candidate.get("agent_command"),
            source_path=source_path,
            task_dir=task_dir,
            instance_path=instance_path,
            patch_path=patch_path,
            instance=instance,
        )
        # run_miniswe_swebench_single.py is platform scaffolding, not a
        # candidate-edited file. Force the absolute repo-root path regardless of
        # how the launcher wrote it, so a proposer-edited in-candidate copy can
        # never affect grading. The agent's behaviour comes from --source-path +
        # the candidate's src/minisweagent/, not from this entry script.
        command = _rewrite_eval_entry_to_abs(command, self.project_root)
        if not command:
            raise ValueError(
                "SWE-bench mini-SWE-agent evaluation requires candidate['command'] "
                "or candidate['agent_command'] when dry_run is false."
            )

        started = time.time()
        try:
            completed = _run_subprocess_with_timeout(
                command,
                cwd=source_path,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            (task_dir / "stdout.txt").write_text(
                _timeout_output_to_text(exc.stdout), encoding="utf-8"
            )
            (task_dir / "stderr.txt").write_text(
                _timeout_output_to_text(exc.stderr), encoding="utf-8"
            )
            if not patch_path.exists():
                patch_path.write_text("", encoding="utf-8")
            return CodingAgentRun(
                prediction="",
                passed=False,
                score=0.0,
                prompt_tokens=_int_metadata(candidate, "prompt_tokens"),
                completion_tokens=_int_metadata(candidate, "completion_tokens"),
                metadata={
                    "benchmark": "swebench",
                    "agent": DEFAULT_MINI_SWE_AGENT_NAME,
                    "source_project_path": str(source_path),
                    "repo": instance.repo,
                    "base_commit": instance.base_commit,
                    "patch_path": str(patch_path),
                    "task_dir": str(task_dir),
                    "task_dump": str(task_dir),
                    "dump_evidence_files": list(_SWEBENCH_DUMP_EVIDENCE_FILES),
                    "returncode": None,
                    "evaluator_returncode": None,
                    "duration_s": time.time() - started,
                    "timed_out": True,
                    "timeout_s": self.timeout_s,
                    "patch_bytes": 0,
                },
            )
        duration_s = time.time() - started
        (task_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (task_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if not patch_path.exists():
            patch_path.write_text(_extract_patch_from_stdout(completed.stdout), encoding="utf-8")

        eval_command = _format_command(
            candidate.get("eval_command") or candidate.get("evaluation_command"),
            source_path=source_path,
            task_dir=task_dir,
            instance_path=instance_path,
            patch_path=patch_path,
            instance=instance,
        )
        eval_command = _rewrite_eval_entry_to_abs(eval_command, self.project_root)
        evaluator_returncode: int | None = None
        if eval_command:
            try:
                evaluated = _run_subprocess_with_timeout(
                    eval_command,
                    cwd=source_path,
                    timeout=self.timeout_s,
                )
                evaluator_returncode = evaluated.returncode
                (task_dir / "eval_stdout.txt").write_text(evaluated.stdout, encoding="utf-8")
                (task_dir / "eval_stderr.txt").write_text(evaluated.stderr, encoding="utf-8")
            except subprocess.TimeoutExpired as exc:
                (task_dir / "eval_stdout.txt").write_text(
                    _timeout_output_to_text(exc.stdout), encoding="utf-8"
                )
                (task_dir / "eval_stderr.txt").write_text(
                    _timeout_output_to_text(exc.stderr), encoding="utf-8"
                )

        passed = completed.returncode == 0 and evaluator_returncode == 0
        if evaluator_returncode is None:
            passed = False
        exit_status = _read_agent_exit_status(task_dir)
        patch_text = patch_path.read_text(encoding="utf-8", errors="ignore")
        metadata: dict[str, Any] = {
            "benchmark": "swebench",
            "agent": DEFAULT_MINI_SWE_AGENT_NAME,
            "source_project_path": str(source_path),
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "patch_path": str(patch_path),
            "task_dir": str(task_dir),
            "task_dump": str(task_dir),
            "dump_evidence_files": list(_SWEBENCH_DUMP_EVIDENCE_FILES),
            "returncode": completed.returncode,
            "evaluator_returncode": evaluator_returncode,
            "duration_s": duration_s,
            "exit_status": exit_status,
            "patch_bytes": len(patch_text.strip()),
        }
        if exit_status and exit_status != "Submitted" and not exit_status.startswith("Submit"):
            tail = _read_stdout_tail(task_dir)
            if tail:
                metadata["error_tail"] = tail
        return CodingAgentRun(
            prediction=patch_text,
            passed=passed,
            score=1.0 if passed else 0.0,
            prompt_tokens=_int_metadata(candidate, "prompt_tokens"),
            completion_tokens=_int_metadata(candidate, "completion_tokens"),
            metadata=metadata,
        )


def load_swebench_instances(
    path: Path | None,
    *,
    split: str = "train",
    limit: int = 0,
) -> list[SwebenchInstance]:
    """Load local SWE-bench-compatible JSON/JSONL rows."""

    if path is None:
        raise ValueError(
            "SWE-bench optimization requires a data path. Use a JSON/JSONL file "
            "with instance_id and problem_statement fields."
        )
    rows = _load_rows(path)
    instances = [SwebenchInstance.from_dict(row) for row in rows]
    selected = [item for item in instances if item.split == split]
    if not selected:
        selected = instances
    if limit:
        selected = selected[:limit]
    return selected


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench data path does not exist: {path}")
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("instances", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("SWE-bench data must be a JSON list or {'instances': [...]}")
    return [dict(row) for row in rows]


def _candidate_source_path(candidate: Mapping[str, Any]) -> Path | None:
    extra = candidate.get("extra") if isinstance(candidate.get("extra"), Mapping) else {}
    for key in ("source_project_path", "upstream_source_path", "mini_swe_agent_source_path"):
        value = candidate.get(key) or extra.get(key)
        if value:
            return Path(str(value)).expanduser()
    return None


def _format_command(
    value: object,
    *,
    source_path: Path,
    task_dir: Path,
    instance_path: Path,
    patch_path: Path,
    instance: SwebenchInstance,
) -> list[str]:
    if not value:
        return []
    replacements = {
        "source_path": str(source_path.resolve()),
        "task_dir": str(task_dir.resolve()),
        "instance_path": str(instance_path.resolve()),
        "patch_path": str(patch_path.resolve()),
        "instance_id": instance.task_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
    }
    if isinstance(value, str):
        return shlex.split(value.format(**replacements))
    if isinstance(value, Iterable):
        return [str(item).format(**replacements) for item in value]
    raise TypeError("command must be a string or list of strings")


def _rewrite_eval_entry_to_abs(tokens: list[str], project_root: Path) -> list[str]:
    """Force the eval/agent command to invoke the repo-root eval entry script."""

    if not tokens:
        return tokens
    abs_target = project_root / "scripts" / EVAL_ENTRY_SCRIPT_FILENAME
    abs_str = str(abs_target)
    rewritten = list(tokens)
    for idx, token in enumerate(rewritten):
        if token in EVAL_ENTRY_RELATIVE_FORMS or Path(token).name == EVAL_ENTRY_SCRIPT_FILENAME:
            rewritten[idx] = abs_str
            return rewritten
    return rewritten


def _extract_patch_from_stdout(stdout: str) -> str:
    marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if marker in stdout:
        return stdout.split(marker, 1)[1].strip()
    return ""


def _read_agent_exit_status(task_dir: Path) -> str | None:
    """Return mini-SWE-agent's terminal exit_status for this task (or None)."""

    try:
        candidates = sorted((task_dir / "miniswe_run").glob("exit_statuses_*.yaml"))
    except OSError:
        return None
    if not candidates:
        return None
    try:
        lines = candidates[-1].read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    in_block = False
    for line in lines:
        if line.startswith("instances_by_exit_status:"):
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if stripped and not stripped.startswith("- ") and stripped.endswith(":"):
                return stripped[:-1].strip()
    return None


def _read_stdout_tail(task_dir: Path, *, max_chars: int = 1500) -> str:
    """Tail of the agent's stdout (the traceback lives here), '' if absent."""

    for name in ("miniswe_stdout.txt", "stdout.txt"):
        path = task_dir / name
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if text.strip():
            return text[-max_chars:]
    return ""


def _run_subprocess_with_timeout(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a command and kill its whole process group on timeout."""

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout or exc.stdout,
            stderr=stderr or exc.stderr,
        ) from exc
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=2)
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _int_metadata(candidate: Mapping[str, Any], key: str) -> int:
    value = candidate.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _score_breakdown(task_results: list[TaskResult]) -> dict[str, dict[str, object]]:
    breakdown: dict[str, dict[str, object]] = {
        "all": {
            "count": len(task_results),
            "passrate": (
                sum(1 for item in task_results if item.passed) / len(task_results)
                if task_results
                else 0.0
            ),
            "average_score": (
                sum(item.score for item in task_results) / len(task_results)
                if task_results
                else 0.0
            ),
        }
    }
    for item in task_results:
        breakdown[item.task_id] = {
            "count": 1,
            "passrate": 1.0 if item.passed else 0.0,
            "average_score": float(item.score),
        }
    return breakdown


def _load_candidate_result(
    result_path: Path,
    *,
    candidate_id: str,
    agent_name: str,
    config: dict[str, Any],
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
        or candidate.scaffold_name != agent_name
        or candidate.config != config
    ):
        return None
    return candidate
