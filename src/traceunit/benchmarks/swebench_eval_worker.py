"""Sealed official SWE-bench patch evaluation worker.

This file is executed directly by whichever Python the WorldCalib runner
selects, so it intentionally depends only on the standard library; the
official harness itself runs through uvx in its own environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

SWEBENCH_HARNESS_SPEC = "swebench==4.1.0"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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


def official_eval_identity(
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


def run_official_patch_evaluation(
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
    run_id, model_name, patch_sha256 = official_eval_identity(
        attempt_id=attempt_id,
        instance_id=instance_id,
        patch_text=patch_text,
    )
    verdict_path = task_dir / "official_verdict.json"
    if not patch_text.strip():
        _write_json(
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
    _write_json(
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
        _write_json(
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
    _write_json(
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


def main(argv: list[str] | None = None) -> int:
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
        return run_official_patch_evaluation(
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
    raise SystemExit(main())
