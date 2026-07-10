"""Ground-truth-isolated AppWorld execution and sealed evaluation worker.

This file is executed directly by the AppWorld virtualenv, so it intentionally
depends only on the standard library and AppWorld.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Direct script execution places this directory first on sys.path. Remove it so
# the sibling adapter module appworld.py cannot shadow the installed package.
_THIS_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _THIS_DIR]


def _load_agent(path: Path):
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("traceunit_appworld_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "solve", None)):
        raise RuntimeError("candidate agent must define solve(world)")
    return module


def _transcript_text(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        chunks.append(
            f"===== {message.get('role', '?')} =====\n{message.get('content', '')}"
        )
    return "\n\n".join(chunks)


def run_candidate(args: argparse.Namespace) -> int:
    from appworld import AppWorld

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    world = None
    row: dict[str, Any]
    try:
        agent = _load_agent(args.agent_path.resolve())
        # This is the hard information boundary: candidate code never receives
        # a world carrying task ground truth or evaluator objects.
        world = AppWorld(
            task_id=args.task_id,
            experiment_name=args.experiment_name,
            max_interactions=args.max_interactions,
            load_ground_truth=False,
            random_seed=args.seed,
        )
        public_task = {
            "task_id": args.task_id,
            "instruction": str(world.task.instruction),
            "app_descriptions": dict(world.task.app_descriptions),
        }
        (out / "public_task.json").write_text(
            json.dumps(public_task, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        telemetry = agent.solve(world)
        messages = list(telemetry.get("transcript") or [])
        (out / "transcript.json").write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out / "transcript.txt").write_text(
            _transcript_text(messages), encoding="utf-8"
        )
        row = {
            "task_id": args.task_id,
            "experiment_name": args.experiment_name,
            "seed": args.seed,
            "steps": telemetry.get("steps"),
            "prompt_tokens": int(telemetry.get("prompt_tokens") or 0),
            "completion_tokens": int(telemetry.get("completion_tokens") or 0),
            "agent_error": telemetry.get("error"),
            "task_completed": bool(world.task_completed()),
            "seconds": round(time.monotonic() - started, 3),
            "error": None,
        }
    except Exception as exc:
        row = {
            "task_id": args.task_id,
            "experiment_name": args.experiment_name,
            "seed": args.seed,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.monotonic() - started, 3),
        }
    finally:
        if world is not None:
            log_dir = Path(str(world.output_logs_directory))
            try:
                world.close()
            except Exception as exc:
                row.setdefault("close_error", f"{type(exc).__name__}: {exc}")
            for name in ("environment_io.md", "api_calls.jsonl"):
                source = log_dir / name
                if source.is_file():
                    shutil.copy2(source, out / name)
    (out / "runtime.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if not row.get("error") else 1


def evaluate_candidate(args: argparse.Namespace) -> int:
    from appworld.evaluator import evaluate_task

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        tracker = evaluate_task(
            task_id=args.task_id,
            experiment_name=args.experiment_name,
            suppress_errors=True,
            save_report=True,
        )
        details = tracker.to_dict()
        row = {
            "task_id": args.task_id,
            "experiment_name": args.experiment_name,
            "success": bool(getattr(tracker, "success", False)),
            "pass_count": int(getattr(tracker, "pass_count", 0) or 0),
            "total_count": int(getattr(tracker, "total_count", 0) or 0),
            "tracker": details,
            "seconds": round(time.monotonic() - started, 3),
            "error": None,
        }
    except Exception as exc:
        row = {
            "task_id": args.task_id,
            "experiment_name": args.experiment_name,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.monotonic() - started, 3),
        }
    (out / "sealed_evaluation.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return 0 if not row.get("error") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "evaluate"))
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agent-path", type=Path)
    parser.add_argument("--max-interactions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "run":
        if args.agent_path is None:
            parser.error("--agent-path is required for run")
        return run_candidate(args)
    return evaluate_candidate(args)


if __name__ == "__main__":
    raise SystemExit(main())
