#!/usr/bin/env python3
"""Run or evaluate one mini-SWE-agent SWE-bench instance.

This bridges MemoMemo's per-instance optimizer runner with mini-SWE-agent's
batch-oriented SWE-bench CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# Defaults target DeepSeek's OpenAI-compatible API; every launcher overrides
# --model / --base-url, so these only matter when the runner is invoked bare.
DEFAULT_MODEL = "openai/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
# mini-SWE-agent's built-in text-based LiteLLM class. It routes through LiteLLM
# to the OpenAI-compatible endpoint from --base-url and parses backtick actions,
# matching swebench_backticks.yaml. The previous default
# (``qwen35_miniswe_model.Qwen35TextModel``) was a stale MemoMemo/vLLM-era
# reference to a module that no longer exists in the seed source, so
# get_model_class() crashed at startup and every candidate scored 0/30 until the
# proposer hand-aliased it. See the registry in minisweagent/models/__init__.py.
DEFAULT_MODEL_CLASS = "litellm_textbased"

# Hard ceiling for the agent step limit. The runner no longer forces a fixed
# step limit: the scaffold's own swebench_backticks.yaml value is honored (so
# the proposer can tune it), but the effective limit never exceeds this cap.
MAX_STEP_LIMIT = 250


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "eval"))
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--instance-path", required=True, type=Path)
    parser.add_argument("--patch-path", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--model-class",
        default=DEFAULT_MODEL_CLASS,
        help=(
            "mini-SWE-agent model class key (see the registry in "
            "minisweagent/models/__init__.py). Default "
            f"'{DEFAULT_MODEL_CLASS}' routes through LiteLLM to the --base-url "
            "endpoint with text-based (backtick) action parsing."
        ),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the model API key.",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=0,
        help=(
            "Agent step limit. 0 (default) honors the scaffold's own "
            "swebench_backticks.yaml value so the proposer can tune it; a "
            "positive value overrides it. Either way the effective limit is "
            f"capped at {MAX_STEP_LIMIT}."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature forwarded as model.model_kwargs.temperature "
            "(default 0.0). Reasoning models (e.g. Gemini-3-flash) often require "
            "temperature=1."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Optional reasoning-effort level forwarded to the model as "
            "model.model_kwargs.reasoning_effort (e.g. 'high'). Unset (default) "
            "leaves the provider default untouched."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    instance = json.loads(args.instance_path.read_text(encoding="utf-8"))
    instance_id = str(instance.get("task_id") or instance.get("instance_id") or "").strip()
    if not instance_id:
        raise SystemExit("instance_path is missing task_id/instance_id")

    if args.mode == "run":
        return run_agent(args, root=root, instance_id=instance_id)
    return eval_patch(args, root=root, instance_id=instance_id)


def run_agent(args: argparse.Namespace, *, root: Path, instance_id: str) -> int:
    source_path = args.source_path.resolve()
    output = args.task_dir / "miniswe_run"
    output.mkdir(parents=True, exist_ok=True)
    args.patch_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_paths(
        env.get("PYTHONPATH", ""),
        [str(root), str(args.source_path / "src")],
    )
    env["MSWEA_COST_TRACKING"] = "ignore_errors"

    cmd = [
        "uvx",
        "--from",
        str(source_path),
        "python",
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        "verified",
        "--split",
        "test",
        "--filter",
        f"^{re.escape(instance_id)}$",
        "--output",
        str(output),
        "--workers",
        "1",
        "--model",
        args.model,
        "--model-class",
        args.model_class,
        "--config",
        "swebench_backticks.yaml",
        "--config",
        f"model.model_kwargs.api_base={args.base_url}",
        "--config",
        f"model.model_kwargs.temperature={args.temperature}",
        "--config",
        f"model.model_kwargs.max_tokens={args.max_tokens}",
        "--config",
        "agent.cost_limit=0",
        "--config",
        f"agent.step_limit={_resolve_step_limit(args)}",
        "--redo-existing",
    ]
    if args.reasoning_effort:
        cmd += [
            "--config",
            f"model.model_kwargs.reasoning_effort={args.reasoning_effort}",
        ]
    api_key = args.api_key
    if not api_key and args.api_key_env:
        api_key = env.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is not set")
    if api_key:
        # Let the OpenAI-compatible client read credentials from the environment
        # instead of placing secrets in argv and mini-SWE logs.
        env["OPENAI_API_KEY"] = api_key
    completed = subprocess.run(
        cmd,
        cwd=args.source_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (args.task_dir / "miniswe_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.task_dir / "miniswe_stderr.txt").write_text(completed.stderr, encoding="utf-8")

    patch = _read_patch_from_preds(output / "preds.json", instance_id)
    if not patch:
        patch = _read_patch_from_trajectory(output, instance_id)
    args.patch_path.write_text(patch, encoding="utf-8")
    return completed.returncode


def _read_patch_from_preds(preds_path: Path, instance_id: str) -> str:
    if not preds_path.exists():
        return ""
    try:
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    prediction = preds.get(instance_id) or {}
    if isinstance(prediction, dict):
        return str(prediction.get("model_patch") or "")
    return str(prediction or "")


def _read_patch_from_trajectory(output: Path, instance_id: str) -> str:
    traj_path = output / instance_id / f"{instance_id}.traj.json"
    try:
        payload = json.loads(traj_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    info = payload.get("info") if isinstance(payload, dict) else {}
    if isinstance(info, dict) and info.get("submission"):
        return str(info.get("submission") or "")
    messages = payload.get("messages") if isinstance(payload, dict) else []
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            extra = message.get("extra")
            if isinstance(extra, dict) and extra.get("submission"):
                return str(extra.get("submission") or "")
    return ""


def eval_patch(args: argparse.Namespace, *, root: Path, instance_id: str) -> int:
    args.task_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.task_dir / "single_pred.json"
    pred_path.write_text(
        json.dumps(
            {
                instance_id: {
                    "model_name_or_path": "memomemo_candidate",
                    "instance_id": instance_id,
                    "model_patch": args.patch_path.read_text(encoding="utf-8", errors="ignore"),
                }
            }
        ),
        encoding="utf-8",
    )
    report_id = f"memomemo_{_safe_id(instance_id)}"
    cmd = [
        "uvx",
        "--from",
        "swebench",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        "princeton-nlp/SWE-Bench_Verified",
        "-s",
        "test",
        "-i",
        instance_id,
        "-p",
        str(pred_path),
        "--max_workers",
        "1",
        "--cache_level",
        "instance",
        "--clean",
        "True",
        "-id",
        report_id,
        "--report_dir",
        str(args.task_dir / "eval_report"),
    ]
    completed = subprocess.run(
        cmd,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    (args.task_dir / "official_eval_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (args.task_dir / "official_eval_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    report = _find_report(root, args.task_dir, report_id)
    if report is None:
        return 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    return 0 if int(payload.get("resolved_instances") or 0) == 1 else 1


def _find_report(root: Path, task_dir: Path, report_id: str) -> Path | None:
    candidates = [
        *task_dir.glob(f"**/*.{report_id}.json"),
        *root.glob(f"*.{report_id}.json"),
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _prepend_paths(existing: str, paths: list[str]) -> str:
    values = [item for item in paths if item]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def _yaml_step_limit(source_path: Path) -> int:
    """Return ``agent.step_limit`` from the scaffold's swebench_backticks.yaml.

    Uses a line regex so this script needs no YAML dependency of its own.
    Returns 0 when the file or the key is absent.
    """
    yaml_path = (
        source_path
        / "src"
        / "minisweagent"
        / "config"
        / "benchmarks"
        / "swebench_backticks.yaml"
    )
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    match = re.search(r"(?m)^[ \t]*step_limit:[ \t]*(\d+)", text)
    return int(match.group(1)) if match else 0


def _resolve_step_limit(args: argparse.Namespace) -> int:
    """Resolve the effective agent step limit.

    The runner does not force a fixed step limit. With ``--step-limit 0`` (the
    default) the scaffold's own swebench_backticks.yaml value is used, so the
    proposer can tune it; a positive ``--step-limit`` overrides that. The
    result is always capped at ``MAX_STEP_LIMIT`` and never below 1.
    """
    requested = args.step_limit
    if requested <= 0:
        requested = _yaml_step_limit(args.source_path)
    if requested <= 0:
        requested = MAX_STEP_LIMIT
    return min(requested, MAX_STEP_LIMIT)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    sys.exit(main())
