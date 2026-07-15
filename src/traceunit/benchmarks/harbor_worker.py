"""Standalone Harbor job runner for Harbor-hosted benchmarks (Terminal-Bench 2.0).

This module is executed as a subprocess, one invocation per candidate/pool-slice
evaluation, so a candidate's edited scaffold is imported in a fresh interpreter
and never collides with another candidate through Python's module cache. It reads
a portable JSON spec, runs a Harbor ``Job`` against the requested dataset slice
with the candidate's editable agent, and writes a benchmark-agnostic result
envelope that :mod:`traceunit.benchmarks.native_eval` normalizes into a
``BenchmarkEvaluation``.

It depends only on the standard library and the installed ``harbor`` package --
never on ``traceunit`` -- so it stays runnable under any interpreter that has
Harbor available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping


def _reward_to_score(rewards: Mapping[str, Any] | None) -> float:
    """Collapse Harbor's reward dict to a single [0, 1]-style scalar.

    Terminal-Bench tasks emit a single ``{"reward": <float>}`` where 1.0 means the
    verifier's tests all passed. Multi-key reward dicts are averaged so a task that
    reports several sub-metrics still yields one comparable number.
    """

    if not rewards:
        return 0.0
    values = [float(value) for value in rewards.values() if _is_number(value)]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _agent_usage(context: Any) -> dict[str, Any]:
    """Read token/cost usage the agent recorded on its Harbor ``AgentContext``."""

    prompt = getattr(context, "n_input_tokens", None)
    completion = getattr(context, "n_output_tokens", None)
    cache = getattr(context, "n_cache_tokens", None)
    cost = getattr(context, "cost_usd", None)
    prompt_tokens = int(prompt or 0)
    completion_tokens = int(completion or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_tokens": int(cache or 0),
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": float(cost or 0.0),
    }


def _build_job_config(spec: Mapping[str, Any]):
    """Translate the portable spec into a Harbor ``JobConfig``."""

    from harbor import DatasetConfig, EnvironmentType, JobConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        VerifierConfig,
    )

    dataset = dict(spec.get("dataset") or {})
    dataset_kwargs: dict[str, Any] = {}
    if dataset.get("path"):
        dataset_kwargs["path"] = Path(str(dataset["path"])).expanduser().resolve()
    else:
        dataset_kwargs["name"] = str(dataset["name"])
        if dataset.get("version"):
            dataset_kwargs["version"] = str(dataset["version"])
    task_names = dataset.get("task_names")
    if task_names:
        dataset_kwargs["task_names"] = [str(name) for name in task_names]
    if dataset.get("exclude_task_names"):
        dataset_kwargs["exclude_task_names"] = [
            str(name) for name in dataset["exclude_task_names"]
        ]
    if dataset.get("n_tasks") is not None:
        dataset_kwargs["n_tasks"] = int(dataset["n_tasks"])
    if dataset.get("registry_url"):
        dataset_kwargs["registry_url"] = str(dataset["registry_url"])
    if dataset.get("registry_path"):
        dataset_kwargs["registry_path"] = (
            Path(str(dataset["registry_path"])).expanduser().resolve()
        )
    if dataset.get("download_dir"):
        dataset_kwargs["download_dir"] = (
            Path(str(dataset["download_dir"])).expanduser().resolve()
        )

    model = dict(spec.get("model") or {})
    agent_spec = dict(spec.get("agent") or {})
    agent_kwargs: dict[str, Any] = {
        "parser_name": str(model.get("parser_name") or "json"),
    }
    if model.get("api_base"):
        agent_kwargs["api_base"] = str(model["api_base"])
    if model.get("temperature") is not None:
        agent_kwargs["temperature"] = float(model["temperature"])
    if model.get("reasoning_effort"):
        agent_kwargs["reasoning_effort"] = str(model["reasoning_effort"])
    if model.get("max_turns") is not None:
        agent_kwargs["max_turns"] = int(model["max_turns"])
    if isinstance(model.get("model_info"), Mapping):
        agent_kwargs["model_info"] = dict(model["model_info"])

    agent_config_kwargs: dict[str, Any] = {}
    if agent_spec.get("import_path"):
        # A custom editable scaffold: forward the model and Terminus kwargs.
        agent_config_kwargs["import_path"] = str(agent_spec["import_path"])
        agent_config_kwargs["model_name"] = model.get("model_name")
        agent_config_kwargs["kwargs"] = agent_kwargs
    else:
        # A built-in Harbor agent (e.g. oracle) drives its own reference
        # solution and must not receive Terminus-specific constructor kwargs.
        agent_config_kwargs["name"] = str(agent_spec.get("name") or "oracle")
        if model.get("model_name"):
            agent_config_kwargs["model_name"] = model.get("model_name")
    run = dict(spec.get("run") or {})
    if run.get("agent_timeout_sec") is not None:
        agent_config_kwargs["override_timeout_sec"] = float(run["agent_timeout_sec"])

    environment = EnvironmentConfig(
        type=EnvironmentType.DOCKER,
        delete=bool(run.get("delete_env", True)),
    )
    verifier = VerifierConfig(disable=bool(run.get("disable_verifier", False)))

    return JobConfig(
        job_name=str(spec.get("job_name") or "traceunit-harbor"),
        jobs_dir=Path(str(run.get("jobs_dir") or "./jobs")).expanduser().resolve(),
        n_concurrent_trials=max(1, int(run.get("n_concurrent", 4))),
        n_attempts=max(1, int(run.get("n_attempts", 1))),
        quiet=bool(run.get("quiet", True)),
        datasets=[DatasetConfig(**dataset_kwargs)],
        agents=[AgentConfig(**agent_config_kwargs)],
        environment=environment,
        verifier=verifier,
    )


async def _run_job(spec: Mapping[str, Any]) -> dict[str, Any]:
    from harbor import Job

    config = _build_job_config(spec)
    job = await Job.create(config)
    result = await job.run()

    tasks: list[dict[str, Any]] = []
    passed_count = 0
    score_sum = 0.0
    total_tokens = 0
    total_cost = 0.0
    for trial in result.trial_results:
        verifier_result = getattr(trial, "verifier_result", None)
        rewards = getattr(verifier_result, "rewards", None) if verifier_result else None
        exception_info = getattr(trial, "exception_info", None)
        score = _reward_to_score(rewards)
        passed = bool(rewards) and score >= 1.0
        if exception_info is not None:
            status = "agent_error"
        elif verifier_result is None:
            status = "verifier_missing"
        elif passed:
            status = "resolved"
        else:
            status = "unresolved"
        usage = _agent_usage(getattr(trial, "agent_result", None))
        total_tokens += usage["total_tokens"]
        total_cost += usage["cost_usd"]
        passed_count += 1 if passed else 0
        score_sum += score
        error = ""
        if exception_info is not None:
            error = str(
                getattr(exception_info, "message", None)
                or getattr(exception_info, "type", None)
                or exception_info
            )[:4000]
        tasks.append(
            {
                "task_id": str(getattr(trial, "task_name", "") or ""),
                "trial_name": str(getattr(trial, "trial_name", "") or ""),
                "source": str(getattr(trial, "source", "") or ""),
                "score": score,
                "passed": passed,
                "status": status,
                "reward": dict(rewards) if isinstance(rewards, Mapping) else None,
                "error": error,
                **usage,
            }
        )

    count = len(tasks)
    stats = getattr(result, "stats", None)
    envelope = {
        "candidate": {
            "candidate_id": str(spec.get("candidate_id") or ""),
            "count": count,
            "passrate": (passed_count / count) if count else 0.0,
            "average_score": (score_sum / count) if count else 0.0,
            "token_consuming": total_tokens,
            "monetary_cost": total_cost,
            "config": {
                "dataset": spec.get("dataset"),
                "model": {
                    key: value
                    for key, value in dict(spec.get("model") or {}).items()
                    if key != "api_key_env"
                },
                "agent": spec.get("agent"),
            },
        },
        "tasks": tasks,
        "job_stats": {
            "n_trials": int(getattr(stats, "n_trials", count) or count),
            "n_errors": int(getattr(stats, "n_errors", 0) or 0),
        },
    }
    return envelope


def _prepare_environment(spec: Mapping[str, Any]) -> None:
    """Put the candidate scaffold on the path and expose the solver API key."""

    source = spec.get("source")
    if source:
        source_path = str(Path(str(source)).expanduser().resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
    model = dict(spec.get("model") or {})
    api_key_env = str(model.get("api_key_env") or "")
    if api_key_env:
        secret = os.environ.get(api_key_env)
        if secret and not os.environ.get("OPENAI_API_KEY"):
            # Terminus drives litellm's OpenAI-compatible path; mirror the solver
            # key into the variable litellm reads unless one is already set.
            os.environ["OPENAI_API_KEY"] = secret


def run_from_spec(spec_path: Path, out_path: Path) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    _prepare_environment(spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        envelope = asyncio.run(_run_job(spec))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the parent
        envelope = {
            "candidate": {
                "candidate_id": str(spec.get("candidate_id") or ""),
                "count": 0,
                "passrate": 0.0,
                "average_score": 0.0,
                "token_consuming": 0,
                "monetary_cost": 0.0,
                "config": {},
            },
            "tasks": [],
            "job_stats": {"n_trials": 0, "n_errors": 0},
            "worker_error": f"{type(exc).__name__}: {exc}",
            "worker_traceback": traceback.format_exc(),
        }
        out_path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 1
    out_path.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run one Harbor job from a JSON spec.")
    run.add_argument("--spec", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_from_spec(args.spec, args.out)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
