"""Native HLE prediction worker (runs the editable scaffold; never sees gold).

Invoked as a subprocess, one call per candidate/pool-slice evaluation, so a
candidate's edited ``hle_qa`` scaffold is imported in a clean interpreter. It
owns the frozen solver client and produces predictions only. Grading against the
gold answer happens back in the host adapter's LLM judge, so the gold answer
never enters this process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from traceunit.benchmarks.openai_chat import ChatError, chat_completion


def _make_call_model(model: Mapping[str, Any], api_key: str, dry_run: bool):
    base_url = str(model.get("base_url") or "")
    model_name = str(model.get("model") or "")
    timeout_s = int(model.get("timeout_s") or 300)

    def call_model(
        *, messages, max_tokens: int, temperature: float = 0.0
    ) -> dict[str, Any]:
        if dry_run:
            # No API traffic: let the scaffold exercise its full loop cheaply.
            return {
                "content": "reasoning omitted\nFINAL ANSWER: dry-run",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        return chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            timeout_s=timeout_s,
        )

    return call_model


def run_from_spec(spec_path: Path, out_path: Path) -> int:
    import os

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    source = spec.get("source")
    if source:
        source_path = str(Path(str(source)).expanduser().resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    from hle_qa import answer_question  # candidate's editable scaffold

    model = dict(spec.get("model") or {})
    dry_run = bool(spec.get("dry_run", False))
    api_key = os.environ.get(str(model.get("api_key_env") or ""), "")
    if not dry_run and not api_key:
        raise RuntimeError(
            f"HLE solver key is missing: {model.get('api_key_env')}"
        )
    call_model = _make_call_model(model, api_key, dry_run)
    max_output_tokens = int(model.get("max_output_tokens") or 4096)

    questions = json.loads(Path(str(spec["questions_path"])).read_text(encoding="utf-8"))
    predictions: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question.get("id") or "")
        answer_type = str(question.get("answer_type") or "exactMatch")
        record: dict[str, Any] = {
            "id": qid,
            "prediction": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "raw": "",
            "error": "",
        }
        try:
            result = answer_question(
                question=str(question.get("question") or ""),
                answer_type=answer_type,
                call_model=call_model,
                max_output_tokens=max_output_tokens,
            )
            record["prediction"] = str(result.get("prediction") or "")
            record["prompt_tokens"] = int(result.get("prompt_tokens") or 0)
            record["completion_tokens"] = int(result.get("completion_tokens") or 0)
            record["raw"] = str(result.get("raw") or "")[:8000]
        except ChatError as exc:
            record["error"] = f"solver_error: {exc}"[:2000]
        except Exception as exc:  # noqa: BLE001 - scaffold bugs are per-task
            record["error"] = f"scaffold_error: {type(exc).__name__}: {exc}"[:2000]
        predictions.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"predictions": predictions}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Produce HLE predictions from a JSON spec.")
    run.add_argument("--spec", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_from_spec(args.spec, args.out)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
