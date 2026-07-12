from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from traceunit.benchmarks import build_benchmark
from traceunit.benchmarks.pools import load_benchmark_plan
from traceunit.config import ProjectConfig, load_config
from traceunit.final_evaluation import FinalEvaluationRunner
from traceunit.io import read_json, write_json
from traceunit.ontology import freeze_ontology
from traceunit.optimizer import OptimizationLoop
from traceunit.store import RunStore
from traceunit.tests_runtime import load_test_packet, verify_frozen_packet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traceunit",
        description="Trace-conditioned causal proxy-test optimization loop",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("optimize", "prepare", "validate-config", "final-evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "optimize":
            command.add_argument(
                "--no-final",
                action="store_true",
                help="do not run the sealed final evaluation after search completes",
            )
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--run-dir", type=Path, required=True)
    packet = sub.add_parser("validate-packet")
    packet.add_argument("--bundle", type=Path, required=True)
    return parser


def _final_evaluate(config: ProjectConfig) -> dict:
    """Seal and run the final evaluation for a completed search run."""

    store = RunStore(config.loop.run_dir)
    state = store.load_state()
    if state is None:
        raise SystemExit(f"no completed search under {config.loop.run_dir}")
    adapter = build_benchmark(config.benchmark)
    freeze_ontology(store.ontology_path)
    if not store.benchmark_plan_path.is_file():
        raise SystemExit(
            f"frozen benchmark plan is missing: {store.benchmark_plan_path}"
        )
    plan = load_benchmark_plan(store.benchmark_plan_path)
    adapter.bind_plan(plan)
    adapter.preflight()
    runner = FinalEvaluationRunner(
        store=store,
        benchmark=adapter,
        benchmark_plan=plan,
    )
    report = runner.run(runner.seal(state))
    summary_path = config.loop.run_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        summary["final_evaluation"] = "opened"
        write_json(summary_path, summary)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        summary = args.run_dir / "summary.json"
        state = args.run_dir / "run_state.json"
        path = summary if summary.is_file() else state
        if not path.is_file():
            raise SystemExit(f"no run state found under {args.run_dir}")
        print(path.read_text(encoding="utf-8"))
        return 0
    if args.command == "validate-packet":
        packet = load_test_packet(args.bundle)
        payload = packet.to_dict()
        payload["hash_valid"] = (
            verify_frozen_packet(args.bundle, packet) if packet.content_sha256 else None
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    config = load_config(args.config)
    if args.command == "validate-config":
        print(json.dumps(asdict(config), indent=2, ensure_ascii=False, default=str))
        return 0
    if args.command == "prepare":
        adapter = build_benchmark(config.benchmark)
        config.loop.run_dir.mkdir(parents=True, exist_ok=True)
        freeze_ontology(config.loop.run_dir / "protocol" / "l0_ontology.json")
        plan = adapter.prepare(config.loop.run_dir)
        print(
            json.dumps(
                {
                    "benchmark": adapter.name,
                    "run_dir": str(config.loop.run_dir),
                    "baseline_source": str(adapter.baseline_source()),
                    "benchmark_plan_sha256": plan.plan_sha256,
                    "status": "prepared",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "final-evaluate":
        report = _final_evaluate(config)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    summary = OptimizationLoop(config).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.no_final and summary.get("status") in {"completed", "converged"}:
        report = _final_evaluate(config)
        print(json.dumps({"final_report": report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
