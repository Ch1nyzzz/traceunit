from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from traceunit.benchmarks import build_benchmark
from traceunit.benchmarks.pools import load_benchmark_plan
from traceunit.config import load_config
from traceunit.final_evaluation import FinalEvaluationRunner
from traceunit.io import write_json
from traceunit.optimizer import OptimizationLoop
from traceunit.proxy_analysis import analyze_proxy
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
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--run-dir", type=Path, required=True)
    packet = sub.add_parser("validate-packet")
    packet.add_argument("--bundle", type=Path, required=True)
    proxy = sub.add_parser("analyze-proxy")
    proxy.add_argument("--run-dir", type=Path, action="append", required=True)
    proxy.add_argument("--output", type=Path)
    proxy.add_argument("--min-train-examples", type=int, default=4)
    proxy.add_argument("--min-category-support", type=int, default=2)
    proxy.add_argument("--l2", type=float, default=1.0)
    proxy.add_argument(
        "--skip-below",
        type=float,
        action="append",
        dest="selection_thresholds",
        help=(
            "proxy probability below which a full natural-task evaluation would be "
            "skipped; may be repeated"
        ),
    )
    proxy.add_argument("--audit-rate", type=float, default=0.1)
    proxy.add_argument("--alignment-bins", type=int, default=10)
    return parser


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
    if args.command == "analyze-proxy":
        report = analyze_proxy(
            args.run_dir,
            min_train_examples=args.min_train_examples,
            min_category_support=args.min_category_support,
            l2=args.l2,
            selection_thresholds=(
                tuple(args.selection_thresholds)
                if args.selection_thresholds
                else (0.1, 0.25, 0.5)
            ),
            audit_rate=args.audit_rate,
            alignment_bins=args.alignment_bins,
        )
        if args.output is not None:
            write_json(args.output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    config = load_config(args.config)
    if args.command == "validate-config":
        print(json.dumps(asdict(config), indent=2, ensure_ascii=False, default=str))
        return 0
    if args.command == "prepare":
        adapter = build_benchmark(config.benchmark)
        config.loop.run_dir.mkdir(parents=True, exist_ok=True)
        plan = adapter.prepare(config.loop.run_dir)
        print(
            json.dumps(
                {
                    "benchmark": adapter.name,
                    "run_dir": str(config.loop.run_dir),
                    "baseline_source": str(adapter.baseline_source()),
                    "benchmark_plan_sha256": plan.plan_sha256,
                    "calibration_shards": len(plan.calibration),
                    "status": "prepared",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "final-evaluate":
        store = RunStore(config.loop.run_dir)
        state = store.load_state()
        if state is None:
            raise SystemExit(f"no completed search under {config.loop.run_dir}")
        adapter = build_benchmark(config.benchmark)
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
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    summary = OptimizationLoop(config).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
