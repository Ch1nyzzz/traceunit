from __future__ import annotations

from traceunit.benchmarks.appworld import AppWorldAdapter
from traceunit.benchmarks.base import BenchmarkAdapter
from traceunit.benchmarks.swebench import SwebenchVerifiedAdapter
from traceunit.config import BenchmarkConfig


def build_benchmark(config: BenchmarkConfig) -> BenchmarkAdapter:
    if config.name == "swebench_verified":
        return SwebenchVerifiedAdapter(config)
    if config.name == "appworld":
        return AppWorldAdapter(config)
    raise ValueError(f"unsupported benchmark: {config.name}")
