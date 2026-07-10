from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from traceunit.models import BenchmarkEvaluation, BenchmarkPlan, PoolSliceRef


class BenchmarkAdapter(ABC):
    name: str

    def preflight(self) -> None:
        """Fail before expensive evaluation when runtime prerequisites are absent."""

    def bind_plan(self, plan: BenchmarkPlan) -> None:
        """Bind an already-frozen benchmark plan without reopening source data."""

        if plan.benchmark != self.name:
            raise ValueError(
                f"benchmark plan is for {plan.benchmark!r}, not {self.name!r}"
            )
        self._plan = plan

    @abstractmethod
    def prepare(self, work_dir: Path) -> BenchmarkPlan:
        """Freeze disjoint search/calibration/final pools and return their plan."""

    @abstractmethod
    def baseline_source(self) -> Path:
        """Return the clean editable agent source tree."""

    @abstractmethod
    def context(self) -> str:
        """Return the benchmark-specific contract shown to code agents."""

    @abstractmethod
    def evaluate(
        self,
        *,
        source: Path,
        candidate_id: str,
        pool: PoolSliceRef,
        out_dir: Path,
    ) -> BenchmarkEvaluation:
        """Evaluate one immutable, content-bound pool slice."""

    @abstractmethod
    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        """Check that a candidate source snapshot is loadable."""

    @abstractmethod
    def policy_violations(self, source: Path, diff_text: str) -> list[str]:
        """Detect evaluator access and obvious task-specific reward hacks."""
