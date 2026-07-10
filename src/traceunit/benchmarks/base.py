from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from traceunit.models import BenchmarkEvaluation


class BenchmarkAdapter(ABC):
    name: str

    def preflight(self) -> None:
        """Fail before expensive evaluation when runtime prerequisites are absent."""

    @abstractmethod
    def prepare(self, work_dir: Path) -> None:
        """Prepare immutable task pools and validate runtime assets."""

    @abstractmethod
    def seed_source(self) -> Path:
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
        split: str,
        out_dir: Path,
        limit_override: int | None = None,
    ) -> BenchmarkEvaluation:
        """Run a natural-task pool and normalize the full trajectory artifacts."""

    @abstractmethod
    def smoke_test(self, source: Path, out_dir: Path) -> tuple[bool, str]:
        """Check that a candidate source snapshot is loadable."""

    @abstractmethod
    def policy_violations(self, source: Path, diff_text: str) -> list[str]:
        """Detect evaluator access and obvious task-specific reward hacks."""
