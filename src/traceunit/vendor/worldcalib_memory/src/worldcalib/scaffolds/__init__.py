"""Shared memory scaffold base types.

The concrete memory scaffold registry (LOCOMO + LongMemEval) now lives in
``worldcalib.memory.scaffolds``; this package only re-exports the
backend-agnostic base classes (``MemoryScaffold`` / ``RetrievalMemoryScaffold``
/ ``ScaffoldConfig`` / ``ScaffoldRun``) that every backend — memory, agentic,
and reasoning — shares from ``worldcalib.scaffolds.base``.
"""

from __future__ import annotations

from worldcalib.scaffolds.base import (
    MemoryScaffold,
    RetrievalMemoryScaffold,
    ScaffoldConfig,
    ScaffoldRun,
)


__all__ = [
    "MemoryScaffold",
    "RetrievalMemoryScaffold",
    "ScaffoldConfig",
    "ScaffoldRun",
]
