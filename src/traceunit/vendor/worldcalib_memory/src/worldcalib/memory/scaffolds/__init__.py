"""Memory scaffold registry (LOCOMO + LongMemEval)."""

from __future__ import annotations

from worldcalib.scaffolds.base import MemoryScaffold, RetrievalMemoryScaffold, ScaffoldConfig, ScaffoldRun
from worldcalib.memory.scaffolds.memgpt_scaffold import MemGPTSourceScaffold


MEMORY_SCAFFOLD_REGISTRY: dict[str, type[MemoryScaffold]] = {
    MemGPTSourceScaffold.name: MemGPTSourceScaffold,
}

DEFAULT_MEMORY_EVOLUTION_SEED_SCAFFOLDS = (
    MemGPTSourceScaffold.name,
)

DEFAULT_MEMORY_BASELINE_SCAFFOLDS = DEFAULT_MEMORY_EVOLUTION_SEED_SCAFFOLDS

DEFAULT_MEMORY_SCAFFOLDS = DEFAULT_MEMORY_EVOLUTION_SEED_SCAFFOLDS

DEFAULT_MEMORY_SCAFFOLD_TOP_KS = {
    MemGPTSourceScaffold.name: 12,
}


def available_memory_scaffolds() -> tuple[str, ...]:
    return tuple(sorted(MEMORY_SCAFFOLD_REGISTRY))


def build_memory_scaffold(name: str) -> MemoryScaffold:
    try:
        return MEMORY_SCAFFOLD_REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(available_memory_scaffolds())
        raise ValueError(f"unknown scaffold {name!r}; available: {available}") from exc


__all__ = [
    "MemoryScaffold",
    "RetrievalMemoryScaffold",
    "ScaffoldConfig",
    "ScaffoldRun",
    "MEMORY_SCAFFOLD_REGISTRY",
    "DEFAULT_MEMORY_BASELINE_SCAFFOLDS",
    "DEFAULT_MEMORY_EVOLUTION_SEED_SCAFFOLDS",
    "DEFAULT_MEMORY_SCAFFOLDS",
    "DEFAULT_MEMORY_SCAFFOLD_TOP_KS",
    "available_memory_scaffolds",
    "build_memory_scaffold",
]
