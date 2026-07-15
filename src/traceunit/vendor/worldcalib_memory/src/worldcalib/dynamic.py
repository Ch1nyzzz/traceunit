"""Dynamic candidate loading for Claude-proposed memory scaffolds."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

from worldcalib.memory.scaffolds import build_memory_scaffold
from worldcalib.scaffolds.base import MemoryScaffold


SOURCE_PROJECT_PATH_KEYS = (
    "source_project_path",
    "project_source_path",
    "memomemo_source_path",
)

SOURCE_SCAFFOLD_CLASSES = {
    "memgpt_source": (
        "worldcalib.memory.scaffolds.memgpt_scaffold",
        "MemGPTSourceScaffold",
    ),
}


def load_candidate_scaffold(candidate: dict[str, Any], *, project_root: Path) -> MemoryScaffold:
    """Instantiate a memory scaffold from pending_eval candidate metadata."""

    if candidate.get("kind") == "agent":
        # Agent scaffolds are agentrl BaseClient subclasses, loaded separately.
        from worldcalib.agentic.backends.agentbench.dynamic import (
            load_candidate_agent_scaffold,
        )

        return load_candidate_agent_scaffold(candidate, project_root=project_root)

    if candidate.get("kind") == "tau2_agent":
        # tau2 scaffolds are LLMAgent factories, loaded separately (agentrl-free).
        from worldcalib.agentic.backends.tau2.dynamic import (
            load_candidate_tau2_scaffold,
        )

        return load_candidate_tau2_scaffold(candidate, project_root=project_root)

    if candidate.get("kind") == "gaia_agent":
        # GAIA scaffolds are FC-loop policies (solve_task), loaded separately.
        from worldcalib.agentic.backends.gaia.dynamic import (
            load_candidate_gaia_scaffold,
        )

        return load_candidate_gaia_scaffold(candidate, project_root=project_root)

    if candidate.get("kind") == "spider2_agent":
        # Spider2 scaffolds are single-shot text-to-SQL policies (solve_task),
        # loaded separately.
        from worldcalib.agentic.backends.spider2.dynamic import (
            load_candidate_spider2_scaffold,
        )

        return load_candidate_spider2_scaffold(candidate, project_root=project_root)

    if candidate.get("kind") == "arc_solver":
        # ARC solvers are single-shot ArcScaffolds, loaded separately (agentrl-free).
        from worldcalib.reasoning.arc_dynamic import (
            load_candidate_arc_scaffold,
        )

        return load_candidate_arc_scaffold(candidate, project_root=project_root)

    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    candidate_root = _candidate_root(candidate, project_root=project_root)
    if candidate_root is not None:
        root_path = str(candidate_root)
        if root_path not in sys.path:
            sys.path.insert(0, root_path)

    source_project_path = _source_project_path(candidate, project_root=project_root)
    scaffold_name = candidate.get("scaffold_name") or candidate.get("seed_name")
    if scaffold_name:
        if source_project_path is not None:
            return _load_source_project_scaffold(
                str(scaffold_name),
                source_project_path=source_project_path,
            )
        return build_memory_scaffold(str(scaffold_name))

    module_path = str(candidate.get("module_path") or "").strip()
    module_name = str(candidate.get("module") or "").strip()
    class_name = str(candidate.get("class") or "").strip()
    factory_name = str(candidate.get("factory") or "").strip()
    if not module_name and not module_path:
        raise ValueError("candidate must provide `module`, `module_path`, or `scaffold_name`")

    importlib.invalidate_caches()
    context = (
        _isolated_memomemo_project(source_project_path)
        if source_project_path is not None
        else nullcontext()
    )
    with context:
        if module_path:
            module = _load_module_path(module_path, project_root=project_root)
        else:
            if candidate_root is not None and module_name.startswith("worldcalib.generated."):
                module_name = module_name.removeprefix("worldcalib.generated.")
            if candidate_root is not None and module_name in sys.modules:
                del sys.modules[module_name]
            module = importlib.import_module(module_name)
            module = importlib.reload(module)

        if class_name:
            cls = getattr(module, class_name)
            scaffold = cls()
        elif factory_name:
            scaffold = getattr(module, factory_name)()
        elif hasattr(module, "build_scaffold"):
            scaffold = module.build_scaffold()
        elif hasattr(module, "SCAFFOLD_CLASS"):
            scaffold = module.SCAFFOLD_CLASS()
        else:
            raise ValueError(
                f"{module_name} must expose class/factory/build_scaffold/SCAFFOLD_CLASS"
            )

    if not isinstance(scaffold, MemoryScaffold):
        required = ("build", "answer", "name")
        if not all(hasattr(scaffold, attr) for attr in required):
            raise TypeError(
                f"{module_name} did not produce a MemoryScaffold-compatible object"
            )
    return scaffold


def _candidate_root(candidate: dict[str, Any], *, project_root: Path) -> Path | None:
    value = candidate.get("candidate_root") or candidate.get("generated_dir")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _source_project_path(candidate: dict[str, Any], *, project_root: Path) -> Path | None:
    extra = candidate.get("extra") if isinstance(candidate.get("extra"), dict) else {}
    for key in SOURCE_PROJECT_PATH_KEYS:
        value = candidate.get(key) or extra.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = project_root / path
            return _source_project_src_root(path)
    return None


def _source_project_src_root(path: Path) -> Path:
    candidates = [
        path,
        path / "src",
        path / "project_source",
        path / "project_source" / "src",
    ]
    for item in candidates:
        if (item / "worldcalib").is_dir():
            return item
    raise FileNotFoundError(
        f"source project path must contain worldcalib package or project_source/src/worldcalib: {path}"
    )


def _load_source_project_scaffold(
    scaffold_name: str,
    *,
    source_project_path: Path,
) -> MemoryScaffold:
    try:
        module_name, class_name = SOURCE_SCAFFOLD_CLASSES[scaffold_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SOURCE_SCAFFOLD_CLASSES))
        raise ValueError(
            f"source_project_path is only supported for: {supported}; got {scaffold_name!r}"
        ) from exc

    with _isolated_memomemo_project(source_project_path):
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        scaffold = cls()

    if not isinstance(scaffold, MemoryScaffold):
        required = ("build", "answer", "name")
        if not all(hasattr(scaffold, attr) for attr in required):
            raise TypeError(
                f"{module_name}.{class_name} did not produce a MemoryScaffold-compatible object"
            )
    return scaffold


@contextmanager
def _isolated_memomemo_project(src_root: Path) -> Iterator[None]:
    """Temporarily import worldcalib modules from a copied project source tree."""

    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "worldcalib" or name.startswith("worldcalib.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    source_text = str(src_root)
    inserted = False
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
        inserted = True
    try:
        yield
    finally:
        for name in [
            item for item in list(sys.modules) if item == "worldcalib" or item.startswith("worldcalib.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if inserted:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass


def _load_module_path(module_path: str, *, project_root: Path) -> object:
    path = Path(module_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"candidate module_path does not exist: {path}")

    module_name = f"_memomemo_candidate_{abs(hash(path.resolve()))}"
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module_path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
