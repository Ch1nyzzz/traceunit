from __future__ import annotations

import hashlib
import difflib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=_json_default))
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/") or "__pycache__" in path.parts:
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_source(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )


def source_diff(before: Path, after: Path) -> str:
    """Return a deterministic unified diff without requiring a Git repository."""

    relative_paths = {
        path.relative_to(root)
        for root in (before, after)
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
    }
    chunks: list[str] = []
    for relative in sorted(relative_paths):
        old_path = before / relative
        new_path = after / relative
        old = _text_lines(old_path)
        new = _text_lines(new_path)
        if old is None or new is None:
            raise ValueError(
                f"binary source changes cannot be represented safely: {relative}"
            )
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=(
                    f"a/{relative.as_posix()}" if old_path.is_file() else "/dev/null"
                ),
                tofile=(
                    f"b/{relative.as_posix()}" if new_path.is_file() else "/dev/null"
                ),
                lineterm="",
            )
        )
    return "\n".join(chunks) + ("\n" if chunks else "")


def _text_lines(path: Path) -> list[str] | None:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def safe_relative_path(root: Path, raw: str) -> Path:
    candidate = (root / raw).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"path escapes root: {raw!r}")
    return candidate


def expand_placeholders(value: str, values: Mapping[str, str]) -> str:
    result = value
    for key, replacement in values.items():
        result = result.replace("{" + key + "}", replacement)
    return result


def select_failed_traces(
    rows: Iterable[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    failed = [dict(row) for row in rows if not bool(row.get("passed"))]
    failed.sort(
        key=lambda row: (float(row.get("score") or 0.0), str(row.get("task_id") or ""))
    )
    return failed[:limit] if limit > 0 else failed
