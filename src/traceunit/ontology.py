from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from traceunit.io import read_json, write_json
from traceunit.models import UnitFamily


_REGISTRY = json.loads(
    files("traceunit").joinpath("data/l0_ontology.json").read_text(encoding="utf-8")
)
if not isinstance(_REGISTRY, dict) or not isinstance(_REGISTRY.get("families"), dict):
    raise RuntimeError("packaged L0 ontology registry is malformed")

ONTOLOGY_ID = str(_REGISTRY["ontology_id"])
ONTOLOGY_VERSION = str(_REGISTRY["version"])
FAMILY_DEFINITIONS: dict[str, str] = {
    str(key): str(value) for key, value in _REGISTRY["families"].items()
}

if set(FAMILY_DEFINITIONS) != {item.value for item in UnitFamily}:
    raise RuntimeError(
        "UnitFamily enum and packaged L0 ontology registry have diverged"
    )


def ontology_document() -> dict[str, Any]:
    return {
        "ontology_id": ONTOLOGY_ID,
        "version": ONTOLOGY_VERSION,
        "families": dict(FAMILY_DEFINITIONS),
    }


def ontology_sha256() -> str:
    encoded = json.dumps(
        ontology_document(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ontology_ref() -> dict[str, str]:
    return {
        "ontology_id": ONTOLOGY_ID,
        "version": ONTOLOGY_VERSION,
        "sha256": ontology_sha256(),
    }


def freeze_ontology(path: Path) -> None:
    payload = {**ontology_document(), "sha256": ontology_sha256()}
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(
                "L0 ontology differs from the frozen run snapshot; use a new run directory"
            )
        return
    write_json(path, payload)


def validate_ontology_ref(value: object) -> bool:
    return isinstance(value, dict) and value == ontology_ref()


def prompt_definitions() -> str:
    return "\n".join(
        f"- {family}: {definition}" for family, definition in FAMILY_DEFINITIONS.items()
    )
