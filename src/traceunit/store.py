from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from traceunit.io import append_jsonl, read_json, write_json
from traceunit.models import RunState


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.state_path = self.root / "run_state.json"
        self.events_path = self.root / "events.jsonl"
        self.benchmark_plan_path = self.root / "benchmark_data" / "plan.json"
        self.ontology_path = self.root / "protocol" / "l0_ontology.json"
        self.memory_root = self.root / "ut_memory"
        self.ut_feedback_episodes_path = self.memory_root / "episodes.jsonl"
        self.ut_world_model_path = self.memory_root / "world_model.md"
        self.packet_store_root = self.root / "frozen_packets"
        self.latent_root = self.packet_store_root / "latent"
        self.sealed_root = self.root / "sealed"

    def initialize(
        self,
        *,
        config_snapshot: dict[str, Any],
        capabilities: dict[str, bool] | None = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        names = [
            "iterations",
            "candidates",
            "evaluations",
            "sealed",
            "protocol",
        ]
        enabled = dict(capabilities or {})
        if enabled.get("generated_packets", True):
            names.extend(["test_library", "frozen_packets"])
        if enabled.get("online_ut_memory", True):
            names.extend(["ut_memory", "ut_memory/reflections"])
        for name in names:
            (self.root / name).mkdir(exist_ok=True)
        config_path = self.root / "config.snapshot.json"
        if config_path.exists():
            if read_json(config_path) != _json_safe(config_snapshot):
                raise RuntimeError(
                    "run configuration differs from the frozen config snapshot; "
                    "use a new loop.run_dir"
                )
        else:
            write_json(config_path, config_snapshot)

    def load_state(self) -> RunState | None:
        if not self.state_path.exists():
            return None
        return RunState.from_dict(read_json(self.state_path))

    def save_state(self, state: RunState) -> None:
        write_json(self.state_path, state.to_dict())

    def append_event(self, event: str, **payload: Any) -> None:
        append_jsonl(
            self.events_path,
            {"ts": time.time(), "event": event, **payload},
        )

    def iteration_dir(self, iteration: int) -> Path:
        path = self.root / "iterations" / f"iter_{iteration:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def candidate_dir(self, candidate_id: str) -> Path:
        path = self.root / "candidates" / candidate_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def evaluation_dir(self, candidate_id: str, pool_id: str) -> Path:
        path = self.root / "evaluations" / candidate_id / pool_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value
