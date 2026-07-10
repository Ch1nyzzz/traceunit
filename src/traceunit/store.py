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
        self.calibration_path = self.root / "calibration.json"

    def initialize(self, *, config_snapshot: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in (
            "iterations",
            "candidates",
            "evaluations",
            "partial_archive",
            "sealed",
            "test_library",
        ):
            (self.root / name).mkdir(exist_ok=True)
        config_path = self.root / "config.snapshot.json"
        if not config_path.exists():
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

    def evaluation_dir(self, candidate_id: str, split: str) -> Path:
        path = self.root / "evaluations" / candidate_id / split
        path.mkdir(parents=True, exist_ok=True)
        return path
