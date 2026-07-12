"""Append-only UT-design world model, WorldCalib style.

The harness owns no schema, digest, sanitization, or fallback text. It seeds
the file header, copies the file into the Test Author's workspace, and copies
whatever the author wrote back. The author is instructed to read the file
first, distill the previous iteration's unit/search evidence - especially any
mismatch - and append a ``## iter_NNN distill`` section. The content is
entirely the agent's; a missing update is recorded as an event, never
replaced with template text.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

WORLD_MODEL_FILENAME = "ut_design_world_model.md"

WORLD_MODEL_HEADER = """\
# UT design world model

Append-only. The Test Author must read this file before designing the next
TestPacket, distill what the previous iteration showed about UT design, then
append a new `## iter_NNN distill` section. Never rewrite or delete prior
entries.

Each iteration produces: the frozen packet and its per-case unit results on
incumbent and candidate, the paired per-task search outcomes, the raw traces
on both sides, and the decision. The unit tests exist as a cheap proxy for
the search distribution, so the question every entry must answer is: did the
tests measure what the search tasks actually demand? When the unit verdict
and paired search disagreed, name what the tests measured that the search
distribution does not (or the reverse), and state how the next packet's tests
will be designed differently. Keep entries to concrete observations and
falsifiable design rules - no generic advice.
"""

_DISTILL_RE = re.compile(r"^## iter_\d+", re.MULTILINE)


class WorldModel:
    """One markdown file; the harness only seeds, stages, and copies back."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def ensure(self) -> None:
        if not self.path.is_file():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(WORLD_MODEL_HEADER, encoding="utf-8")

    def stage_into(self, workspace: Path) -> Path:
        self.ensure()
        workspace.mkdir(parents=True, exist_ok=True)
        staged = workspace / WORLD_MODEL_FILENAME
        shutil.copy2(self.path, staged)
        return staged

    def commit_from(self, workspace: Path) -> bool:
        """Copy the author's version back; True when the file actually grew."""

        staged = workspace / WORLD_MODEL_FILENAME
        if not staged.is_file():
            return False
        new_text = staged.read_text(encoding="utf-8")
        self.ensure()
        old_text = self.path.read_text(encoding="utf-8")
        if new_text == old_text or not new_text.strip():
            return False
        shutil.copy2(staged, self.path)
        return True

    @property
    def distill_count(self) -> int:
        if not self.path.is_file():
            return 0
        return len(_DISTILL_RE.findall(self.path.read_text(encoding="utf-8")))
