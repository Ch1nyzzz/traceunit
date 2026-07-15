"""AppWorld backend — external-process (subprocess-harvest) code-agent eval.

AppWorld pins pydantic-v1 / SQLAlchemy 1.4 (conflicts with the worldcalib venv),
so eval runs in an isolated ``.venv-appworld`` via :mod:`eval_entry`; the runner
(:class:`AppWorldSourceRunner`) spawns it and harvests per-task results. The
editable surface the proposer evolves is the single worldcalib-free seed agent
``agent.py`` (a minimal ReAct code agent). Mirrors the toolathlon backend.
"""

from __future__ import annotations

DEFAULT_APPWORLD_AGENT_NAME = "appworld_passthrough"
DEFAULT_APPWORLD_SEED_SCAFFOLDS: tuple[str, ...] = (DEFAULT_APPWORLD_AGENT_NAME,)

__all__ = [
    "DEFAULT_APPWORLD_AGENT_NAME",
    "DEFAULT_APPWORLD_SEED_SCAFFOLDS",
]
