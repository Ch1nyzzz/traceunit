from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _use_snapshot_sandbox_for_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production default is Docker/bwrap; unit tests exercise copy isolation."""

    monkeypatch.setenv("TRACEUNIT_TEST_SANDBOX_MODE", "copy")
