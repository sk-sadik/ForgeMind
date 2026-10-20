"""Global test setup: every ForgeMind test runs in MOCK mode by default.

Individual tests that exercise live code paths override this inside the
test body via ``monkeypatch.setenv("FORGEMIND_MOCK", "0")`` (which runs
after all fixtures).
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_mock_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
