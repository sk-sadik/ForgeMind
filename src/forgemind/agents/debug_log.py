"""Process-level debug registry: prompts, raw outputs, and model choices.

The Streamlit Debug panel reads these entries; nothing here is persisted.
Public entry points: :func:`record`, :func:`get_entries`, :func:`clear`.
"""
from __future__ import annotations

from typing import Any

_entries: list[dict[str, Any]] = []


def record(kind: str, **data: Any) -> None:
    """Append one debug entry (e.g. kind="generator_prompt", model=...)."""
    _entries.append({"kind": kind, **data})


def get_entries() -> list[dict[str, Any]]:
    """Return a copy of all debug entries in order."""
    return [dict(entry) for entry in _entries]


def clear() -> None:
    """Drop all debug entries (new runs / tests)."""
    _entries.clear()
