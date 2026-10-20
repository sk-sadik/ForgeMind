"""Shared helpers for ForgeMind research tools.

Every tool returns :class:`SearchResult` objects (title + http(s) URL +
snippet + source). In ``FORGEMIND_MOCK`` mode tools return deterministic
mock results without any network call. In live mode tool failures or empty
responses return an empty list and are recorded in the tool-warning registry
(:func:`record_tool_failure`) so the UI can show a clearly visible warning
naming the failed tool — live mode NEVER silently substitutes mock data.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from ..llm import router

logger = logging.getLogger("forgemind.tools")

SourceKind = Literal["web", "reddit", "hn", "github", "arxiv", "devpost"]

TIMEOUT = 15.0
DEFAULT_HEADERS = {
    "User-Agent": "ForgeMind/0.1 (research; https://github.com/sk-sadik/ForgeMind)"
}

# ---------------------------------------------------------------------------
# Visible failure registry (surfaced in the UI as warnings)
# ---------------------------------------------------------------------------

_tool_warnings: list[dict[str, str]] = []


def record_tool_failure(tool: str, error: str) -> None:
    """Record a live-mode tool failure/degradation for UI display."""
    _tool_warnings.append({"tool": tool, "error": error})
    logger.warning("tool %s degraded: %s", tool, error)


def get_tool_warnings() -> list[dict[str, str]]:
    """Return recorded tool failures (copy)."""
    return [dict(w) for w in _tool_warnings]


def clear_tool_warnings() -> None:
    """Reset the failure registry (start of a new pipeline run)."""
    _tool_warnings.clear()


class SearchResult(BaseModel):
    """A single evidence candidate returned by any tool."""

    title: str
    url: str
    snippet: str = ""
    source: SourceKind
    score: float = Field(default=0.5, ge=0.0, le=1.0)

    def is_http(self) -> bool:
        return self.url.startswith(("http://", "https://"))


def fetch(url: str, **kwargs) -> httpx.Response:
    """GET helper with shared timeout/headers; raises on non-2xx responses."""
    resp = httpx.get(
        url,
        timeout=TIMEOUT,
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def slugify(text: str, limit: int = 40) -> str:
    """URL-safe slug from arbitrary text."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "result"


def _mock_url(source: SourceKind, slug: str, i: int) -> str:
    if source == "reddit":
        return f"https://www.reddit.com/r/programming/comments/1abc{i}0/{slug}/"
    if source == "hn":
        return f"https://news.ycombinator.com/item?id={40000000 + i}"
    if source == "github":
        return f"https://github.com/mock-org/{slug}/issues/{i}"
    if source == "arxiv":
        return f"https://arxiv.org/abs/2501.{i:05d}"
    if source == "devpost":
        return f"https://devpost.com/software/{slug}-{i}"
    return f"https://example.com/{slug}/{i}"


def mock_fallback(source: SourceKind, query: str, k: int = 5) -> list[SearchResult]:
    """Deterministic fake results (mock mode or live-failure fallback)."""
    slug = slugify(query)
    return [
        SearchResult(
            title=f"[mock:{source}] {query} — result {i}",
            url=_mock_url(source, slug, i),
            snippet=f"Mock {source} snippet for {query!r} #{i}.",
            source=source,
            score=max(0.1, 0.9 - 0.1 * i),
        )
        for i in range(1, max(k, 1) + 1)
    ]


def dedupe(results: list[SearchResult], limit: int = 20) -> list[SearchResult]:
    """Drop non-http and duplicate URLs, keep order, cap at ``limit``."""
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        if not r.is_http() or r.url in seen:
            continue
        seen.add(r.url)
        out.append(r)
        if len(out) >= limit:
            break
    return out
