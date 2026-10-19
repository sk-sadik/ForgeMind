"""Devpost hackathon project search (HTML scrape, no key).

Public entry point: :func:`search`. Used primarily by the Novelty checker
(M4) to detect crowded hackathon space around a candidate.
"""
from __future__ import annotations

import logging
import re

import httpx

from ..llm import router
from .common import SearchResult, dedupe, fetch, mock_fallback

logger = logging.getLogger("forgemind.tools.devpost")

_SEARCH_URL = "https://devpost.com/software/search"
_ITEM_RE = re.compile(
    r'<a[^>]+href="(?P<url>https?://devpost\.com/software/[^"?#]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


def _parse(html: str) -> list[SearchResult]:
    """Extract project links + titles from a Devpost search page."""
    results: list[SearchResult] = []
    for match in _ITEM_RE.finditer(html):
        url = match.group("url").strip()
        title = re.sub(r"<[^>]+>", "", match.group("title")).strip()
        if not title:
            continue
        results.append(SearchResult(title=title, url=url, snippet="", source="devpost"))
    return results


def search(query: str, k: int = 5) -> list[SearchResult]:
    """Search Devpost for ``query``; returns up to ``k`` project results."""
    if router.mock_mode():
        return mock_fallback("devpost", query, k)
    try:
        resp = fetch(_SEARCH_URL, params={"query": query})
        results = dedupe(_parse(resp.text), limit=k)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("devpost search failed (%s); using mock fallback", exc)
        return mock_fallback("devpost", query, k)
    if not results:
        logger.warning("devpost search empty for %r; using mock fallback", query)
        return mock_fallback("devpost", query, k)
    return results
