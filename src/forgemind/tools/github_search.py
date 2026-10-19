"""GitHub repository search (public API; optional ``GITHUB_TOKEN``).

Public entry point: :func:`search`. Used by the Scout for issue-driven
problems and by Novelty (M4) for crowded-space detection.
"""
from __future__ import annotations

import logging
import os

import httpx

from ..llm import router
from .common import SearchResult, dedupe, mock_fallback

logger = logging.getLogger("forgemind.tools.github_search")

_API = "https://api.github.com/search/repositories"


def search(query: str, k: int = 5) -> list[SearchResult]:
    """Search GitHub repos for ``query``; returns up to ``k`` results."""
    if router.mock_mode():
        return mock_fallback("github", query, k)
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(
            _API,
            params={"q": query, "per_page": k, "sort": "stars", "order": "desc"},
            headers=headers,
            timeout=20.0,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("github search failed (%s); using mock fallback", exc)
        return mock_fallback("github", query, k)
    results = [
        SearchResult(
            title=str(item.get("full_name") or query),
            url=str(item.get("html_url") or ""),
            snippet=str(item.get("description") or "")[:300],
            source="github",
            score=min(1.0, float(item.get("stargazers_count") or 0) / 5000),
        )
        for item in items
    ]
    return dedupe(results, limit=k)
