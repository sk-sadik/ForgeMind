"""Reddit + Hacker News search.

Public entry point: :func:`search` (merges both sources). Reddit uses the
public ``search.json`` endpoint; HN uses the Algolia API. Both need no key.
"""
from __future__ import annotations

import logging

import httpx

from ..llm import router
from .common import SearchResult, dedupe, fetch, mock_fallback

logger = logging.getLogger("forgemind.tools.reddit_hn")

_HN_API = "https://hn.algolia.com/api/v1/search"
_REDDIT_API = "https://www.reddit.com/search.json"


def search_reddit(query: str, k: int = 5) -> list[SearchResult]:
    """Search Reddit posts for ``query`` (public JSON, no key)."""
    if router.mock_mode():
        return mock_fallback("reddit", query, k)
    try:
        resp = fetch(
            _REDDIT_API,
            params={"q": query, "limit": k, "sort": "relevance", "type": "link"},
        )
        children = resp.json().get("data", {}).get("children", [])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("reddit search failed (%s); using mock fallback", exc)
        return mock_fallback("reddit", query, k)
    results = []
    for child in children:
        data = child.get("data", {})
        permalink = data.get("permalink")
        if not permalink:
            continue
        results.append(
            SearchResult(
                title=str(data.get("title") or query),
                url="https://www.reddit.com" + permalink,
                snippet=str(data.get("selftext") or "")[:300],
                source="reddit",
                score=min(1.0, float(data.get("ups") or 0) / 1000),
            )
        )
    return dedupe(results, limit=k)


def search_hn(query: str, k: int = 5) -> list[SearchResult]:
    """Search Hacker News stories via the Algolia API (no key)."""
    if router.mock_mode():
        return mock_fallback("hn", query, k)
    try:
        resp = fetch(
            _HN_API,
            params={"query": query, "tags": "story", "hitsPerPage": k},
        )
        hits = resp.json().get("hits", [])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("hn search failed (%s); using mock fallback", exc)
        return mock_fallback("hn", query, k)
    results = []
    for hit in hits:
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        title = hit.get("title")
        if not title or not str(url).startswith(("http://", "https://")):
            continue
        results.append(
            SearchResult(
                title=str(title),
                url=str(url),
                snippet=f"{hit.get('points', 0)} points, {hit.get('num_comments', 0)} comments",
                source="hn",
                score=min(1.0, float(hit.get("points") or 0) / 500),
            )
        )
    return dedupe(results, limit=k)


def search(query: str, k: int = 5) -> list[SearchResult]:
    """Search both Reddit and HN; returns up to ``2 * k`` results."""
    return dedupe(search_reddit(query, k) + search_hn(query, k), limit=2 * k)
