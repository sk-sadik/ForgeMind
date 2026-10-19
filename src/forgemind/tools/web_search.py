"""Web search tool: Tavily when ``TAVILY_API_KEY`` is set, else DuckDuckGo HTML.

Public entry point: :func:`search`. Mock mode / live failures return
deterministic fallback results (see ``tools/common.py``).
"""
from __future__ import annotations

import logging
import os
import re
import urllib.parse

import httpx

from ..llm import router
from .common import SearchResult, dedupe, fetch, mock_fallback

logger = logging.getLogger("forgemind.tools.web_search")

_MAX_K = 10


def _tavily(query: str, k: int) -> list[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": k},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20.0,
    )
    resp.raise_for_status()
    return [
        SearchResult(
            title=str(item.get("title") or query),
            url=str(item.get("url") or ""),
            snippet=str(item.get("content") or "")[:300],
            source="web",
        )
        for item in resp.json().get("results", [])
    ]


def _parse_ddg(html: str) -> list[tuple[str, str]]:
    """Extract (title, href) pairs from DuckDuckGo HTML results."""
    out: list[tuple[str, str]] = []
    for attrs, inner in re.findall(r"<a([^>]*)>(.*?)</a>", html, re.DOTALL):
        if "result__a" not in attrs:
            continue
        href_match = re.search(r'href="([^"]+)"', attrs)
        if not href_match:
            continue
        title = re.sub(r"<[^>]+>", "", inner).strip()
        out.append((title, _resolve_ddg_href(href_match.group(1))))
    return out


def _resolve_ddg_href(href: str) -> str:
    """DDG wraps results in //duckduckgo.com/l/?uddg=<encoded> redirects."""
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if query.get("uddg"):
            return query["uddg"][0]
    return href


def _duckduckgo(query: str, k: int) -> list[SearchResult]:
    resp = fetch("https://html.duckduckgo.com/html/", params={"q": query})
    pairs = _parse_ddg(resp.text)[:k]
    return [
        SearchResult(title=title or query, url=url, snippet="", source="web")
        for title, url in pairs
        if url.startswith(("http://", "https://"))
    ]


def search(query: str, k: int = 5) -> list[SearchResult]:
    """Run a web search; returns up to ``k`` http(s)-URL results."""
    k = min(k, _MAX_K)
    if router.mock_mode():
        return mock_fallback("web", query, k)
    try:
        results = (
            _tavily(query, k)
            if os.getenv("TAVILY_API_KEY")
            else _duckduckgo(query, k)
        )
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("web_search failed (%s); using mock fallback", exc)
        return mock_fallback("web", query, k)
    results = dedupe(results, limit=k)
    if not results:
        logger.warning("web_search returned nothing for %r; using mock fallback", query)
        return mock_fallback("web", query, k)
    return results
