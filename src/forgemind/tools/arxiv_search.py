"""arXiv paper search (Atom API, no key).

Public entry point: :func:`search`. Used by the Scout to surface research
gaps as problem evidence.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import httpx

from ..llm import router
from .common import SearchResult, dedupe, mock_fallback, record_tool_failure

logger = logging.getLogger("forgemind.tools.arxiv_search")

_API = "http://export.arxiv.org/api/query"
_ATOM = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_atom(xml_text: str) -> list[SearchResult]:
    """Parse the arXiv Atom feed into search results."""
    root = ET.fromstring(xml_text)
    results: list[SearchResult] = []
    for entry in root.findall("atom:entry", _ATOM):
        title = (entry.findtext("atom:title", "", _ATOM) or "").strip().replace("\n", " ")
        url = (entry.findtext("atom:id", "", _ATOM) or "").strip()
        summary = (entry.findtext("atom:summary", "", _ATOM) or "").strip().replace("\n", " ")
        if not title or not url.startswith(("http://", "https://")):
            continue
        results.append(
            SearchResult(title=title, url=url, snippet=summary[:300], source="arxiv", score=0.6)
        )
    return results


def search(query: str, k: int = 5) -> list[SearchResult]:
    """Search arXiv for ``query``; returns up to ``k`` paper results."""
    if router.mock_mode():
        return mock_fallback("arxiv", query, k)
    try:
        resp = fetch_atom(query, k)
        results = _parse_atom(resp)
    except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
        record_tool_failure("arxiv", f"{type(exc).__name__}: {exc}")
        return []
    results = dedupe(results, limit=k)
    if not results:
        record_tool_failure("arxiv", f"no results for query {query!r}")
    return results


def fetch_atom(query: str, k: int) -> str:
    """Fetch the raw Atom XML for ``query`` (separated for easy mocking)."""
    from .common import fetch  # local import keeps module init light

    resp = fetch(
        _API,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": k,
            "sortBy": "relevance",
        },
    )
    return resp.text
