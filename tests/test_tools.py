"""Tests for research tools (MOCK mode + parse helpers + mock fallbacks)."""
from __future__ import annotations

import httpx
import pytest

from forgemind.llm import router
from forgemind.tools import arxiv_search, devpost, github_search, reddit_hn, web_search
from forgemind.tools.common import (
    SearchResult,
    clear_tool_warnings,
    dedupe,
    get_tool_warnings,
    mock_fallback,
)


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    yield


def test_web_search_mock_returns_results() -> None:
    results = web_search.search("ai agents problem", k=3)
    assert len(results) == 3
    assert all(r.is_http() for r in results)
    assert all(r.source == "web" for r in results)


def test_reddit_hn_search_mock_sources() -> None:
    results = reddit_hn.search("dev tools", k=3)
    assert {r.source for r in results} <= {"reddit", "hn"}
    assert all(r.is_http() for r in results)


def test_github_search_mock() -> None:
    results = github_search.search("langgraph", k=4)
    assert len(results) == 4
    assert all(r.source == "github" for r in results)


def test_arxiv_search_mock() -> None:
    results = arxiv_search.search("llm agents", k=5)
    assert len(results) == 5
    assert all("arxiv.org" in r.url for r in results)


def test_devpost_search_mock() -> None:
    results = devpost.search("hackathon ai", k=5)
    assert len(results) == 5
    assert all("devpost.com" in r.url for r in results)


def test_mock_fallback_is_deterministic() -> None:
    a = mock_fallback("web", "same query", 3)
    b = mock_fallback("web", "same query", 3)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]


def test_dedupe_drops_non_http_and_duplicates() -> None:
    items = [
        SearchResult(title="a", url="https://x.test/1", source="web"),
        SearchResult(title="a2", url="https://x.test/1", source="web"),
        SearchResult(title="bad", url="ftp://nope", source="web"),
        SearchResult(title="b", url="https://y.test/2", source="hn"),
    ]
    out = dedupe(items)
    assert [r.url for r in out] == ["https://x.test/1", "https://y.test/2"]


def test_live_failure_returns_empty_and_records_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live mode (mock off) + broken endpoint -> [] + visible warning, no mock."""
    monkeypatch.setenv("FORGEMIND_MOCK", "0")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    clear_tool_warnings()

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "get", _boom)
    results = web_search.search("anything", k=3)
    assert results == []
    warnings = get_tool_warnings()
    assert any(w["tool"] == "web_search" for w in warnings)


def test_ddg_html_parse() -> None:
    html = (
        '<a rel="nofollow" class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage&amp;rut=abc">'
        "Example <b>Title</b></a>"
    )
    pairs = web_search._parse_ddg(html)
    assert pairs == [("Example Title", "https://example.org/page")]


def test_arxiv_atom_parse() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <title>Agents That Plan</title>
    <summary>We study planning agents.</summary>
  </entry>
  <entry>
    <id>not-a-url</id>
    <title>Bad Entry</title>
    <summary>x</summary>
  </entry>
</feed>"""
    results = arxiv_search._parse_atom(xml)
    assert len(results) == 1
    assert results[0].url == "http://arxiv.org/abs/2501.00001v1"
    assert results[0].source == "arxiv"


def test_devpost_html_parse() -> None:
    html = (
        '<a href="https://devpost.com/software/cool-agent">Cool Agent</a>'
        '<a href="https://devpost.com/software/other">Other</a>'
    )
    results = devpost._parse(html)
    assert [r.title for r in results] == ["Cool Agent", "Other"]
    assert all(r.source == "devpost" for r in results)


def test_search_results_feed_scout_prompt() -> None:
    """All mock tool results are valid SearchResult rows (URL + source)."""
    for mod in (web_search, reddit_hn, github_search, arxiv_search, devpost):
        results = mod.search("test query", k=2)
        assert results
        for r in results:
            assert r.is_http()
            assert r.title


def test_mock_mode_flag_detected() -> None:
    assert router.mock_mode() is True
