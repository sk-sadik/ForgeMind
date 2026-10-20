"""Scout agent: research real, evidenced problems for the profile's interests.

Public entry point: :func:`run_scout`. Queries the tool layer (web search,
Reddit/HN, GitHub, arXiv), distills findings into ``ProblemRecord`` objects
via a structured LLM call, then enforces evidence rules: **records without
a valid http(s) URL are discarded**, URLs must trace back to tool results
in live mode, results are deduped and capped.
"""
from __future__ import annotations

import logging

from .. import state as S
from ..llm import router
from ..memory import db
from ..tools import arxiv_search, github_search, reddit_hn, web_search
from ..tools import common
from ..tools.common import SearchResult
from . import debug_log

logger = logging.getLogger("forgemind.scout")

MAX_PROBLEMS = 10
MAX_INTERESTS = 3
RESULTS_PER_QUERY = 4


def _build_queries(profile: S.Profile, refinement: str | None = None) -> list[str]:
    """Derive search queries from interests plus optional refinement text."""
    interests = [i.strip() for i in profile.interests if i.strip()][:MAX_INTERESTS]
    if not interests:
        interests = ["developer productivity"]
    queries: list[str] = []
    for interest in interests:
        queries.append(f"{interest} problem frustrated developers")
        queries.append(f"{interest} issue pain point reddit")
    if refinement:
        queries.append(refinement.strip())
    return queries


def _gather(queries: list[str]) -> list[SearchResult]:
    """Run queries across all tools; order: web, reddit/HN, github, arxiv."""
    results: list[SearchResult] = []
    for query in queries[:6]:  # bound total live calls
        results.extend(web_search.search(query, k=RESULTS_PER_QUERY))
        results.extend(reddit_hn.search(query, k=RESULTS_PER_QUERY))
    if queries:
        results.extend(github_search.search(f"{queries[0]} in:name,description", k=3))
        results.extend(arxiv_search.search(queries[0], k=3))
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    return unique


def _build_prompt(profile: S.Profile, results: list[SearchResult]) -> str:
    lines = "\n".join(
        f"[{i}] ({r.source}) {r.title} — {r.url}\n    {r.snippet}"
        for i, r in enumerate(results, start=1)
    )
    return (
        "You are the Scout agent. From the profile and tool evidence below, "
        "extract concrete, real problems developers actually face "
        "(complaints, open issues, research gaps).\n\n"
        f"Profile interests: {', '.join(profile.interests) or 'general dev'}\n"
        f"Profile skills: {', '.join(profile.skills) or 'n/a'}\n"
        f"Experience: {profile.experience_level.value}; goal: {profile.goal.value}\n"
        f"Constraints: {', '.join(profile.constraints) or 'none'}\n"
        f"Profile summary: {profile.summary or 'n/a'}\n\n"
        f"Tool evidence:\n{lines}\n\n"
        f"Return 3-{MAX_PROBLEMS} ProblemRecords. Rules:\n"
        "- id: p1, p2, ... (sequential)\n"
        "- source_url MUST be copied verbatim from a tool evidence URL above\n"
        "- evidence_summary: one line citing who complains and what breaks\n"
        "- source_type must match the evidence source in parentheses\n"
        "- Prioritize problems that match THIS profile's interests and skills"
    )


def _filter_records(
    records: list[S.ProblemRecord],
    allowed_urls: set[str] | None,
    enforce_membership: bool,
) -> list[S.ProblemRecord]:
    """Discard URL-less/invalid records, dedupe by URL, cap at MAX_PROBLEMS."""
    seen: set[str] = set()
    kept: list[S.ProblemRecord] = []
    for record in records:
        url = (record.source_url or "").strip()
        if not url.startswith(("http://", "https://")):
            logger.warning("dropping problem without valid URL: %r", record.id)
            continue
        if enforce_membership and allowed_urls is not None and url not in allowed_urls:
            logger.warning("dropping problem with URL outside tool evidence: %s", url)
            continue
        if url in seen:
            continue
        seen.add(url)
        kept.append(record)
        if len(kept) >= MAX_PROBLEMS:
            break
    return kept


def run_scout(state: S.ForgeMindState) -> S.ForgeMindState:
    """Scout real problems for the profile; writes ``state.problems``.

    Live mode: tool failures degrade to empty results with recorded
    warnings; if ALL tools fail/return nothing the scout raises so the
    UI shows the real error instead of fake evidence.
    """
    state = state.model_copy(deep=True)
    state.phase = "scout"
    state.activity_log.append(S.AgentActivity(agent="scout", status="running"))

    queries = _build_queries(state.profile, state.refinement_text)
    results = _gather(queries)
    if not results:
        failures = common.get_tool_warnings()
        detail = (
            "; ".join(f"{w['tool']}: {w['error']}" for w in failures)
            or "no results from any tool"
        )
        state.activity_log.append(
            S.AgentActivity(agent="scout", status="error", detail=detail)
        )
        db.save_state(state)
        raise RuntimeError(
            f"Scout found no evidence (all research tools failed or returned "
            f"nothing): {detail}"
        )

    prompt = _build_prompt(state.profile, results)
    debug_log.record(
        "scout_prompt",
        prompt=prompt,
        num_results=len(results),
        urls=[r.url for r in results],
    )
    output = router.structured_call(
        "scout",
        prompt,
        S.ScoutOutput,
        context={"num_problems": min(MAX_PROBLEMS, max(3, len(results) // 2))},
    )
    debug_log.record(
        "scout_output",
        model=router.last_model_used("scout") or router.get_llm("scout").model,
        problems=[p.model_dump(mode="json") for p in output.problems],
    )

    enforce_membership = not router.mock_mode()
    allowed = {r.url for r in results}
    state.problems = _filter_records(output.problems, allowed, enforce_membership)

    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(
            agent="scout",
            status="done",
            detail=f"{len(state.problems)} problems from {len(results)} tool results",
        )
    )
    return state
