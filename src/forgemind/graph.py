"""LangGraph orchestration for ForgeMind.

Flow::

    interview --continue--> interview
        |
      (profile complete / answers exhausted)
        v
      scout -> generate -> (novelty || feasibility in parallel)
                              \\                /
                               -> filter gate -
                                     |            \\
                        (<5 survive & loops left)  else
                                     |              \\
                                 generate          critic -> ranker -> present -> END

Every node persists the full state via ``memory.db.save_state`` (inside the
agents + a wrapper-level save), so sessions can resume at any phase.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from . import state as S
from .agents.critic import run_critic
from .agents.feasibility import run_feasibility
from .agents.generators import run_generators
from .agents.interviewer import run_interview_step
from .agents.novelty import run_novelty
from .agents.ranker import run_ranker
from .agents.scout import run_scout
from .memory import db

logger = logging.getLogger("forgemind.graph")

MIN_CANDIDATES = 5
MAX_INTERVIEW_VISITS = 12

AgentFn = Callable[[S.ForgeMindState], S.ForgeMindState]


# ---------------------------------------------------------------------------
# Interview node (consumes one scripted/UI answer per visit)
# ---------------------------------------------------------------------------

def _interview_node(state: S.ForgeMindState) -> S.ForgeMindState:
    answer: str | None = None
    if state.pending_question and state.pending_answers:
        state = state.model_copy(deep=True)
        answer = state.pending_answers[0]
        state.pending_answers = state.pending_answers[1:]
    out = run_interview_step(state, answer)
    out.interview_visits += 1
    return out


def _route_interview(state: S.ForgeMindState | dict[str, Any]) -> str:
    state = _coerce(state)
    if (
        state.pending_question
        and not state.profile.is_complete
        and state.pending_answers
        and state.interview_visits < MAX_INTERVIEW_VISITS
    ):
        return "continue"
    return "next"


# ---------------------------------------------------------------------------
# Filter gate: regenerate when fewer than MIN_CANDIDATES survive
# ---------------------------------------------------------------------------

def _survives(state: S.ForgeMindState, candidate: S.Candidate) -> bool:
    novelty = state.novelty_reports.get(candidate.id)
    feasibility = state.feasibility_reports.get(candidate.id)
    if novelty is None or feasibility is None:
        return False
    return novelty.novelty_score >= 0.1 and feasibility.feasibility_score >= 0.1


def _filter_gate(state: S.ForgeMindState) -> S.ForgeMindState:
    state = state.model_copy(deep=True)
    surviving = [c for c in state.candidates if _survives(state, c)]
    can_loop = bool(state.candidates) and state.loop_count < state.max_loops
    if len(surviving) < MIN_CANDIDATES and can_loop:
        state.loop_count += 1
        state.regeneration_feedback = (
            f"Only {len(surviving)} of {len(state.candidates)} candidates survived "
            f"novelty/feasibility filters (need {MIN_CANDIDATES}). Generate more, "
            "broader or stronger candidates."
        )
        state.phase = "generate"
        detail = f"regen loop {state.loop_count}/{state.max_loops}: {len(surviving)} survived"
    else:
        state.phase = "critic"
        detail = f"{len(surviving)} survived; routing to critic"
    state.activity_log.append(S.AgentActivity(agent="gate", status="done", detail=detail))
    db.save_state(state)
    return state


def _route_gate(state: S.ForgeMindState | dict[str, Any]) -> str:
    state = _coerce(state)
    return "regenerate" if state.phase == "generate" else "critic"


# ---------------------------------------------------------------------------
# Present node
# ---------------------------------------------------------------------------

def _present(state: S.ForgeMindState) -> S.ForgeMindState:
    state = state.model_copy(deep=True)
    state.phase = "present"
    if state.ranking and state.ranking.ranked:
        lines = [
            f"{i}. {entry.candidate.title} — score {entry.weighted_score:.2f}"
            for i, entry in enumerate(state.ranking.ranked, start=1)
        ]
        message = "Top ideas:\n" + "\n".join(lines)
    else:
        message = "No candidates survived to ranking."
    state.chat.append({"role": "assistant", "content": message})
    state.phase = "done"
    db.save_state(state)
    state.activity_log.append(S.AgentActivity(agent="present", status="done"))
    return state


# ---------------------------------------------------------------------------
# Node wrapper: JSON updates + reducer deltas + persistence per node
# ---------------------------------------------------------------------------

_REDUCER_KEYS = ("chat", "activity_log")


def _coerce(state: S.ForgeMindState | dict[str, Any]) -> S.ForgeMindState:
    """Accept either a ForgeMindState or its dict form from LangGraph."""
    if isinstance(state, S.ForgeMindState):
        return state
    return S.ForgeMindState.model_validate(state)


def _wrap(agent_fn: AgentFn) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Adapt a full-state agent function into a LangGraph node.

    Returns JSON-serialized field updates; reducer fields carry only the
    delta since the node started. Persists the merged state afterwards.
    """

    def node(state: S.ForgeMindState | dict[str, Any]) -> dict[str, Any]:
        state = _coerce(state)
        old_chat = len(state.chat)
        old_activity = len(state.activity_log)
        out = agent_fn(state)

        base = state.model_dump(mode="json")
        # Only emit keys whose value actually changed: parallel branches
        # (novelty/feasibility) may both run, and plain fields are
        # LastValue channels that accept exactly one write per super-step.
        update = {
            key: value
            for key, value in out.model_dump(mode="json", exclude=set(_REDUCER_KEYS)).items()
            if value != base.get(key)
        }
        if out.chat[old_chat:]:
            update["chat"] = out.chat[old_chat:]
        activity_delta = out.activity_log[old_activity:]
        if activity_delta:
            update["activity_log"] = [
                a.model_dump(mode="json") if isinstance(a, S.AgentActivity) else a
                for a in activity_delta
            ]

        merged = dict(base)
        for key, value in update.items():
            merged[key] = base[key] + value if key in _REDUCER_KEYS else value
        db.save_state(S.ForgeMindState.model_validate(merged))
        return update

    return node


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Compile the ForgeMind LangGraph state machine."""
    graph = StateGraph(S.ForgeMindState)

    graph.add_node("interview", _wrap(_interview_node))
    graph.add_node("scout", _wrap(run_scout))
    graph.add_node("generate", _wrap(run_generators))
    graph.add_node("novelty", _wrap(run_novelty))
    graph.add_node("feasibility", _wrap(run_feasibility))
    graph.add_node("gate", _wrap(_filter_gate))
    graph.add_node("critic", _wrap(run_critic))
    graph.add_node("ranker", _wrap(run_ranker))
    graph.add_node("present", _wrap(_present))

    graph.add_edge(START, "interview")
    graph.add_conditional_edges(
        "interview",
        _route_interview,
        {"continue": "interview", "next": "scout"},
    )
    graph.add_edge("scout", "generate")
    graph.add_edge("generate", "novelty")
    graph.add_edge("generate", "feasibility")
    graph.add_edge(["novelty", "feasibility"], "gate")
    graph.add_conditional_edges(
        "gate",
        _route_gate,
        {"regenerate": "generate", "critic": "critic"},
    )
    graph.add_edge("critic", "ranker")
    graph.add_edge("ranker", "present")
    graph.add_edge("present", END)

    return graph.compile()


def run_pipeline(
    state: S.ForgeMindState, recursion_limit: int = 200
) -> S.ForgeMindState:
    """Run the full ForgeMind pipeline to completion; returns final state."""
    db.init_db()
    app = build_graph()
    raw = app.invoke(
        state.model_dump(mode="json"),
        config={"recursion_limit": recursion_limit},
    )
    final = (
        raw if isinstance(raw, S.ForgeMindState) else S.ForgeMindState.model_validate(raw)
    )
    db.save_state(final)
    return final
