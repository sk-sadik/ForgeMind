"""Streamlit UI for ForgeMind.

Run with::

    streamlit run src/forgemind/ui/app.py

Flow mirrors the graph: interview (direct interviewer steps) -> research
(run_pipeline) -> swipe cards (queue feedback, then run_pipeline again:
refine loops, accept builds the plan) -> plan view.

Everything interesting lives in :mod:`forgemind.ui.helpers` (pure) and the
agents; this module is only widgets + wiring.
"""
from __future__ import annotations

import secrets
import traceback

import streamlit as st

# Absolute imports: streamlit executes this file as a top-level script
# (__package__ is None), so package-relative imports would fail here.
from forgemind import state as S
from forgemind.agents import debug_log
from forgemind.agents.interviewer import run_interview_step
from forgemind.graph import run_pipeline
from forgemind.llm import router
from forgemind.memory import db
from forgemind.tools.common import clear_tool_warnings, get_tool_warnings
from forgemind.ui import helpers as H


def _fresh_session_id() -> str:
    return f"ui-{secrets.token_hex(4)}"


def _get_state(session_id: str) -> S.ForgeMindState:
    """Load the persisted state for ``session_id`` (or a fresh one)."""
    db.init_db()
    loaded = db.load_state(session_id)
    if loaded is not None:
        return loaded
    return S.ForgeMindState(session_id=session_id)


def _run_pipeline_ui(state: S.ForgeMindState) -> None:
    """Run the pipeline, failing loudly with the real error on screen."""
    clear_tool_warnings()
    debug_log.clear()
    try:
        with st.spinner("Scouting, generating, evaluating, ranking..."):
            st.session_state["state"] = run_pipeline(state)
    except Exception as exc:  # surfaced, never swallowed
        st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        tail = traceback.format_exc().strip().splitlines()[-8:]
        st.code("\n".join(tail), language="text")
        return
    st.rerun()


def _sidebar() -> S.ForgeMindState:
    """Session picker + status; keeps st.session_state['state'] in sync."""
    st.sidebar.title("ForgeMind")
    st.sidebar.caption("From interview to a validated project plan.")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = _fresh_session_id()
    session_id = st.sidebar.text_input("Session ID", value=st.session_state["session_id"])
    st.session_state["session_id"] = session_id
    if st.session_state.get("_loaded_for") != session_id:
        st.session_state["_loaded_for"] = session_id
        st.session_state["state"] = _get_state(session_id)

    state: S.ForgeMindState = st.session_state["state"]

    st.sidebar.markdown(
        f"**Phase:** {H.phase_label(state.phase)}  \n"
        f"**Loops:** {state.loop_count}/{state.max_loops}"
    )

    if state.pending_feedback:
        st.sidebar.warning(f"{len(state.pending_feedback)} feedback event(s) queued")

    if st.sidebar.button("New session (fresh ID)"):
        st.session_state["session_id"] = _fresh_session_id()
        st.session_state["state"] = S.ForgeMindState(session_id=st.session_state["session_id"])
        st.session_state["_loaded_for"] = st.session_state["session_id"]
        st.rerun()
    if st.sidebar.button("Load saved session ID"):
        st.session_state["state"] = _get_state(st.session_state["session_id"])
        st.session_state["_loaded_for"] = st.session_state["session_id"]
        st.rerun()
    return state


def _mode_banner() -> None:
    if router.mock_mode():
        st.error(
            "MOCK MODE — candidates come from deterministic fake data, NOT live "
            "research. Restart without FORGEMIND_MOCK to use real LLMs."
        )
    else:
        models = router.list_models()
        shown = ", ".join(f"{r}={m}" for r, m in sorted(models.items())[:6])
        st.success(f"LIVE MODE — models: {shown}")


def _tool_warnings_panel() -> None:
    warnings = get_tool_warnings()
    if not warnings:
        return
    st.warning(
        "Research tools degraded — results may be incomplete:\n\n"
        + "\n".join(f"- **{w['tool']}**: {w['error']}" for w in warnings)
    )


def _debug_panel(state: S.ForgeMindState) -> None:
    """Raw prompts/outputs/models from the last pipeline run."""
    entries = debug_log.get_entries()
    if not entries and not state.problems:
        return
    with st.expander("Debug — last run details", expanded=False):
        st.markdown("**Profile**")
        st.json(state.profile.model_dump(mode="json"))
        if state.problems:
            st.markdown("**Scouted problems**")
            st.json([p.model_dump(mode="json") for p in state.problems])
        for entry in entries:
            kind = entry.get("kind", "?")
            st.markdown(f"**{kind}**")
            payload = {k: v for k, v in entry.items() if k != "kind"}
            st.json(payload)


def _transcript(state: S.ForgeMindState) -> None:
    with st.expander("Chat transcript", expanded=False):
        for message in state.chat:
            with st.chat_message(message.get("role", "assistant")):
                st.markdown(message.get("content", ""))


def _activity(state: S.ForgeMindState) -> None:
    with st.expander("Agent activity log", expanded=False):
        for entry in reversed(state.activity_log):
            icon = {"done": "ok", "running": "...", "error": "ERR"}[entry.status]
            st.text(
                f"[{entry.ts[11:19]}] {icon} {entry.agent}: {entry.detail}"
            )


def _taste_panel(state: S.ForgeMindState) -> None:
    if not state.taste_weights and not state.taste_summary:
        return
    with st.expander("Learned taste", expanded=False):
        if state.taste_summary:
            st.caption(state.taste_summary)
        rows = H.taste_rows(state)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        if state.taste_vector:
            st.caption(
                f"Taste vector: {len(state.taste_vector)} dims, "
                f"norm {sum(x * x for x in state.taste_vector) ** 0.5:.3f}"
            )


def _queue_feedback(
    state: S.ForgeMindState,
    candidate_id: str,
    action: S.FeedbackAction,
    comment: str = "",
) -> None:
    """Append a feedback event unless an identical one is already queued."""
    for existing in state.pending_feedback:
        if (
            existing.candidate_id == candidate_id
            and existing.action == action
            and existing.comment == comment
        ):
            return
    state.pending_feedback.append(
        S.Feedback(candidate_id=candidate_id, action=action, comment=comment)
    )


def _interview_view(state: S.ForgeMindState) -> None:
    st.subheader("1 · Interview")
    st.caption("Answer a few questions so ForgeMind can target your profile.")
    st.markdown(H.profile_markdown(state.profile))

    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_input(
            state.pending_question or "Your answer",
            key=f"answer_{state.interview_visits}",
        )
        submitted = st.form_submit_button("Send", type="primary")

    if submitted and answer.strip():
        st.session_state["state"] = run_interview_step(state, answer.strip())
        st.rerun()

    if st.button("Finish interview early (use profile as-is)"):
        state = state.model_copy(deep=True)
        state.profile.is_complete = True
        db.save_profile(state.session_id, state.profile)
        db.save_state(state)
        st.session_state["state"] = state
        st.rerun()


def _research_view(state: S.ForgeMindState) -> None:
    st.subheader("2 · Research & ranking")
    st.markdown("**Your profile**")
    st.markdown(H.profile_markdown(state.profile))
    st.caption(
        "Scout gathers real problems with evidence, generators propose candidates, "
        "novelty/feasibility filter them, the critic attacks them, and the ranker "
        "scores the survivors."
    )
    if st.button("Run research & ranking", type="primary"):
        _run_pipeline_ui(state)


def _swipe_card(state: S.ForgeMindState, candidate: S.Candidate) -> None:
    with st.container(border=True):
        st.markdown(f"### {candidate.title}")
        st.caption(f"style: {candidate.style} · id: {candidate.id}")
        st.markdown(H.candidate_card(state, candidate))

        c1, c2, c3 = st.columns(3)
        if c1.button("Like", key=f"like_{candidate.id}"):
            _queue_feedback(state, candidate.id, S.FeedbackAction.LIKE)
            st.rerun()
        if c2.button("Dislike", key=f"dislike_{candidate.id}"):
            _queue_feedback(state, candidate.id, S.FeedbackAction.DISLIKE)
            st.rerun()
        if c3.button("Accept this idea", key=f"accept_{candidate.id}", type="primary"):
            _queue_feedback(state, candidate.id, S.FeedbackAction.ACCEPT)
            st.rerun()

        refine_col, button_col = st.columns([4, 1])
        refine_text = refine_col.text_input(
            "Refinement request", key=f"refine_{candidate.id}", label_visibility="collapsed"
        )
        if button_col.button("Queue refine", key=f"refine_btn_{candidate.id}"):
            if refine_text.strip():
                _queue_feedback(
                    state, candidate.id, S.FeedbackAction.REFINE, refine_text.strip()
                )
            st.rerun()


def _review_view(state: S.ForgeMindState) -> None:
    st.subheader("3 · Review candidates")
    assert state.ranking is not None
    st.dataframe(H.ranking_rows(state.ranking), hide_index=True, use_container_width=True)
    st.caption(state.ranking.summary)

    queued = H.queued_feedback_lines(state)
    if queued:
        st.info("Queued feedback:\n" + "\n".join(f"- {line}" for line in queued))

    st.markdown("**Swipe cards** — like/dislike shape future rankings, "
                "refine regenerates ideas, accept builds the project plan.")
    for candidate in state.candidates:
        _swipe_card(state, candidate)

    if queued and st.button("Submit feedback & continue", type="primary"):
        _run_pipeline_ui(state)


def _plan_view(state: S.ForgeMindState) -> None:
    st.subheader("4 · Your project plan")
    assert state.plan is not None
    st.markdown(H.plan_markdown(state.plan))
    if st.button("Keep exploring (clear plan, back to review)"):
        state = state.model_copy(deep=True)
        state.plan = None
        state.chosen_candidate_id = None
        state.phase = "done"
        db.save_state(state)
        st.session_state["state"] = state
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="ForgeMind", layout="wide")
    _mode_banner()
    state = _sidebar()

    if H.needs_interview(state):
        _interview_view(state)
        state = st.session_state["state"]
    elif H.needs_research(state):
        _research_view(state)
        state = st.session_state["state"]
    elif H.plan_ready(state):
        _plan_view(state)
        state = st.session_state["state"]
    elif H.review_ready(state):
        _review_view(state)
        state = st.session_state["state"]
    else:
        st.info(f"Pipeline finished (phase: {H.phase_label(state.phase)}).")
        if st.button("Run again"):
            _run_pipeline_ui(state)

    st.divider()
    _tool_warnings_panel()
    _debug_panel(state)
    _taste_panel(state)
    _transcript(state)
    _activity(state)


if __name__ == "__main__":
    main()
