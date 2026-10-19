"""Interviewer agent: adaptive multi-turn interview that builds a Profile.

Public entry point: :func:`run_interview_step`. One question at a time,
max ``MAX_QUESTIONS`` (8) questions, early stop once the profile is complete.
Every turn is persisted through the shared ForgeMindState (callers persist
via ``memory.db.save_state`` after applying the returned updates).
"""
from __future__ import annotations

from typing import Optional

from ..llm import router
from ..memory import db
from ..state import AgentActivity, ForgeMindState, InterviewerOutput, InterviewTurn, Profile

MAX_QUESTIONS = 8

_SEED_QUESTIONS = [
    "Hi! I'm ForgeMind. What kinds of problems or domains genuinely interest you?",
    "What languages, frameworks, or tools are you strongest with?",
]

_PROFILE_FIELDS = (
    "interests",
    "skills",
    "experience_level",
    "hours_per_week",
    "team_size",
    "goal",
    "constraints",
)


def _profile_complete(profile: Profile) -> bool:
    """Heuristic completeness check (used to enforce early stop)."""
    return bool(
        profile.interests
        and profile.skills
        and profile.hours_per_week > 0
        and profile.constraints is not None
        and profile.summary
    )


def _merge_profile(current: Profile, update: Profile) -> Profile:
    """Merge a partial profile update into the current profile.

    Lists are unioned (order-preserving); scalars take the update's value
    unless the update still holds the field's default.
    """
    merged = current.model_copy(deep=True)
    for field_name in ("interests", "skills", "constraints"):
        existing = getattr(merged, field_name)
        for item in getattr(update, field_name):
            if item not in existing:
                existing.append(item)
    if update.experience_level != Profile().experience_level:
        merged.experience_level = update.experience_level
    if update.hours_per_week != Profile().hours_per_week:
        merged.hours_per_week = update.hours_per_week
    if update.team_size != Profile().team_size:
        merged.team_size = update.team_size
    if update.goal != Profile().goal:
        merged.goal = update.goal
    if update.summary:
        merged.summary = update.summary
    return merged


def _build_prompt(state: ForgeMindState, next_index: int) -> str:
    """Compose the interviewer prompt from current state."""
    profile_json = state.profile.model_dump_json(indent=2)
    history = "\n".join(
        f"Q: {t.question}\nA: {t.answer}" for t in state.interview_turns
    ) or "(no turns yet)"
    remaining = MAX_QUESTIONS - len(state.interview_turns)
    return (
        f"You are interviewing a developer (question {next_index} of max {MAX_QUESTIONS}, "
        f"{remaining} remaining).\n\n"
        f"Current profile JSON:\n{profile_json}\n\n"
        f"Conversation so far:\n{history}\n\n"
        "Ask the single most useful next question to complete the profile "
        "(interests, skills, experience, hours, team, goal, constraints). "
        "If enough information is present, set is_complete=true and next_question=null. "
        "Always return the full updated profile in profile_update."
    )


def run_interview_step(
    state: ForgeMindState, user_answer: Optional[str] = None
) -> ForgeMindState:
    """Advance the interview by one turn and return the updated state.

    - If ``user_answer`` is given, it is appended to ``interview_turns``.
    - Enforces ``MAX_QUESTIONS`` hard cap and early stop on completeness.
    - Persists the profile to SQLite when the interview completes.
    """
    state = state.model_copy(deep=True)
    state.phase = "interview"

    if user_answer is not None and state.pending_question:
        state.interview_turns.append(
            InterviewTurn(question=state.pending_question, answer=user_answer.strip())
        )
        state.chat.append({"role": "user", "content": user_answer.strip()})

    # Hard cap: never ask more than MAX_QUESTIONS questions.
    over_budget = len(state.interview_turns) >= MAX_QUESTIONS

    if over_budget or (_profile_complete(state.profile) and len(state.interview_turns) >= 1):
        state.profile.is_complete = True
        state.pending_question = None
        db.save_profile(state.session_id, state.profile)
        db.save_state(state)
        state.activity_log.append(
            AgentActivity(agent="interviewer", status="done", detail="profile complete")
        )
        state.chat.append(
            {"role": "assistant", "content": "Great — I have everything I need. Thanks!"}
        )
        return state

    question_index = len(state.interview_turns) + 1
    prompt = _build_prompt(state, question_index)

    # Seed question for the very first turn (no LLM needed if none asked yet).
    if not state.interview_turns and not state.chat:
        question = _SEED_QUESTIONS[0]
        state.pending_question = question
        state.chat.append({"role": "assistant", "content": question})
        state.activity_log.append(
            AgentActivity(agent="interviewer", status="done", detail="seed question")
        )
        return state

    output = router.structured_call(
        "interviewer",
        prompt,
        InterviewerOutput,
        context={"answered": len(state.interview_turns)},
    )

    state.profile = _merge_profile(state.profile, output.profile_update)

    if output.is_complete or len(state.interview_turns) >= MAX_QUESTIONS:
        state.profile.is_complete = True
        state.pending_question = None
        farewell = output.reason or "Great — I have everything I need. Thanks!"
        state.chat.append({"role": "assistant", "content": farewell})
    else:
        question = output.next_question or _SEED_QUESTIONS[
            min(len(state.interview_turns), len(_SEED_QUESTIONS) - 1)
        ]
        state.pending_question = question
        state.chat.append({"role": "assistant", "content": question})

    db.save_profile(state.session_id, state.profile)
    db.save_state(state)
    state.activity_log.append(
        AgentActivity(
            agent="interviewer",
            status="done",
            detail=f"turn {question_index}/{MAX_QUESTIONS}",
        )
    )
    return state
