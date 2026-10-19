"""Interactive CLI to exercise the Interviewer end-to-end.

Usage:
    python -m forgemind.cli                  # interactive interview
    python -m forgemind.cli --auto           # scripted answers (works with FORGEMIND_MOCK=1)
    python -m forgemind.cli --session s42    # resume/save under a specific session id
    python -m forgemind.cli --show-profile   # print the saved profile and exit
"""
from __future__ import annotations

import argparse
import sys

from .agents.interviewer import MAX_QUESTIONS, run_interview_step
from .memory import db
from .state import ForgeMindState

_AUTO_ANSWERS = [
    "AI agents, developer productivity, and hackathon tooling.",
    "Python, FastAPI, LangGraph, some React.",
    "About 15 hours a week; team of 2; want a hackathon-ready portfolio piece.",
    "No paid APIs and I dislike pure CRUD apps.",
]


def _print_prompt(state: ForgeMindState) -> None:
    if state.pending_question:
        print(f"\nForgeMind> {state.pending_question}")
        print("(press Enter to skip, or type 'done' to finish early)")


def _show_profile(state: ForgeMindState) -> None:
    p = state.profile
    print("\n=== Profile ===")
    print(f"Interests:    {', '.join(p.interests) or '-'}")
    print(f"Skills:       {', '.join(p.skills) or '-'}")
    print(f"Experience:   {p.experience_level.value}")
    print(f"Hours/week:   {p.hours_per_week}")
    print(f"Team size:    {p.team_size}")
    print(f"Goal:         {p.goal.value}")
    print(f"Constraints:  {', '.join(p.constraints) or '-'}")
    print(f"Summary:      {p.summary or '-'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forgemind", description="ForgeMind CLI interview")
    parser.add_argument("--session", default="default", help="session id (default: default)")
    parser.add_argument("--auto", action="store_true", help="use scripted answers (demo/mock)")
    parser.add_argument("--show-profile", action="store_true", help="print saved profile and exit")
    args = parser.parse_args(argv)

    db.init_db()

    if args.show_profile:
        profile = db.load_profile(args.session)
        if profile is None:
            print(f"no saved profile for session {args.session!r}")
            return 1
        state = ForgeMindState(session_id=args.session, profile=profile)
        _show_profile(state)
        return 0

    saved = db.load_state(args.session)
    state = saved if saved is not None else ForgeMindState(session_id=args.session)
    if state.profile.is_complete:
        print("Existing complete profile found; showing it (use --session to start anew).")
        _show_profile(state)
        return 0

    print("ForgeMind interview — I'll ask up to", MAX_QUESTIONS, "questions.")
    auto_iter = iter(_AUTO_ANSWERS)

    state = run_interview_step(state)  # seed/first question
    _print_prompt(state)

    while state.pending_question and not state.profile.is_complete:
        if args.auto:
            try:
                answer = next(auto_iter)
            except StopIteration:
                answer = "No constraints beyond those already mentioned."
            print(f"you> {answer}")
        else:
            try:
                answer = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye!")
                return 130
            if answer.lower() in {"done", "quit", "exit"}:
                break
        state = run_interview_step(state, answer)
        _print_prompt(state)

    if not state.profile.is_complete:
        # Finish early on user request: mark complete with what we have.
        state.profile.is_complete = True
        db.save_profile(state.session_id, state.profile)
        db.save_state(state)

    _show_profile(state)
    print(f"\n(saved to session {state.session_id!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
