"""SQLite persistence (SQLModel) for profiles, graph state, and feedback.

The database path comes from ``FORGEMIND_DB`` (default ``data/forgemind.db``).
Functions are small and typed; the full graph state is stored as JSON so any
session can be resumed after a crash or reload.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from ..state import ForgeMindState, Profile


def _db_url() -> str:
    path = os.getenv("FORGEMIND_DB", "data/forgemind.db")
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _engine():
    return create_engine(_db_url(), echo=False)


class ProfileRecord(SQLModel, table=True):
    """One developer profile per session."""

    __tablename__ = "profiles"

    session_id: str = Field(primary_key=True)
    profile_json: str
    updated_at: str


class StateRecord(SQLModel, table=True):
    """Full graph-state snapshot for resume support (latest per session)."""

    __tablename__ = "states"

    session_id: str = Field(primary_key=True)
    state_json: str
    phase: str
    updated_at: str


class FeedbackRecord(SQLModel, table=True):
    """Raw swipe/feedback events for the taste learner."""

    __tablename__ = "feedbacks"

    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str
    candidate_id: str
    action: str
    comment: str = ""
    created_at: str


class TasteProfileRecord(SQLModel, table=True):
    """Learned preference weights + embedding taste vector per session."""

    __tablename__ = "taste_profiles"

    session_id: str = Field(primary_key=True)
    weights_json: str
    vector_json: str
    summary: str = ""
    updated_at: str


def init_db() -> None:
    """Create tables if missing."""
    SQLModel.metadata.create_all(_engine())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_profile(session_id: str, profile: Profile) -> None:
    """Upsert the profile for a session."""
    with Session(_engine()) as session:
        record = session.get(ProfileRecord, session_id)
        payload = profile.model_dump_json()
        if record is None:
            session.add(ProfileRecord(session_id=session_id, profile_json=payload, updated_at=_now()))
        else:
            record.profile_json = payload
            record.updated_at = _now()
            session.add(record)
        session.commit()


def load_profile(session_id: str) -> Optional[Profile]:
    """Load a saved profile, or None."""
    with Session(_engine()) as session:
        record = session.get(ProfileRecord, session_id)
        if record is None:
            return None
        return Profile.model_validate(json.loads(record.profile_json))


def save_state(state: ForgeMindState) -> None:
    """Persist the full graph state (called after every node)."""
    with Session(_engine()) as session:
        record = session.get(StateRecord, state.session_id)
        payload = state.model_dump_json()
        phase = state.phase
        if record is None:
            session.add(StateRecord(session_id=state.session_id, state_json=payload, phase=phase, updated_at=_now()))
        else:
            record.state_json = payload
            record.phase = phase
            record.updated_at = _now()
            session.add(record)
        session.commit()


def load_state(session_id: str) -> Optional[ForgeMindState]:
    """Load the latest state snapshot for a session, or None."""
    with Session(_engine()) as session:
        record = session.get(StateRecord, session_id)
        if record is None:
            return None
        return ForgeMindState.model_validate(json.loads(record.state_json))


def record_feedback(session_id: str, candidate_id: str, action: str, comment: str = "") -> None:
    """Append a feedback event (used by Taste Learner in M6)."""
    with Session(_engine()) as session:
        session.add(
            FeedbackRecord(
                session_id=session_id,
                candidate_id=candidate_id,
                action=action,
                comment=comment,
                created_at=_now(),
            )
        )
        session.commit()


def save_taste(
    session_id: str,
    weights: dict[str, float],
    vector: list[float],
    summary: str = "",
) -> None:
    """Upsert the learned taste profile (weights + embedding vector)."""
    with Session(_engine()) as session:
        record = session.get(TasteProfileRecord, session_id)
        weights_json = json.dumps(weights)
        vector_json = json.dumps(vector)
        if record is None:
            session.add(
                TasteProfileRecord(
                    session_id=session_id,
                    weights_json=weights_json,
                    vector_json=vector_json,
                    summary=summary,
                    updated_at=_now(),
                )
            )
        else:
            record.weights_json = weights_json
            record.vector_json = vector_json
            record.summary = summary
            record.updated_at = _now()
            session.add(record)
        session.commit()


def load_taste(session_id: str) -> Optional[dict]:
    """Load the taste profile as ``{weights, vector, summary, updated_at}``."""
    with Session(_engine()) as session:
        record = session.get(TasteProfileRecord, session_id)
        if record is None:
            return None
        return {
            "weights": json.loads(record.weights_json),
            "vector": json.loads(record.vector_json),
            "summary": record.summary,
            "updated_at": record.updated_at,
        }
