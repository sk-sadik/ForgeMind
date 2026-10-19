"""Pydantic v2 schemas for shared LangGraph state and all structured LLM outputs.

Every agent consumes/produces objects defined here. The graph state
(`ForgeMindState`) is itself a Pydantic model so it can be serialized to
SQLite after every node for session resume.
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow() -> str:
    """ISO timestamp helper used across schemas."""
    return datetime.now(timezone.utc).isoformat()


def _last_wins(current: Any, new: Any) -> Any:
    """Reducer for scalar fields written by parallel graph branches."""
    return new


# ---------------------------------------------------------------------------
# Profile / interview
# ---------------------------------------------------------------------------

class ExperienceLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Goal(str, Enum):
    HACKATHON = "hackathon"
    PORTFOLIO = "portfolio"
    STARTUP = "startup"
    LEARNING = "learning"
    RESEARCH = "research"


class Profile(BaseModel):
    """Developer profile built by the Interviewer agent."""

    interests: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE
    hours_per_week: int = Field(default=10, ge=0)
    team_size: int = Field(default=1, ge=1)
    goal: Goal = Goal.HACKATHON
    constraints: list[str] = Field(default_factory=list)
    summary: str = ""
    is_complete: bool = False


class InterviewTurn(BaseModel):
    """One Q/A exchange in the interview."""

    question: str
    answer: str = ""


class InterviewerOutput(BaseModel):
    """Structured output of one Interviewer step."""

    next_question: Optional[str] = None
    profile_update: Profile = Field(default_factory=Profile)
    is_complete: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Scout findings
# ---------------------------------------------------------------------------

SourceType = Literal["web", "reddit", "hn", "github", "arxiv", "devpost"]


class ProblemRecord(BaseModel):
    """A real problem found by the Scout, always with an evidence URL."""

    id: str
    title: str
    description: str
    evidence_summary: str
    source_url: str
    source_type: SourceType = "web"
    interest: str = ""
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("source_url")
    @classmethod
    def _url_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"source_url must be an http(s) URL, got: {v!r}")
        return v


class ScoutOutput(BaseModel):
    """Structured output of the Scout agent."""

    problems: list[ProblemRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidates (problem statements)
# ---------------------------------------------------------------------------

PSStyle = Literal["impact", "technical", "fusion"]


class Candidate(BaseModel):
    """A generated problem-statement candidate."""

    id: str
    style: PSStyle
    title: str
    problem: str
    proposed_solution: str
    target_users: str
    agentic_angle: str
    evidence_ids: list[str] = Field(default_factory=list)
    impact_hypothesis: str = ""

    @field_validator("evidence_ids")
    @classmethod
    def _must_cite_evidence(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("every candidate must cite at least one evidence_id")
        return v


class GeneratorOutput(BaseModel):
    """Structured output of one generator style."""

    candidates: list[Candidate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation: novelty, feasibility, critique
# ---------------------------------------------------------------------------

class SimilarProject(BaseModel):
    name: str
    url: str
    why_similar: str
    similarity: float = Field(ge=0.0, le=1.0)


class NoveltyReport(BaseModel):
    candidate_id: str
    novelty_score: float = Field(ge=0.0, le=1.0)
    similar_projects: list[SimilarProject] = Field(default_factory=list)
    differentiation: str = ""


class FeasibilityReport(BaseModel):
    candidate_id: str
    feasibility_score: float = Field(ge=0.0, le=1.0)
    estimated_weeks: int = Field(ge=1)
    risks: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    required_apis: list[str] = Field(default_factory=list)


class CritiqueReport(BaseModel):
    candidate_id: str
    attacks: list[str] = Field(default_factory=list)
    critique_penalty: float = Field(ge=0.0, le=1.0)
    regret_prediction: str
    impact_score: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class ScoredCandidate(BaseModel):
    candidate: Candidate
    fit_score: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    critique_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    weighted_score: float = 0.0
    justification: str = ""
    novelty: Optional[NoveltyReport] = None
    feasibility: Optional[FeasibilityReport] = None
    critique: Optional[CritiqueReport] = None


class RankingOutput(BaseModel):
    """Top-3 ranking with weights and justifications."""

    weights: dict[str, float] = Field(default_factory=dict)
    ranked: list[ScoredCandidate] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Feedback / taste
# ---------------------------------------------------------------------------

class FeedbackAction(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    REFINE = "refine"
    ACCEPT = "accept"


class Feedback(BaseModel):
    candidate_id: str
    action: FeedbackAction
    comment: str = ""


class TasteUpdate(BaseModel):
    """Structured output of the Taste Learner."""

    taste_vector: list[float] = Field(default_factory=list)
    weight_adjustments: dict[str, float] = Field(default_factory=dict)
    learned_summary: str = ""


# ---------------------------------------------------------------------------
# Copilot (project plan)
# ---------------------------------------------------------------------------

class RoadmapWeek(BaseModel):
    week: int
    focus: str
    deliverables: list[str] = Field(default_factory=list)


class ProjectPlan(BaseModel):
    """Full project plan for the chosen candidate (copilot output)."""

    candidate_id: str
    problem_statement: str
    proposed_solution: str
    agent_architecture: str
    tech_stack: list[str] = Field(default_factory=list)
    roadmap: list[RoadmapWeek] = Field(default_factory=list)
    mvp_scope: list[str] = Field(default_factory=list)
    demo_script: str
    one_line_pitch: str


# ---------------------------------------------------------------------------
# Bookkeeping / activity
# ---------------------------------------------------------------------------

class AgentActivity(BaseModel):
    agent: str
    status: Literal["running", "done", "error"] = "running"
    detail: str = ""
    ts: str = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ForgeMindState(BaseModel):
    """Shared state passed between LangGraph nodes.

    Fields that accumulate across nodes (chat messages, activity) use
    ``Annotated[..., operator.add]`` reducers; ``phase`` uses a last-wins
    reducer because the parallel novelty/feasibility branches both write
    it; everything else is replaced by the node that writes it. The full
    model is persisted after every node so a session can be resumed at
    any phase.
    """

    session_id: str = "default"
    phase: Annotated[
        Literal[
            "init", "interview", "scout", "generate", "evaluate",
            "critic", "rank", "present", "feedback", "plan", "done",
        ],
        _last_wins,
    ] = "init"
    loop_count: int = Field(default=0, description="regeneration/refinement loops used")
    max_loops: int = 3

    # Interview
    profile: Profile = Field(default_factory=Profile)
    interview_turns: list[InterviewTurn] = Field(default_factory=list)
    pending_question: Optional[str] = None
    pending_answers: list[str] = Field(
        default_factory=list, description="scripted/UI answers consumed one per visit"
    )
    interview_visits: int = 0

    # Research + generation
    problems: list[ProblemRecord] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    regeneration_feedback: str = ""

    # Evaluation
    novelty_reports: dict[str, NoveltyReport] = Field(default_factory=dict)
    feasibility_reports: dict[str, FeasibilityReport] = Field(default_factory=dict)
    critiques: dict[str, CritiqueReport] = Field(default_factory=dict)

    # Ranking + output
    ranking: Optional[RankingOutput] = None
    chosen_candidate_id: Optional[str] = None
    plan: Optional[ProjectPlan] = None

    # Feedback loop
    feedback_history: list[Feedback] = Field(default_factory=list)
    pending_feedback: list[Feedback] = Field(
        default_factory=list, description="swipes waiting to be consumed by the feedback node"
    )
    refinement_text: Optional[str] = None
    taste_weights: dict[str, float] = Field(
        default_factory=dict, description="cumulative weight adjustments from Taste Learner"
    )
    taste_vector: list[float] = Field(default_factory=list)
    taste_summary: str = ""

    # UX / observability (accumulate across nodes via reducers)
    chat: Annotated[list[dict[str, str]], operator.add] = Field(
        default_factory=list, description="chat transcript: [{role, content}]"
    )
    activity_log: Annotated[list[AgentActivity], operator.add] = Field(
        default_factory=list
    )

    def model_dump_json_safe(self) -> dict[str, Any]:
        """JSON-serializable dump (enums -> str) for SQLite persistence."""
        return self.model_dump(mode="json")
