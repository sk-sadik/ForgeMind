"""LiteLLM-based model router.

Responsibilities:
- Map agent roles to an ordered model FALLBACK CHAIN via ``config/models.yaml``.
- ``structured_call(role, prompt, schema)`` -> validated Pydantic object.
- On billing/auth/rate-limit/model-not-found/connection errors, move to the
  next model in the chain; if all fail, raise :class:`LLMUnavailableError`
  listing every model and its friendly error (never a silent mock fallback).
- Automatic retry (max 2) per model on JSON/validation failure.
- MOCK mode (``FORGEMIND_MOCK=1``): deterministic fake outputs, no keys needed.
- Log token usage, estimated cost, and the model actually used per call.
- ``check_availability()``: key/reachability preflight (no paid calls).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, TypeAdapter, ValidationError

from .. import state as S
from .errors import FALLBACK_CODES, LLMUnavailableError, classify_llm_error

__all__ = [
    "LLMUnavailableError",
    "ModelSpec",
    "CallRecord",
    "structured_call",
    "get_llm",
    "list_models",
    "list_model_chains",
    "check_availability",
    "classify_llm_error",
    "last_model_used",
    "mock_mode",
    "get_call_log",
    "reset_call_log",
    "total_cost_usd",
]

# Load .env (if present) exactly once at router import. Existing environment
# variables always win (python-dotenv default: override=False), so tests and
# shells that set FORGEMIND_MOCK explicitly are unaffected.
load_dotenv()

logger = logging.getLogger("forgemind.router")

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 2
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "models.yaml"
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# Provider prefix -> env var required for a paid (non-Ollama) provider.
_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


@dataclass
class ModelSpec:
    """Resolved model configuration for a role (ordered fallback chain)."""

    role: str
    model: str = ""  # primary (first) model; kept for backward compatibility
    temperature: float = 0.4
    max_tokens: int = 2048
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    models: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.models:
            self.models = [self.model] if self.model else []
        elif not self.model:
            self.model = self.models[0]


@dataclass
class CallRecord:
    """One LLM (or mock) call record for usage/cost accounting."""

    role: str
    model: str
    schema_name: str
    duration_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    attempts: int = 1
    mock: bool = False
    fallback_from: str = ""  # primary model skipped before `model` (if any)


_call_log: list[CallRecord] = []
_last_model_used: dict[str, str] = {}


def get_call_log() -> list[CallRecord]:
    """Return the session-wide list of call records."""
    return list(_call_log)


def reset_call_log() -> None:
    """Clear recorded calls and last-model map (used by tests)."""
    _call_log.clear()
    _last_model_used.clear()


def last_model_used(role: str = "") -> str:
    """Model that actually served the most recent call ("" if none yet)."""
    if role:
        return _last_model_used.get(role, "")
    return next(reversed(_last_model_used.values()), "")


def total_cost_usd() -> float:
    """Sum of estimated cost across all recorded calls."""
    return sum(r.cost_usd for r in _call_log)


def mock_mode() -> bool:
    """True when FORGEMIND_MOCK is enabled."""
    return os.getenv("FORGEMIND_MOCK", "0").strip().lower() in {"1", "true", "yes", "on"}


def _cheap_mode_enabled(cfg: dict[str, Any]) -> bool:
    """cheap_mode comes from config/models.yaml or FORGEMIND_CHEAP_MODE=1."""
    if os.getenv("FORGEMIND_CHEAP_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return bool(cfg.get("cheap_mode", False))


def _role_models(cfg: dict[str, Any], role: str, role_cfg: dict[str, Any]) -> list[str]:
    """Resolve the ordered model chain for a role (cheap mode applied)."""
    raw = role_cfg.get("models")
    if isinstance(raw, str):
        models = [raw]
    elif isinstance(raw, (list, tuple)):
        models = [str(m).strip() for m in raw if str(m).strip()]
    else:
        single = role_cfg.get("model")
        models = [str(single).strip()] if single else []
    if not models:
        models = ["openai/gpt-4o-mini"]

    cheap_model = str(cfg.get("cheap_model") or "").strip()
    cheap_roles = [str(r) for r in (cfg.get("cheap_roles") or [])]
    if cheap_model and _cheap_mode_enabled(cfg) and role in cheap_roles:
        models = [cheap_model] + [m for m in models if m != cheap_model]
    return models


def list_model_chains() -> dict[str, list[str]]:
    """Role -> ordered model chain currently in effect (config/models.yaml)."""
    cfg = _load_config()
    defaults = dict(cfg.get("defaults") or {})
    chains: dict[str, list[str]] = {}
    for role, spec in (cfg.get("roles") or {}).items():
        role_cfg = dict(defaults)
        role_cfg.update(spec or {})
        chains[role] = _role_models(cfg, role, role_cfg)
    return chains


def list_models() -> dict[str, str]:
    """Role -> primary model mapping currently in effect (config/models.yaml)."""
    return {role: (chain[0] if chain else "")
            for role, chain in list_model_chains().items()}


def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"models config not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_llm(role: str) -> ModelSpec:
    """Resolve a role to a ModelSpec (ordered fallback chain) from models.yaml.

    Unknown roles fall back to a cheap default model so the graph never
    hard-crashes on a typo'd role.
    """
    cfg = _load_config()
    defaults = cfg.get("defaults", {})
    role_cfg = dict(defaults)
    role_cfg.update((cfg.get("roles") or {}).get(role, {}))
    if role not in (cfg.get("roles") or {}) and role != "default":
        logger.warning("role %r not in models.yaml; using defaults", role)
    models = _role_models(cfg, role, role_cfg)
    return ModelSpec(
        role=role,
        model=models[0],
        models=models,
        temperature=float(role_cfg.get("temperature", 0.4)),
        max_tokens=int(role_cfg.get("max_tokens", 2048)),
        input_cost_per_mtok=float(role_cfg.get("input_cost_per_mtok", 0.0)),
        output_cost_per_mtok=float(role_cfg.get("output_cost_per_mtok", 0.0)),
    )


# ---------------------------------------------------------------------------
# Startup availability check (NO paid calls)
# ---------------------------------------------------------------------------

_OLLAMA_TIMEOUT = 2.0


def _check_model(model: str) -> tuple[bool, str]:
    """(ok, reason) for one model — key presence or Ollama reachability."""
    provider = model.split("/", 1)[0].lower() if "/" in model else ""
    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            import httpx  # local import: router stays importable without it

            resp = httpx.get(f"{base}/api/tags", timeout=_OLLAMA_TIMEOUT)
            if resp.status_code == 200:
                return True, f"reachable ({base})"
            return False, f"HTTP {resp.status_code} from {base} — run `ollama serve`"
        except Exception as exc:  # noqa: BLE001 — any transport failure = down
            return False, f"not reachable ({type(exc).__name__}) — run `ollama serve`"
    env_var = _PROVIDER_KEY_ENV.get(provider)
    if env_var is None:
        return True, f"provider '{provider}': key cannot be verified without a call"
    key = os.getenv(env_var, "").strip()
    if not key:
        return False, f"{env_var} not set in .env"
    return True, f"{env_var} present"


def check_availability() -> dict[str, list[dict[str, Any]]]:
    """Preflight every role's model chain WITHOUT making paid LLM calls.

    Returns ``role -> [{"model", "ok", "reason"}, ...]``; a role is healthy
    when at least one model in its chain is ok (green), else red.
    """
    chains = list_model_chains()
    return {
        role: [{"model": m, **dict(zip(("ok", "reason"), _check_model(m)))} for m in chain]
        for role, chain in chains.items()
    }


# ---------------------------------------------------------------------------
# JSON extraction / validation
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """Parse a JSON object/array out of raw LLM text (fences tolerated)."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} or [...] span.
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise


def _validate(schema: Type[T], data: Any) -> T:
    if isinstance(data, schema):
        return data
    return schema.model_validate(data)


# ---------------------------------------------------------------------------
# MOCK payloads (deterministic; keyed by schema)
# ---------------------------------------------------------------------------

def _mock_url(i: int) -> str:
    return f"https://example.com/problems/{i}"


def _mock_payload(schema_name: str, prompt: str, ctx: dict[str, Any]) -> Any:
    """Deterministic fake structured output for a schema name."""
    if schema_name == "Profile":
        return S.Profile(
            interests=["AI agents", "developer tools", "hackathons"],
            skills=["Python", "LLMs", "web dev"],
            experience_level=S.ExperienceLevel.INTERMEDIATE,
            hours_per_week=15,
            team_size=2,
            goal=S.Goal.HACKATHON,
            constraints=["no paid APIs", "one weekend"],
            summary="Full-stack dev exploring agentic tooling.",
            is_complete=True,
        )
    if schema_name == "InterviewerOutput":
        answered = int(ctx.get("answered", 0))
        complete = bool(ctx.get("interview_complete", answered >= 3))
        if complete:
            profile = _mock_payload("Profile", prompt, ctx)
            return S.InterviewerOutput(
                next_question=None,
                profile_update=profile,
                is_complete=True,
                reason="mock interview complete",
            )
        profile = S.Profile(
            interests=["AI agents", "developer tools"][: max(1, min(answered, 2))],
            skills=["Python"][: min(answered, 1)],
            summary=f"mock partial profile after {answered} answers",
            is_complete=False,
        )
        questions = [
            "What kinds of problems do you personally keep running into?",
            "Which languages and tools are you most comfortable with?",
            "How many hours per week can you spend on a project?",
        ]
        return S.InterviewerOutput(
            next_question=questions[answered % len(questions)],
            profile_update=profile,
            is_complete=False,
            reason="mock: interview still in progress",
        )
    if schema_name == "ScoutOutput":
        topics = ["AI agents", "developer tools", "hackathons"]
        stypes: list[str] = ["web", "github", "reddit", "hn", "arxiv"]
        n = min(int(ctx.get("num_problems", 3)), 10)
        problems = [
            S.ProblemRecord(
                id=f"p{i}",
                title=f"Real pain point #{i} in {topics[(i - 1) % len(topics)]}",
                description=(
                    f"Developers report repeated friction with "
                    f"{topics[(i - 1) % len(topics)]}."
                ),
                evidence_summary=(
                    f"Multiple threads describe {topics[(i - 1) % len(topics)]} "
                    "as time-consuming."
                ),
                source_url=_mock_url(i),
                source_type=stypes[(i - 1) % len(stypes)],
                interest=topics[(i - 1) % len(topics)],
                relevance=0.8,
            )
            for i in range(1, n + 1)
        ]
        return S.ScoutOutput(problems=problems)
    if schema_name == "GeneratorOutput":
        style = str(ctx.get("style") or "").lower()
        if style not in {"impact", "technical", "fusion"}:
            style = (
                "fusion"
                if "fusion" in prompt.lower()
                else "technical"
                if "technical" in prompt.lower()
                else "impact"
            )
        n = int(ctx.get("num_candidates", 3))
        cands = []
        for i in range(1, n + 1):
            cands.append(
                S.Candidate(
                    id=f"{style}-{i}",
                    style=style,
                    title=f"{style.title()} PS #{i}: agentic helper for devs",
                    problem=f"Devs waste hours on repetitive {style} workflows (evidence p{i}).",
                    proposed_solution=f"An LLM agent that automates the {style} workflow end-to-end.",
                    target_users="Indie developers and hackathon teams",
                    agentic_angle="Multi-step planner + tool-using executor with human checkpoints",
                    evidence_ids=[f"p{i}"],
                    impact_hypothesis="Saves several hours per week for target users.",
                )
            )
        return S.GeneratorOutput(candidates=cands)
    if schema_name == "NoveltyReport":
        cid = str(ctx.get("candidate_id", "impact-1"))
        return S.NoveltyReport(
            candidate_id=cid,
            novelty_score=0.72,
            similar_projects=[
                S.SimilarProject(
                    name="Similar OSS tool",
                    url="https://github.com/example/similar-tool",
                    why_similar="Also targets workflow automation for devs.",
                    similarity=0.55,
                )
            ],
            differentiation="Focus on multi-agent handoffs and evidence-backed scoping.",
        )
    if schema_name == "FeasibilityReport":
        cid = str(ctx.get("candidate_id", "impact-1"))
        return S.FeasibilityReport(
            candidate_id=cid,
            feasibility_score=0.7,
            estimated_weeks=3,
            risks=["LLM API costs", "scope creep"],
            missing_skills=["vector DB ops"],
            required_apis=["LLM API"],
        )
    if schema_name == "CritiqueReport":
        cid = str(ctx.get("candidate_id", "impact-1"))
        return S.CritiqueReport(
            candidate_id=cid,
            attacks=["Evidence is thin beyond one thread", "Adjacent OSS tools exist"],
            critique_penalty=0.25,
            regret_prediction="Likely to over-scope the planner and miss the MVP demo.",
            impact_score=0.65,
        )
    if schema_name == "RankingOutput":
        ranked: list[S.ScoredCandidate] = ctx.get("ranked", [])
        weights: dict[str, float] = ctx.get("weights") or {
            "fit": 0.3, "novelty": 0.2, "feasibility": 0.2, "impact": 0.2, "critique": 0.1,
        }
        return S.RankingOutput(
            weights=weights,
            ranked=ranked[:3],
            summary=f"Mock ranking: top {min(len(ranked), 3)} by weighted score.",
        )
    if schema_name == "TasteUpdate":
        return S.TasteUpdate(
            taste_vector=[0.4, 0.6, 0.5, 0.3, 0.7],
            weight_adjustments={"novelty": 0.05},
            learned_summary="Mock: user prefers novel, agent-heavy, hackathon-scale ideas.",
        )
    if schema_name == "ProjectPlan":
        cid = str(ctx.get("candidate_id", "impact-1"))
        return S.ProjectPlan(
            candidate_id=cid,
            problem_statement="Developers waste hours on repetitive agentic workflows.",
            proposed_solution="A multi-agent helper that plans, executes, and verifies tasks.",
            agent_architecture="Planner -> Tool-executor -> Critic loop with shared state.",
            tech_stack=["Python", "LangGraph", "LiteLLM", "ChromaDB", "Streamlit"],
            roadmap=[
                S.RoadmapWeek(week=1, focus="Scaffold + core agent loop", deliverables=["CLI demo"]),
                S.RoadmapWeek(week=2, focus="Tools + memory", deliverables=["search + vector store"]),
                S.RoadmapWeek(week=3, focus="UI + polish", deliverables=["Streamlit app"]),
            ],
            mvp_scope=["Single agent loop", "One tool integration", "Minimal UI"],
            demo_script="1) Describe problem 2) Watch agent plan 3) Show verified result",
            one_line_pitch="Your AI teammate that turns vague ideas into validated project plans.",
        )
    raise ValueError(f"no mock payload registered for schema {schema_name!r}")


# ---------------------------------------------------------------------------
# Structured call
# ---------------------------------------------------------------------------

def structured_call(
    role: str,
    prompt: str,
    schema: Type[T],
    *,
    system: str | None = None,
    context: dict[str, Any] | None = None,
) -> T:
    """Run a structured LLM call for ``role`` and return a validated ``schema``.

    Tries each model in the role's fallback chain in order, moving to the
    next on billing/auth/rate-limit/model-not-found/connection errors.
    Each model retries up to MAX_RETRIES times on JSON/validation failure.
    If every model fails, raises :class:`LLMUnavailableError` listing each
    model and its friendly error (never a silent mock fallback). In MOCK
    mode returns a deterministic fake object with zero cost.
    """
    spec = get_llm(role)
    ctx = context or {}
    started = time.perf_counter()

    if mock_mode():
        payload = _mock_payload(schema.__name__, prompt, ctx)
        result = _validate(schema, payload)
        _record(spec, schema, started, attempts=1, mock=True, model=spec.model)
        _last_model_used[role] = spec.model
        return result

    # Real call via LiteLLM (imported lazily so mock mode needs no API stack).
    import litellm  # noqa: PLC0415

    # Some models reject sampling params entirely (e.g. the 5.5 family only
    # allows temperature=1). Drop unsupported params instead of raising.
    litellm.drop_params = True

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (system or "You are the {role} agent.").format(role=role)
            + " Respond with a single JSON object matching this JSON Schema. No prose.",
        },
        {"role": "user", "content": f"{prompt}\n\nJSON Schema:\n{schema_json}"},
    ]

    model_errors: list[dict[str, Any]] = []
    for model in spec.models:
        try:
            result, attempts, prompt_tokens, completion_tokens = _attempt_model(
                litellm, model, spec, messages, schema, role
            )
        except Exception as exc:  # noqa: BLE001 — classified per model below
            if isinstance(exc, (json.JSONDecodeError, ValidationError, ValueError)):
                code = "invalid_output"
                friendly = f"invalid JSON output for schema {schema.__name__}: {exc}"
            else:
                code, friendly = classify_llm_error(exc)
            model_errors.append(
                {
                    "model": model,
                    "code": code,
                    "message": friendly,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            logger.warning(
                "model %s failed for role=%s [%s]: %s — %s",
                model, role, code, friendly,
                "trying next model" if code in FALLBACK_CODES else "giving up",
            )
            continue

        cost = (
            prompt_tokens / 1e6 * spec.input_cost_per_mtok
            + completion_tokens / 1e6 * spec.output_cost_per_mtok
        )
        _record(
            spec, schema, started, attempts=attempts, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost_usd=cost,
            fallback_from=(spec.models[0] if spec.models[0] != model else ""),
        )
        _last_model_used[role] = model
        _record_debug_call(role, model, schema.__name__, attempts, spec.models)
        if model_errors:
            logger.warning(
                "role=%s recovered on fallback model %s after %d failure(s)",
                role, model, len(model_errors),
            )
        return result

    raise LLMUnavailableError(role, model_errors)


def _attempt_model(
    litellm: Any,
    model: str,
    spec: ModelSpec,
    messages: list[dict[str, str]],
    schema: Type[T],
    role: str,
) -> tuple[T, int, int, int]:
    """Call one model with per-model JSON retries.

    Returns ``(result, attempts, prompt_tokens, completion_tokens)``.
    API/provider errors propagate immediately (handled by the fallback
    chain in :func:`structured_call`); only JSON/validation failures retry.
    """
    working = list(messages)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + up to 2 retries
        response = litellm.completion(
            model=model,
            messages=working,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        try:
            data = _extract_json(text)
            result = _validate(schema, data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "structured_call retry %d/%d role=%s model=%s error=%s",
                attempt, MAX_RETRIES + 1, role, model, exc,
            )
            working.append({"role": "assistant", "content": text})
            working.append(
                {
                    "role": "user",
                    "content": (
                        "Your reply was invalid: "
                        f"{exc}\nReturn corrected JSON only."
                    ),
                }
            )
            continue
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return result, attempt, prompt_tokens, completion_tokens
    raise ValueError(
        f"model {model} returned invalid output for role={role} "
        f"schema={schema.__name__} after {MAX_RETRIES + 1} attempts: {last_error}"
    )


def _record_debug_call(
    role: str, model: str, schema_name: str, attempts: int, chain: list[str]
) -> None:
    """Push the model actually used into the UI debug-panel registry."""
    try:
        from ..agents import debug_log  # lazy: avoids import cycles
    except Exception:  # noqa: BLE001 — debug logging must never break calls
        return
    debug_log.record(
        "llm_call",
        role=role,
        model=model,
        schema=schema_name,
        attempts=attempts,
        fallback_chain=chain,
    )


def _record(
    spec: ModelSpec,
    schema: Type[BaseModel],
    started: float,
    *,
    attempts: int,
    mock: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    model: str = "",
    fallback_from: str = "",
) -> None:
    rec = CallRecord(
        role=spec.role,
        model=model or spec.model,
        schema_name=schema.__name__,
        duration_ms=int((time.perf_counter() - started) * 1000),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        attempts=attempts,
        mock=mock,
        fallback_from=fallback_from,
    )
    _call_log.append(rec)
    logger.info(
        "llm call role=%s model=%s schema=%s attempts=%d tokens=%d/%d cost=$%.6f mock=%s%s",
        rec.role, rec.model, rec.schema_name, rec.attempts,
        rec.prompt_tokens, rec.completion_tokens, rec.cost_usd, rec.mock,
        f" (after fallback from {rec.fallback_from})" if rec.fallback_from else "",
    )
