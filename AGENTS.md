# ForgeMind — Agent Conventions

## Architecture
LangGraph state machine over specialized agents, all LLM calls via LiteLLM
(`llm/router.py`), all structured outputs are Pydantic v2 models in `state.py`.

## Rules for contributors / coding agents
1. Every LLM call goes through `llm/router.structured_call(role, prompt, schema, ...)`. Never call LiteLLM/Anthropic/OpenAI directly.
2. Roles are mapped in `config/models.yaml`. Add a role there before using it.
3. Set `FORGEMIND_MOCK=1` for deterministic fake outputs (tests, demos without keys).
4. One public function per agent module in `src/forgemind/agents/`; keep functions small and typed.
5. Every suggested problem statement must cite at least one evidence URL (`Candidate.evidence_ids` must be non-empty and resolve to `ProblemRecord.source_url`).
6. Never hardcode API keys; use `.env` (see `.env.example`).
7. State mutations happen only through the LangGraph state (`ForgeMindState`); persist after every node via `memory/db.py`.
8. Tests use MOCK mode and live in `tests/`, named `test_<module>.py`. Run `pytest` after every change.

## Commands
- Install: `uv sync` (or `pip install -r requirements.txt`)
- Test: `pytest`
- CLI interview: `python -m forgemind.cli`
- UI: `streamlit run src/forgemind/ui/app.py`
