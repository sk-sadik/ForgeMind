# ForgeMind

Multi-agent system that interviews a developer, researches **real** problems on
the web (with evidence URLs), generates and critically evaluates project
problem-statement candidates, learns your taste from swipe feedback, and turns
your accepted idea into a full project plan.

Built with **LangGraph** (state machine) + **LiteLLM** (model routing) +
**Pydantic v2** (all structured outputs) + **ChromaDB/sentence-transformers**
(novelty + taste embeddings) + **Streamlit** (UI) + **SQLite** (session
persistence).

## Quick start

```bash
cp .env.example .env        # add keys, or just set FORGEMIND_MOCK=1
uv sync                     # or: pip install -r requirements.txt
pytest                      # 155 tests, fully in MOCK mode, no keys needed
```

### Mock vs live mode

| Mode | How | What happens |
|---|---|---|
| Mock | `FORGEMIND_MOCK=1` | Deterministic fake LLM/tool outputs; no keys, no network. Used by all tests. |
| Live | `FORGEMIND_MOCK=0` + `.env` keys | Real LLM calls via LiteLLM, Tavily/DDG search, ChromaDB embeddings. |

Keys are read from `.env` only (see `.env.example`) — never hardcode them.

### Run the UI

```bash
streamlit run src/forgemind/ui/app.py
```

Four steps: **Interview → Research & ranking → Review candidates (swipe
cards) → Project plan**. Like/dislike feed the Taste Learner (better future
rankings), *refine* regenerates ideas (max 3 loops), *accept* builds the plan.

### Run the CLI interview

```bash
python -m forgemind.cli              # interactive interview
python -m forgemind.cli --auto       # scripted demo answers (works with MOCK=1)
python -m forgemind.cli --session s42 --show-profile
```

## Architecture

```
interview ──▶ scout ──▶ generate ──▶ novelty ║ feasibility   (parallel)
                                                ‖
                                          filter gate ──(too few survivors & loops left)──▶ generate
                                                │
                                             critic ──▶ ranker ──▶ present ──▶ feedback
                                                                               │     │
                                                          (refine + loops left)│     │(accept)
                                                                               ▼     ▼
                                                          taste ──▶ generate    copilot ──▶ END
```

One public function per agent module in `src/forgemind/agents/`:

| Agent | Role |
|---|---|
| `interviewer` | Adaptive multi-turn interview → developer `Profile` (max 8 questions) |
| `scout` | Web/Reddit/HN/GitHub/arXiv/Devpost sweep → `ProblemRecord`s, every one with an evidence URL |
| `generators` | 3 styles (impact / technical / fusion) on 3 different models → candidates |
| `novelty` | ChromaDB k-NN against past candidates + Devpost/GitHub scans → `NoveltyReport` |
| `feasibility` | Skills, risks, timeline → `FeasibilityReport` |
| `critic` | Attacks surviving candidates → `CritiqueReport` (different model than any generator) |
| `ranker` | Weighted top-3 (weights from goal **+ learned taste**) → `RankingOutput` |
| `taste_learner` | Embedding taste vector (`mean(likes) − mean(dislikes)`) + weight adjustments |
| `copilot` | Accepted idea → `ProjectPlan` (architecture, tech stack, roadmap, MVP, demo, pitch) |

Key conventions (see `AGENTS.md`):

- Every LLM call goes through `llm/router.structured_call(role, prompt, schema)`
  — roles are mapped in `config/models.yaml`, retry/parse/cost logging included.
- Every suggested problem statement cites ≥1 evidence URL.
- State (`ForgeMindState`, all Pydantic) is persisted to SQLite after **every**
  graph node, so any session can be resumed at any phase.

## Project layout

```
config/models.yaml      role → model mapping (+ pricing)
src/forgemind/
  state.py              all Pydantic v2 schemas + LangGraph state
  graph.py              graph assembly, routing, regeneration gate
  cli.py                interview CLI
  llm/router.py         LiteLLM wrapper, MOCK mode, retries, cost log
  agents/               one public function per agent (see table above)
  memory/               SQLite persistence (profile/state/feedback/taste) + vector store
  tools/                web search, Reddit/HN, GitHub, arXiv, Devpost
  ui/                   Streamlit app + pure presentation helpers
tests/                  155 tests, all in MOCK mode
```

## Status

All 9 milestones are complete: scaffold & router → interview & DB → scout &
tools → generators & evaluation → critic/ranker/pipeline → taste learner &
feedback loop → copilot plan → Streamlit UI → docs. **155 tests passing.**
