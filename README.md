# ForgeMind

Multi-agent system that interviews a developer, researches real problems,
generates validated project problem statements (PS), ranks them, and produces
a full project plan.

> Setup and demo instructions land in Milestone 9. Current status: Milestone 1
> (scaffold, state schemas, model router with MOCK mode) — 17 tests passing.

## Quick start

```bash
cp .env.example .env          # add keys, or set FORGEMIND_MOCK=1
uv sync                       # or: pip install -r requirements.txt
pytest                        # runs fully in MOCK mode, no keys needed
```
