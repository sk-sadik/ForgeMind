"""Friendly LLM error classification and the LLMUnavailableError type.

``classify_llm_error`` maps raw provider/litellm exceptions to a stable
code + human message. ``LLMUnavailableError`` is raised when every model in
a role's fallback chain failed; its message lists each model with its
friendly error and a concrete fix suggestion. Never silently falls back to
mock data.
"""
from __future__ import annotations

from typing import Any

# Codes that justify moving to the next model in the fallback chain.
FALLBACK_CODES = frozenset(
    {"no_credits", "invalid_key", "rate_limit", "model_not_found", "connection", "ollama_down"}
)

_CODE_SUGGESTIONS = {
    "no_credits": (
        "Top up credits at https://console.anthropic.com -> Plans & Billing "
        "(or the provider's billing page), then retry."
    ),
    "invalid_key": (
        "Fix the API key in .env (e.g. ANTHROPIC_API_KEY=...) — the current "
        "key is missing, invalid, or lacks access to this model."
    ),
    "rate_limit": "Wait a minute for the rate limit to reset, then retry.",
    "model_not_found": (
        "Edit config/models.yaml and use a model your key can access "
        "(check the provider's model list), then retry."
    ),
    "ollama_down": "Start the local Ollama server with `ollama serve`, then retry.",
    "connection": "Check network connectivity / VPN / proxy settings, then retry.",
    "invalid_output": "The model returned invalid JSON repeatedly — try a different model.",
    "unknown": "Inspect the exception details below and check config/models.yaml.",
}


def classify_llm_error(exc: Exception) -> tuple[str, str]:
    """Map an LLM/provider exception to ``(code, friendly_message)``.

    Works without litellm installed (uses status_code/message/name sniffing).
    """
    name = type(exc).__name__.lower()
    msg = str(exc)
    low = msg.lower()
    status = getattr(exc, "status_code", None)

    if "credit balance is too low" in low or "insufficient_quota" in low or (
        "exceeded your current quota" in low
    ):
        return "no_credits", "API credit balance is too low (out of funds)."
    if status == 429 or "rate limit" in low or "rate_limit" in low or "too many requests" in low:
        return "rate_limit", "Rate limit hit — wait a moment and retry."
    if (
        status in (401, 403)
        or "authenticationerror" in name
        or "permissiondenied" in name
        or "invalid api key" in low
        or "unauthorized" in low
        or "api key" in low and ("invalid" in low or "missing" in low or "wrong" in low)
    ):
        return "invalid_key", "API key is missing, invalid, or lacks permission for this model."
    if (
        status == 404
        or "notfounderror" in name
        or "model" in low
        and (
            "not found" in low
            or "does not exist" in low
            or "invalid model" in low
            or "not supported" in low
            or "no endpoints found" in low
        )
    ):
        return "model_not_found", f"Model not found or not enabled for this API key ({msg[:120]})."
    if "ollama" in low:
        return "ollama_down", "Ollama is not reachable — start it with `ollama serve`."
    if (
        isinstance(exc, (ConnectionError, TimeoutError, OSError))
        or "connection" in low
        or "connect" in low
        or "timeout" in low
        or "apiconnectionerror" in name
        or "timeout" in name
    ):
        return "connection", "Could not reach the LLM provider (network/connection problem)."
    return "unknown", f"{type(exc).__name__}: {msg[:200]}"


class LLMUnavailableError(RuntimeError):
    """Every model in a role's fallback chain failed.

    Attributes:
        role: the agent role that was being called.
        model_errors: one dict per failed model —
            ``{"model", "code", "message", "exception"}``.
    """

    def __init__(self, role: str, model_errors: list[dict[str, Any]]):
        self.role = role
        self.model_errors = list(model_errors)
        codes = sorted({e.get("code", "unknown") for e in self.model_errors})
        lines = [
            f"All {len(self.model_errors)} model(s) failed for role '{role}':",
        ]
        for entry in self.model_errors:
            lines.append(
                f"  - {entry.get('model')}: {entry.get('message')} "
                f"[{entry.get('exception', '')}]"
            )
        lines.append("How to fix: " + " ".join(
            _CODE_SUGGESTIONS.get(code, _CODE_SUGGESTIONS["unknown"]) for code in codes
        ))
        super().__init__("\n".join(lines))
