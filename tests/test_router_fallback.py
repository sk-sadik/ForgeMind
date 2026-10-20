"""Fallback-chain, error-classification, and availability tests (monkeypatched litellm)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from forgemind.llm import router
from forgemind.llm.errors import LLMUnavailableError, classify_llm_error
from forgemind.state import ScoutOutput


class FakeProviderError(Exception):
    """Stand-in for a litellm/provider exception (has optional status_code)."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        if status is not None:
            self.status_code = status


def _response(payload_json: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload_json))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _scout_json() -> str:
    payload = router._mock_payload("ScoutOutput", "", {"num_problems": 1})
    return payload.model_dump_json()


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("FORGEMIND_MOCK", "0")
    router.reset_call_log()
    yield
    router.reset_call_log()


_CONFIG_TWO_MODELS = """
defaults: {temperature: 1.0, max_tokens: 1000}
roles:
  scout:
    models: [test/model-a, test/model-b]
    temperature: 1.0
"""


# ---------------------------------------------------------------------------
# 1. Fallback moves to the next model
# ---------------------------------------------------------------------------


def test_fallback_second_model_succeeds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(_CONFIG_TWO_MODELS, encoding="utf-8")
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)

    import litellm

    calls: list[str] = []

    def completion(model: str, **kwargs):
        calls.append(model)
        if model == "test/model-a":
            raise FakeProviderError(
                "Your credit balance is too low to access the Anthropic API.", 400
            )
        return _response(_scout_json())

    monkeypatch.setattr(litellm, "completion", completion)

    result = router.structured_call("scout", "find problems", ScoutOutput)

    assert isinstance(result, ScoutOutput)
    assert calls == ["test/model-a", "test/model-b"]
    assert router.last_model_used("scout") == "test/model-b"
    rec = router.get_call_log()[-1]
    assert rec.model == "test/model-b"
    assert rec.fallback_from == "test/model-a"


def test_fallback_on_model_not_found(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(_CONFIG_TWO_MODELS, encoding="utf-8")
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)

    import litellm

    calls: list[str] = []

    def completion(model: str, **kwargs):
        calls.append(model)
        if model == "test/model-a":
            raise FakeProviderError("model 'test/model-a' does not exist", 404)
        return _response(_scout_json())

    monkeypatch.setattr(litellm, "completion", completion)
    result = router.structured_call("scout", "x", ScoutOutput)
    assert isinstance(result, ScoutOutput)
    assert calls == ["test/model-a", "test/model-b"]


# ---------------------------------------------------------------------------
# 2. All-fail raises LLMUnavailableError with per-model details
# ---------------------------------------------------------------------------


def test_all_models_fail_raises_llm_unavailable(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(_CONFIG_TWO_MODELS, encoding="utf-8")
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)

    import litellm

    def completion(model: str, **kwargs):
        raise FakeProviderError(
            "Your credit balance is too low to access the Anthropic API.", 400
        )

    monkeypatch.setattr(litellm, "completion", completion)

    with pytest.raises(LLMUnavailableError) as excinfo:
        router.structured_call("scout", "x", ScoutOutput)

    err = excinfo.value
    assert err.role == "scout"
    assert [e["model"] for e in err.model_errors] == ["test/model-a", "test/model-b"]
    assert all(e["code"] == "no_credits" for e in err.model_errors)

    message = str(err)
    assert "test/model-a" in message and "test/model-b" in message
    assert "credit balance is too low" in message  # friendly, not a raw traceback
    assert "console.anthropic.com" in message  # fix suggestion


def test_llm_unavailable_error_lists_all_failure_kinds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
defaults: {temperature: 1.0}
roles:
  scout:
    models: [test/model-a, test/model-b]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)

    import litellm

    def completion(model: str, **kwargs):
        if model == "test/model-a":
            raise FakeProviderError("Incorrect API key provided: sk-bad", 401)
        raise FakeProviderError("Connection refused by provider", 10)

    monkeypatch.setattr(litellm, "completion", completion)

    with pytest.raises(LLMUnavailableError) as excinfo:
        router.structured_call("scout", "x", ScoutOutput)

    codes = [e["code"] for e in excinfo.value.model_errors]
    assert codes == ["invalid_key", "connection"]
    message = str(excinfo.value)
    assert "API key" in message
    assert "network/connectivity" in message or "network" in message


# ---------------------------------------------------------------------------
# 3. Friendly error classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (FakeProviderError("Your credit balance is too low to access the API.", 400), "no_credits"),
        (FakeProviderError("You exceeded your current quota: insufficient_quota", 429), "no_credits"),
        (FakeProviderError("Incorrect API key provided", 401), "invalid_key"),
        (FakeProviderError("Rate limit exceeded, slow down", 429), "rate_limit"),
        (FakeProviderError("The model `foo` does not exist", 404), "model_not_found"),
        (FakeProviderError("Connection refused by ollama at localhost:11434", 10), "ollama_down"),
        (FakeProviderError("Read timed out connecting to provider", 10), "connection"),
        (FakeProviderError("weird mystery failure", None), "unknown"),
    ],
)
def test_classify_llm_error_codes(exc: Exception, expected_code: str) -> None:
    code, friendly = classify_llm_error(exc)
    assert code == expected_code
    assert friendly


def test_friendly_messages_have_no_raw_traceback() -> None:
    code, message = classify_llm_error(
        FakeProviderError("Your credit balance is too low", 400)
    )
    assert code == "no_credits"
    assert "Traceback" not in message
    assert message.endswith(".")


# ---------------------------------------------------------------------------
# 4. cheap_mode
# ---------------------------------------------------------------------------

_CHEAP_CONFIG = """
defaults: {temperature: 1.0}
cheap_mode: true
cheap_model: test/cheap
cheap_roles: [scout, novelty, taste_learner]
roles:
  scout:
    models: [test/expensive-1, test/expensive-2]
  novelty:
    models: [test/novelty-model]
  critic:
    models: [test/critic-model]
"""


def test_cheap_mode_points_high_volume_roles_at_cheap_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(_CHEAP_CONFIG, encoding="utf-8")
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)

    assert router.get_llm("scout").models == ["test/cheap", "test/expensive-1", "test/expensive-2"]
    assert router.get_llm("novelty").models[0] == "test/cheap"
    assert router.get_llm("taste_learner").models[0] == "test/cheap"
    # roles outside cheap_roles are untouched
    assert router.get_llm("critic").models == ["test/critic-model"]


def test_cheap_mode_env_override_without_config_flag(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(_CHEAP_CONFIG.replace("cheap_mode: true", "cheap_mode: false"),
                   encoding="utf-8")
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)
    monkeypatch.delenv("FORGEMIND_CHEAP_MODE", raising=False)
    assert router.get_llm("scout").models[0] == "test/expensive-1"
    monkeypatch.setenv("FORGEMIND_CHEAP_MODE", "1")
    assert router.get_llm("scout").models[0] == "test/cheap"


def test_real_config_every_role_has_fallback_chain() -> None:
    chains = router.list_model_chains()
    assert chains
    for role, chain in chains.items():
        assert chain, f"role {role} has no models"
        assert all(isinstance(m, str) and "/" in m for m in chain), role


# ---------------------------------------------------------------------------
# 5. Startup availability check (NO paid calls)
# ---------------------------------------------------------------------------


def test_check_availability_makes_no_llm_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    def _forbidden(*args, **kwargs):
        raise AssertionError("check_availability must not make paid LLM calls")

    monkeypatch.setattr(litellm, "completion", _forbidden)
    router.check_availability()  # must not raise / call litellm


def test_check_availability_red_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status = router.check_availability()
    anthropic_entries = [
        e for entries in status.values() for e in entries
        if e["model"].startswith("anthropic/")
    ]
    assert anthropic_entries
    assert all(not e["ok"] for e in anthropic_entries)
    assert all("ANTHROPIC_API_KEY" in e["reason"] for e in anthropic_entries)


def test_check_availability_green_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    status = router.check_availability()
    for role, entries in status.items():
        assert entries, role
        assert any(e["ok"] for e in entries), role


def test_check_availability_ollama_unreachable_is_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_text = """
defaults: {temperature: 1.0}
roles:
  scout:
    models: [ollama/llama3.1:8b]
"""
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    cfg = tmp / "models.yaml"
    cfg.write_text(cfg_text, encoding="utf-8")
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)

    import httpx

    def _boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    status = router.check_availability()
    assert status["scout"][0]["ok"] is False
    assert "ollama serve" in status["scout"][0]["reason"]


# ---------------------------------------------------------------------------
# 6. Debug panel sees the model actually used
# ---------------------------------------------------------------------------


def test_debug_log_records_actual_model_after_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "models.yaml"
    cfg.write_text(_CONFIG_TWO_MODELS, encoding="utf-8")
    monkeypatch.setattr(router, "_CONFIG_PATH", cfg)

    import litellm
    from forgemind.agents import debug_log

    debug_log.clear()

    def completion(model: str, **kwargs):
        if model == "test/model-a":
            raise FakeProviderError("rate limit", 429)
        return _response(_scout_json())

    monkeypatch.setattr(litellm, "completion", completion)
    router.structured_call("scout", "x", ScoutOutput)

    entries = [e for e in debug_log.get_entries() if e["kind"] == "llm_call"]
    assert entries, "expected an llm_call debug entry"
    assert entries[-1]["model"] == "test/model-b"
    assert entries[-1]["role"] == "scout"
    assert entries[-1]["fallback_chain"] == ["test/model-a", "test/model-b"]
    debug_log.clear()
