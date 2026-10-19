"""Tests for the vector store (memory path in MOCK, Chroma path live)."""
from __future__ import annotations

import pytest

from forgemind.memory import vector_store


@pytest.fixture(autouse=True)
def _reset():
    vector_store.reset_memory_store()
    yield
    vector_store.reset_memory_store()


def test_embed_texts_deterministic_in_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    a = vector_store.embed_texts(["agentic workflow automation"])
    b = vector_store.embed_texts(["agentic workflow automation"])
    assert a == b
    assert len(a) == 1
    assert len(a[0]) == vector_store._HASH_DIM


def test_embed_empty_list() -> None:
    assert vector_store.embed_texts([]) == []


def test_cosine_similarity_bounds() -> None:
    v = [1.0, 0.0, 0.0]
    assert vector_store.cosine_similarity(v, v) == pytest.approx(1.0)
    assert vector_store.cosine_similarity(v, [0.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert vector_store.cosine_similarity([], [1.0]) == 0.0


def test_upsert_and_query_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    ids = ["a", "b"]
    docs = [
        "python llm agent orchestration pipeline",
        "knitting patterns for winter scarves",
    ]
    stored = vector_store.upsert_documents("t", ids, docs, [{"kind": "x"}, {"kind": "y"}])
    assert stored == 2
    hits = vector_store.query_similar("t", "python llm agent", k=2)
    assert hits
    assert hits[0]["id"] == "a"
    assert hits[0]["metadata"]["kind"] == "x"
    assert 0.0 <= hits[0]["distance"] <= 2.0


def test_query_respects_exclude_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    vector_store.upsert_documents("t", ["a", "b"], ["same text here", "same text here"])
    hits = vector_store.query_similar("t", "same text here", k=5, exclude_ids=["a"])
    assert all(h["id"] != "a" for h in hits)
    assert any(h["id"] == "b" for h in hits)


def test_query_unknown_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    assert vector_store.query_similar("missing", "text") == []


def test_upsert_no_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    assert vector_store.upsert_documents("t", [], []) == 0


def test_chroma_live_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode (mock off) exercises persistent ChromaDB with hash embeddings.

    embed_texts is stubbed to the hash embedder so no model download happens.
    """
    pytest.importorskip("chromadb")
    monkeypatch.setenv("FORGEMIND_MOCK", "0")
    monkeypatch.setenv("FORGEMIND_VECTOR_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(vector_store, "_chroma_client", None)
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts: [vector_store._hash_embed(t) for t in texts],
    )
    n = vector_store.upsert_documents(
        "live_test", ["x1", "x2"], ["rust async runtime internals", "cake recipes"]
    )
    assert n == 2
    hits = vector_store.query_similar("live_test", "rust async runtime", k=2)
    assert hits
    assert hits[0]["id"] == "x1"
    assert all(h["id"] != "nope" for h in hits)
