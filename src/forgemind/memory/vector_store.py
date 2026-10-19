"""Vector memory: ChromaDB + sentence-transformers (all-MiniLM-L6-v2).

Public entry points:
- :func:`embed_texts` — text -> vectors
- :func:`upsert_documents` — store documents in a named collection
- :func:`query_similar` — nearest-neighbour search over a collection

Mode behavior:
- MOCK mode uses a deterministic pure-Python hash embedder and an
  in-memory store (fast, hermetic tests — no model download, no disk).
- Live mode uses sentence-transformers + persistent ChromaDB, falling
  back to the hash embedder / memory store if the heavy stack is missing.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any, Optional

from ..llm import router

logger = logging.getLogger("forgemind.vector_store")

ST_MODEL_NAME = "all-MiniLM-L6-v2"
_HASH_DIM = 256

_st_model: Any = None
_memory_store: dict[str, dict[str, dict[str, Any]]] = {}
_chroma_client: Any = None
_warnings: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warnings:
        _warnings.add(key)
        logger.warning(message)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _hash_embed(text: str, dim: int = _HASH_DIM) -> list[float]:
    """Deterministic bag-of-tokens embedding (mock/fallback)."""
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h >> 8) % dim] += 0.5
        vec[(h >> 16) % dim] += 0.25
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _get_st_model():
    """Lazy-load the sentence-transformers model (live mode only)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(ST_MODEL_NAME)
    return _st_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts; returns one vector per input text."""
    if not texts:
        return []
    if router.mock_mode():
        return [_hash_embed(t) for t in texts]
    try:
        model = _get_st_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]
    except Exception as exc:  # noqa: BLE001 - degrade to hash embedder
        _warn_once("embed", f"sentence-transformers unavailable ({exc}); using hash embedder")
        return [_hash_embed(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Storage (ChromaDB live / in-memory mock)
# ---------------------------------------------------------------------------

def _vector_dir() -> str:
    path = os.getenv("FORGEMIND_VECTOR_DIR", "data/chroma")
    os.makedirs(path, exist_ok=True)
    return path


def _get_client():
    """Lazy persistent ChromaDB client (live mode)."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        _chroma_client = chromadb.PersistentClient(path=_vector_dir())
    return _chroma_client


def _collection(name: str):
    return _get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def _use_memory() -> bool:
    if router.mock_mode():
        return True
    try:
        import chromadb  # noqa: F401
    except ImportError:
        _warn_once("chroma", "chromadb unavailable; using in-memory vector store")
        return True
    return False


def upsert_documents(
    collection: str,
    ids: list[str],
    documents: list[str],
    metadatas: Optional[list[dict[str, Any]]] = None,
) -> int:
    """Insert/update documents in ``collection``; returns count stored."""
    if not ids:
        return 0
    assert len(ids) == len(documents)
    # Chroma dislikes empty metadata dicts; ensure at least one key.
    metadatas = [m if m else {"source": "forgemind"} for m in (metadatas or [])]
    while len(metadatas) < len(ids):
        metadatas.append({"source": "forgemind"})

    if _use_memory():
        bucket = _memory_store.setdefault(collection, {})
        vectors = embed_texts(documents)
        for doc_id, doc, meta, vec in zip(ids, documents, metadatas, vectors, strict=True):
            bucket[doc_id] = {"document": doc, "metadata": meta, "embedding": vec}
        return len(ids)

    embeddings = embed_texts(documents)
    _collection(collection).upsert(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )
    return len(ids)


def query_similar(
    collection: str, text: str, k: int = 5, exclude_ids: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Return up to ``k`` nearest documents to ``text``.

    Each hit: ``{"id", "document", "metadata", "distance"}`` (distance =
    1 - cosine; lower = more similar). Results in ``exclude_ids`` are
    skipped (e.g. the candidate querying about itself).
    """
    exclude = set(exclude_ids or [])
    if not text:
        return []

    if _use_memory():
        bucket = _memory_store.get(collection, {})
        if not bucket:
            return []
        query_vec = embed_texts([text])[0]
        scored = []
        for doc_id, item in bucket.items():
            if doc_id in exclude:
                continue
            sim = cosine_similarity(query_vec, item["embedding"])
            scored.append((sim, doc_id, item))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            {
                "id": doc_id,
                "document": item["document"],
                "metadata": item["metadata"],
                "distance": 1.0 - sim,
            }
            for sim, doc_id, item in scored[:k]
        ]

    coll = _collection(collection)
    count = coll.count()
    if count == 0:
        return []
    query_vec = embed_texts([text])[0]
    n = min(k + len(exclude), count)
    raw = coll.query(query_embeddings=[query_vec], n_results=n)
    hits: list[dict[str, Any]] = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
        if doc_id in exclude:
            continue
        hits.append(
            {"id": doc_id, "document": doc, "metadata": meta or {}, "distance": float(dist)}
        )
        if len(hits) >= k:
            break
    return hits


def reset_memory_store() -> None:
    """Clear the in-memory store (tests)."""
    _memory_store.clear()
