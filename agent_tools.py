"""Agent tool stubs for SBA Copilot.

These helpers expose retrieval, citation preview, email drafting, and CRM logging
capabilities so the orchestration layer (FastAPI, CLI, etc.) can call them directly.
They intentionally mirror the logic in the Streamlit demo but avoid Streamlit
dependencies so they can run in background services.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import faiss
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import CrossEncoder, SentenceTransformer

IDX_PATH = pathlib.Path("rag/index.faiss")
META_PATH = pathlib.Path("rag/meta.jsonl")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-7B-Instruct")

_emb_cache: Optional[SentenceTransformer] = None
_index_cache: Optional[faiss.Index] = None
_meta_cache: Optional[List[Dict[str, Any]]] = None
_reranker_cache: Optional[CrossEncoder] = None


class MissingArtifactError(FileNotFoundError):
    """Raised when required RAG artifacts are missing."""


@dataclass
class RetrievalHit:
    """Structured retrieval result."""

    score: float
    text: str
    metadata: Dict[str, Any]

    def citation_label(self, idx: int) -> str:
        doc_id = self.metadata.get("doc", self.metadata.get("doc_id", ""))
        page = self.metadata.get("page")
        parts = [f"[{idx}]", doc_id]
        if page:
            parts.append(f"p.{page}")
        return " ".join(str(p) for p in parts if p)


def _load_embeddings() -> SentenceTransformer:
    global _emb_cache
    if _emb_cache is None:
        _emb_cache = SentenceTransformer("thenlper/gte-large")
    return _emb_cache


def _load_index() -> faiss.Index:
    global _index_cache
    if _index_cache is None:
        if not IDX_PATH.exists():
            raise MissingArtifactError(f"Missing FAISS index at {IDX_PATH}")
        _index_cache = faiss.read_index(str(IDX_PATH))
    return _index_cache


def _load_metadata() -> List[Dict[str, Any]]:
    global _meta_cache
    if _meta_cache is None:
        if not META_PATH.exists():
            raise MissingArtifactError(f"Missing metadata file at {META_PATH}")
        with open(META_PATH, "r", encoding="utf-8") as fh:
            _meta_cache = [json.loads(line) for line in fh]
    return _meta_cache


def _load_reranker() -> Optional[CrossEncoder]:
    global _reranker_cache
    if _reranker_cache is None:
        try:
            _reranker_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker_cache = None
    return _reranker_cache


def _ensure_chunks_available():
    _load_embeddings()
    _load_index()
    _load_metadata()


def search_docs(query: str, k: int = 40) -> List[RetrievalHit]:
    """Vector search over indexed SBA corpus, returning top-k hits."""
    if not query.strip():
        return []
    _ensure_chunks_available()
    emb = _emb_cache
    index = _index_cache
    meta = _meta_cache

    qv = emb.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.asarray(qv, dtype="float32"), k)
    sims = (1 - distances[0] / 2.0).tolist()

    hits: List[RetrievalHit] = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        info = dict(meta[idx])
        info.setdefault("rank", rank)
        hits.append(RetrievalHit(score=float(sims[rank]), text=info.pop("text", ""), metadata=info))
    return hits


def rerank_hits(query: str, hits: Iterable[RetrievalHit], topn: int = 6) -> List[RetrievalHit]:
    """Rerank hits with cross-encoder when available."""
    hits = list(hits)
    if not hits:
        return []
    reranker = _load_reranker()
    if reranker is None:
        trimmed = sorted(hits, key=lambda h: -h.score)[:topn]
        for idx, hit in enumerate(trimmed):
            hit.metadata["rank"] = idx
        return trimmed

    scores = reranker.predict([(query, h.text) for h in hits])
    for hit, score in zip(hits, scores):
        hit.metadata["rr_score"] = float(score)
    hits.sort(key=lambda h: -h.metadata.get("rr_score", h.score))
    trimmed = hits[:topn]
    for idx, hit in enumerate(trimmed):
        hit.metadata["rank"] = idx
    return trimmed


def quality_score(query: str, hits: Iterable[RetrievalHit]) -> float:
    """Blend similarity and fuzzy coverage to produce a confidence score."""
    hits = list(hits)
    if not hits:
        return 0.0
    sim_values = [hit.metadata.get("rr_score", hit.score) for hit in hits]
    sim = float(np.mean(sim_values)) if sim_values else 0.0
    coverages = [fuzz.token_set_ratio(query, hit.text) for hit in hits]
    coverage = (max(coverages) if coverages else 0.0) / 100.0
    return round(0.6 * sim + 0.4 * coverage, 3)


def synthesize_answer(query: str, ctx_hits: List[RetrievalHit]) -> str:
    """Extractive answer with citations when no LLM is available."""
    if not ctx_hits:
        return "No supporting evidence found."
    ctx_chunks = [hit.text for hit in ctx_hits]
    best_idx = int(np.argmax([fuzz.partial_ratio(query, chunk) for chunk in ctx_chunks]))
    best_chunk = ctx_hits[best_idx].text
    pos = best_chunk.lower().find(query.split()[0].lower()) if query.split() else 0
    start = max(0, pos - 350)
    end = min(len(best_chunk), start + 750)
    citations = " ".join(hit.citation_label(idx + 1) for idx, hit in enumerate(ctx_hits))
    return f"{best_chunk[start:end].strip()}\n\nSources: {citations}"


def answer_query(query: str, k: int = 40, topn: int = 6) -> Dict[str, Any]:
    """Convenience wrapper for agent workflows."""
    t0 = time.time()
    hits = search_docs(query, k=k)
    top = rerank_hits(query, hits, topn=topn)
    answer = synthesize_answer(query, top)
    latency_ms = int((time.time() - t0) * 1000)
    score = quality_score(query, top)
    return {
        "question": query,
        "answer": answer,
        "quality": score,
        "sources": [open_source(hit) for hit in top],
        "latency_ms": latency_ms,
    }


def draft_email(recipient: str, subject: str, bullet_points: Iterable[str]) -> str:
    """Basic structured email draft helper."""
    bullets = "\n".join(f"• {point.strip()}" for point in bullet_points)
    return (
        f"Subject: {subject}\n\n"
        f"Hi {recipient},\n\n"
        f"Here’s a quick summary based on the latest SBA guidance:\n"
        f"{bullets}\n\n"
        "Please let me know if you need additional details or supporting documents.\n\n"
        "Best regards,\nSBA Copilot"
    )


def log_crm_note(payload: Dict[str, Any], path: pathlib.Path | None = None) -> pathlib.Path:
    """Append a CRM-style note to a JSONL log for auditing."""
    path = path or pathlib.Path("logs/crm_notes.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    note = payload | {"timestamp": time.time()}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(note, ensure_ascii=False) + "\n")
    return path


def open_source(hit: RetrievalHit) -> Dict[str, Any]:
    """Return a structured view suitable for tool responses."""
    return {
        "text": hit.text,
        "metadata": hit.metadata,
        "citation": hit.citation_label(hit.metadata.get("rank", 0) + 1),
    }


__all__ = [
    "RetrievalHit",
    "search_docs",
    "rerank_hits",
    "synthesize_answer",
    "answer_query",
    "draft_email",
    "log_crm_note",
    "open_source",
]
