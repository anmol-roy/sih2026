"""
Traditional Knowledge Matcher
───────────────────────────────
Searches the AYUSH / TK corpus (domain = "ayush" in Qdrant) for
formulations, ingredients, or uses that overlap with the invention.

Returns a list of (LegalChunk, score) pairs and a boolean
`traditional_knowledge_match` flag.

Design
------
- Uses the same Qdrant collection as Phase 2 (ip_sakti_legal),
  filtered to domain = "ayush".
- Runs component-level BM25 queries so individual ingredient names
  score highly even in long text.
- Runs a semantic query with the full invention summary for context.
- Merges and deduplicates, then returns the top matches with
  a per-chunk component-overlap list.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention, LegalChunk
from ingestion.bm25_store import BM25Store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "ip_sakti_legal"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_PATH     = str(Path(__file__).parent.parent.parent / "qdrant_db")
BM25_PATH       = Path(__file__).parent.parent.parent / "bm25_index.pkl"

# Minimum cosine similarity to count as a TK match
_TK_THRESHOLD = 0.28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _doc_to_chunk(doc: Document) -> LegalChunk:
    m = doc.metadata
    return LegalChunk(
        chunk_id      = m.get("chunk_id", "unknown"),
        document_id   = m.get("document_id", "unknown"),
        title         = m.get("title", "Unknown"),
        source        = m.get("source", "AYUSH"),
        source_url    = m.get("source_url"),
        document_type = m.get("document_type", "guideline"),
        domain        = m.get("domain", "ayush"),
        authority_level = m.get("authority_level", "secondary"),
        chapter       = m.get("chapter"),
        section       = m.get("section"),
        subsection    = m.get("subsection"),
        page          = m.get("page"),
        language      = m.get("language", "english"),
        version       = m.get("version"),
        effective_date= m.get("effective_date"),
        last_verified = m.get("last_verified", "2026-08-25"),
        text          = doc.page_content,
    )


def _component_overlap(invention: Invention, text: str) -> list[str]:
    tl = text.lower()
    return [c for c in invention.components if c.lower() in tl]


# ---------------------------------------------------------------------------
# TKMatcher
# ---------------------------------------------------------------------------

class TKMatcher:
    """
    Match an invention against the TK/AYUSH corpus.

    Parameters
    ----------
    embeddings    : shared HuggingFaceEmbeddings (reuse from app)
    top_k         : maximum TK results to return
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        top_k: int = 5,
    ):
        self._emb    = embeddings
        self._top_k  = top_k
        self._qdrant = QdrantClient(path=QDRANT_PATH)

        self._ayush_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value="ayush"))]
        )

        existing = {c.name for c in self._qdrant.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._vector_store = None
        else:
            self._vector_store = QdrantVectorStore(
                client          = self._qdrant,
                collection_name = COLLECTION_NAME,
                embedding       = self._emb,
            )

        try:
            self._bm25 = BM25Store.load(BM25_PATH)
        except FileNotFoundError:
            self._bm25 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, invention: Invention) -> dict:
        """
        Returns
        -------
        {
          "traditional_knowledge_match": bool,
          "matches": [
            {
              "chunk"             : LegalChunk,
              "score"             : float,
              "matched_components": list[str],
              "source"            : str,
              "page"              : int | None,
            }
          ]
        }
        """
        if not invention.components and not invention.intended_use:
            return {"traditional_knowledge_match": False, "matches": []}

        candidate_map: dict[str, tuple[LegalChunk, float]] = {}

        # ── BM25: one query per component ────────────────────────────────
        if self._bm25:
            for comp in invention.components:
                bm25_hits = self._bm25.query(comp, top_k=8)
                for i, chunk in enumerate(bm25_hits):
                    if chunk.domain != "ayush":
                        continue
                    score = 1.0 / (i + 1) * 0.4
                    cid = chunk.chunk_id
                    if cid not in candidate_map or score > candidate_map[cid][1]:
                        candidate_map[cid] = (chunk, score)

            # BM25 query with full use description
            if invention.intended_use:
                for i, chunk in enumerate(
                    self._bm25.query(invention.intended_use, top_k=8)
                ):
                    if chunk.domain != "ayush":
                        continue
                    score = 1.0 / (i + 1) * 0.35
                    cid = chunk.chunk_id
                    if cid not in candidate_map or score > candidate_map[cid][1]:
                        candidate_map[cid] = (chunk, score)

        # ── Vector: semantic query with invention summary ─────────────────
        if self._vector_store:
            summary = " ".join(filter(None, [
                " ".join(invention.components[:5]),
                invention.intended_use,
                invention.technical_field,
            ]))
            try:
                docs = self._vector_store.similarity_search_with_score(
                    summary,
                    k=10,
                    filter=self._ayush_filter,
                )
                for doc, vscore in docs:
                    chunk = _doc_to_chunk(doc)
                    cid   = chunk.chunk_id
                    prev  = candidate_map.get(cid, (chunk, 0.0))[1]
                    candidate_map[cid] = (chunk, max(prev, float(vscore) * 0.6))
            except Exception:
                pass

        if not candidate_map:
            return {"traditional_knowledge_match": False, "matches": []}

        # ── Semantic rerank ───────────────────────────────────────────────
        inv_text = " ".join(filter(None, [
            invention.title,
            " ".join(invention.components),
            invention.intended_use,
        ]))
        q_emb = self._emb.embed_query(inv_text)

        items = list(candidate_map.values())   # (chunk, base_score)
        c_embs = self._emb.embed_documents([c.text for c, _ in items])

        scored: list[tuple[LegalChunk, float, list[str]]] = []
        for (chunk, base), c_emb in zip(items, c_embs):
            sem   = _cosine(q_emb, c_emb)
            final = 0.5 * sem + 0.5 * base
            matched = _component_overlap(invention, chunk.text)
            scored.append((chunk, round(final, 4), matched))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Only keep chunks that actually contain at least one component
        # OR have a decent semantic score
        relevant = [
            (c, s, m) for c, s, m in scored
            if m or s >= _TK_THRESHOLD
        ]

        matches = [
            {
                "chunk"             : chunk,
                "score"             : score,
                "matched_components": matched,
                "source"            : chunk.source,
                "page"              : chunk.page,
            }
            for chunk, score, matched in relevant[: self._top_k]
        ]

        return {
            "traditional_knowledge_match": len(matches) > 0,
            "matches": matches,
        }
