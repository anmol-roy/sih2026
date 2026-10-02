"""
Prior Art Search  (Phase 4)
────────────────────────────
Generates combinatorial queries from invention features and builds a
candidate pool of prior-art documents.

Strategy
--------
1. Single-feature queries    (one per feature)
2. Two-feature combos        (pairs of components + use)
3. Three-feature combos      (triples)
4. Full invention query      (all features together)
5. IPC-scoped queries        (suggested IPC code + key terms)

All queries feed into hybrid BM25 + vector retrieval.
After deduplication and reranking the top-N candidates are returned.

The caller (Phase 4 pipeline) then runs feature-by-feature matching
against this candidate pool via patent_matcher.py.
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import (
    Invention, InventionFeature, ClaimRepresentation, PatentChunk
)
from ingestion.bm25_store import BM25Store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATENT_COLLECTION = "ip_sakti_patents"
LEGAL_COLLECTION  = "ip_sakti_legal"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_PATH       = str(Path(__file__).parent.parent.parent / "qdrant_db")
PATENT_BM25_PATH  = Path(__file__).parent.parent.parent / "bm25_patent_index.pkl"
LEGAL_BM25_PATH   = Path(__file__).parent.parent.parent / "bm25_index.pkl"


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def generate_prior_art_queries(
    features: list[InventionFeature],
    claim: ClaimRepresentation,
    invention: Invention,
    max_queries: int = 20,
) -> list[str]:
    """
    Generate a rich set of search queries from invention features.

    Returns a deduplicated list capped at *max_queries*.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)

    # ── 1. Single-feature queries ─────────────────────────────────────────
    for f in features:
        _add(f.feature)

    # ── 2. Two-feature combinations ───────────────────────────────────────
    feature_texts = [f.feature for f in features]
    for a, b in combinations(feature_texts[:6], 2):
        _add(f"{a} {b}")

    # ── 3. Three-feature combinations ─────────────────────────────────────
    for a, b, c in combinations(feature_texts[:5], 3):
        _add(f"{a} {b} {c}")

    # ── 4. Full invention query ────────────────────────────────────────────
    _add(" ".join(feature_texts[:8]))
    _add(claim.independent_claim)

    # ── 5. Use + component combinations ───────────────────────────────────
    use_features   = [f.feature for f in features if f.category == "use"]
    comp_features  = [f.feature for f in features if f.category == "component"]
    ratio_features = [f.feature for f in features if f.category == "ratio"]

    for use in use_features:
        for comp in comp_features[:3]:
            _add(f"{comp} {use}")
        if comp_features:
            _add(f"{' '.join(comp_features[:3])} {use}")

    # ── 6. IPC-scoped queries ─────────────────────────────────────────────
    for ipc in claim.ipc_suggested[:3]:
        _add(f"{ipc} {' '.join(feature_texts[:3])}")
        if use_features:
            _add(f"{ipc} {use_features[0]}")

    # ── 7. Ratio-specific ─────────────────────────────────────────────────
    for ratio in ratio_features:
        if comp_features:
            _add(f"{comp_features[0]} {ratio}")

    # ── 8. Title + technical field ────────────────────────────────────────
    _add(invention.title)
    if invention.technical_field:
        _add(f"{invention.technical_field} {' '.join(feature_texts[:4])}")

    return queries[:max_queries]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _doc_to_patent_chunk(doc: Document) -> PatentChunk:
    m = doc.metadata
    ipc_raw = m.get("ipc_codes", [])
    if isinstance(ipc_raw, str):
        ipc_raw = [ipc_raw]
    return PatentChunk(
        chunk_id           = m.get("chunk_id", "unknown"),
        document_id        = m.get("document_id", "unknown"),
        publication_number = m.get("publication_number", "UNKNOWN"),
        application_number = m.get("application_number"),
        title              = m.get("title", "Unknown Patent"),
        applicant          = m.get("applicant"),
        inventor           = m.get("inventor"),
        filing_date        = m.get("filing_date"),
        publication_date   = m.get("publication_date"),
        classification     = m.get("classification"),
        ipc_codes          = ipc_raw,
        status             = m.get("status"),
        source             = m.get("source", "IP India"),
        source_url         = m.get("source_url"),
        section_type       = m.get("section_type", "abstract"),
        claim_number       = m.get("claim_number"),
        page               = m.get("page"),
        text               = doc.page_content,
    )


def _date_filter(
    chunk: PatentChunk,
    cutoff_date: Optional[str],
) -> bool:
    """
    Return True if the document is a valid prior-art reference
    (i.e. published/filed BEFORE cutoff_date).

    cutoff_date format: "YYYY-MM-DD"
    Returns True (keep) when either no cutoff is set or we cannot determine date.
    """
    if not cutoff_date:
        return True
    date_str = chunk.filing_date or chunk.publication_date
    if not date_str:
        return True     # cannot filter, keep it
    # Compare as string — ISO dates sort lexicographically
    return date_str[:10] < cutoff_date[:10]


# ---------------------------------------------------------------------------
# PriorArtSearcher
# ---------------------------------------------------------------------------

class PriorArtSearcher:
    """
    Build a candidate pool of prior-art documents using combinatorial queries.

    Parameters
    ----------
    embeddings        : shared HuggingFaceEmbeddings
    top_k_per_query   : candidates pulled per query
    final_pool_size   : max candidates after dedup + rerank
    cutoff_date       : ISO date string — exclude documents filed/published after this
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        top_k_per_query: int = 10,
        final_pool_size: int = 20,
        cutoff_date: Optional[str] = None,
    ):
        self._emb            = embeddings
        self._top_k_per_q    = top_k_per_query
        self._pool_size      = final_pool_size
        self._cutoff         = cutoff_date

        from qdrant_singleton import get_qdrant_client
        self._qdrant  = get_qdrant_client()
        existing      = {c.name for c in self._qdrant.get_collections().collections}

        if PATENT_COLLECTION in existing:
            self._collection  = PATENT_COLLECTION
            bm25_path         = PATENT_BM25_PATH
        else:
            self._collection  = LEGAL_COLLECTION
            bm25_path         = LEGAL_BM25_PATH

        self._vector_store = QdrantVectorStore(
            client=self._qdrant,
            collection_name=self._collection,
            embedding=self._emb,
        )

        self._qdrant_filter = None
        if self._collection == LEGAL_COLLECTION:
            self._qdrant_filter = Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value="patent"))]
            )

        try:
            self._bm25 = BM25Store.load(bm25_path)
        except FileNotFoundError:
            self._bm25 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_candidate_pool(
        self,
        queries: list[str],
        invention: Invention,
        claim: ClaimRepresentation,
    ) -> list[PatentChunk]:
        """
        Run all queries, deduplicate, date-filter, rerank, return top-N.

        Returns list[PatentChunk] — the candidate pool.
        """
        candidate_map: dict[str, tuple[PatentChunk, float]] = {}

        # ── Collect from all queries ──────────────────────────────────────
        for query in queries:
            # Vector
            try:
                docs = self._vector_store.similarity_search_with_score(
                    query,
                    k=self._top_k_per_q,
                    filter=self._qdrant_filter,
                )
                for doc, vscore in docs:
                    chunk = _doc_to_patent_chunk(doc)
                    cid   = chunk.chunk_id
                    if cid not in candidate_map or vscore > candidate_map[cid][1]:
                        candidate_map[cid] = (chunk, float(vscore))
            except Exception:
                pass

            # BM25
            if self._bm25:
                for rank, chunk in enumerate(
                    self._bm25.query(query, top_k=self._top_k_per_q)
                ):
                    cid   = chunk.chunk_id
                    score = 1.0 / (rank + 1) * 0.5
                    prev  = candidate_map.get(cid, (chunk, 0.0))[1]
                    if score > prev:
                        candidate_map[cid] = (chunk, score)

        if not candidate_map:
            return []

        # ── Date filter ───────────────────────────────────────────────────
        filtered = {
            cid: (chunk, score)
            for cid, (chunk, score) in candidate_map.items()
            if _date_filter(chunk, self._cutoff)
        }
        if not filtered:
            filtered = candidate_map   # fallback: ignore date filter

        # ── Semantic rerank against full invention ────────────────────────
        inv_text = " ".join(filter(None, [
            invention.title,
            claim.independent_claim,
            " ".join(f.feature for f in []),   # features not directly available here
            invention.intended_use or "",
        ]))
        q_emb  = self._emb.embed_query(inv_text)
        items  = list(filtered.values())   # (chunk, base_score)
        c_embs = self._emb.embed_documents([c.text for c, _ in items])

        ranked: list[tuple[PatentChunk, float]] = []
        for (chunk, base), c_emb in zip(items, c_embs):
            sem    = _cosine(q_emb, c_emb)
            final  = 0.5 * sem + 0.5 * base
            ranked.append((chunk, final))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # ── Deduplicate by publication_number ─────────────────────────────
        seen_pubs: set[str] = set()
        pool: list[PatentChunk] = []
        for chunk, _ in ranked:
            pub = chunk.publication_number or chunk.chunk_id
            if pub not in seen_pubs:
                seen_pubs.add(pub)
                pool.append(chunk)
            if len(pool) >= self._pool_size:
                break

        return pool
