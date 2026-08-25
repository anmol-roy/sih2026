"""
Patent Searcher
────────────────
Searches the patent corpus (stored in Qdrant under collection
"ip_sakti_patents") using hybrid BM25 + vector retrieval, then
scores each result for component/use/field overlap with the invention.

Returns a ranked list of patent similarity records.
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
from ingestion.schema import Invention, PatentChunk
from ingestion.bm25_store import BM25Store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATENT_COLLECTION  = "ip_sakti_patents"
LEGAL_COLLECTION   = "ip_sakti_legal"   # fallback when patent DB is empty
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_PATH        = str(Path(__file__).parent.parent.parent / "qdrant_db")
PATENT_BM25_PATH   = Path(__file__).parent.parent.parent / "bm25_patent_index.pkl"
LEGAL_BM25_PATH    = Path(__file__).parent.parent.parent / "bm25_index.pkl"


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
    return PatentChunk(
        chunk_id          = m.get("chunk_id", "unknown"),
        document_id       = m.get("document_id", "unknown"),
        publication_number= m.get("publication_number", "UNKNOWN"),
        application_number= m.get("application_number"),
        title             = m.get("title", "Unknown Patent"),
        applicant         = m.get("applicant"),
        inventor          = m.get("inventor"),
        filing_date       = m.get("filing_date"),
        publication_date  = m.get("publication_date"),
        classification    = m.get("classification"),
        status            = m.get("status"),
        source            = m.get("source", "IP India"),
        source_url        = m.get("source_url"),
        section_type      = m.get("section_type", "abstract"),
        claim_number      = m.get("claim_number"),
        page              = m.get("page"),
        text              = doc.page_content,
    )


def _component_overlap(
    invention: Invention,
    text: str,
) -> list[str]:
    """Return which of the invention's components appear in *text*."""
    text_lower = text.lower()
    return [c for c in invention.components if c.lower() in text_lower]


def _similarity_score(
    invention: Invention,
    chunk_text: str,
    semantic_score: float,
    matched_components: list[str],
) -> float:
    """
    Weighted similarity:
      50% semantic (embedding cosine)
      30% component overlap ratio
      20% use/field keyword match
    """
    comp_ratio = (
        len(matched_components) / len(invention.components)
        if invention.components else 0.0
    )

    use_text   = (invention.intended_use   or "").lower()
    field_text = (invention.technical_field or "").lower()
    text_lower = chunk_text.lower()
    use_hit    = 1.0 if use_text   and use_text   in text_lower else 0.0
    field_hit  = 1.0 if field_text and field_text in text_lower else 0.0
    keyword_score = (use_hit + field_hit) / 2.0

    return round(
        0.50 * semantic_score
        + 0.30 * comp_ratio
        + 0.20 * keyword_score,
        4,
    )


# ---------------------------------------------------------------------------
# PatentSearcher
# ---------------------------------------------------------------------------

class PatentSearcher:
    """
    Search the patent corpus for documents similar to a given invention.

    Parameters
    ----------
    embeddings      : shared HuggingFaceEmbeddings instance (reuse from app)
    top_k_retrieval : candidates pulled from each retrieval method
    top_k_final     : results returned after scoring
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        top_k_retrieval: int = 10,
        top_k_final: int = 5,
    ):
        self._emb          = embeddings
        self._top_k_ret    = top_k_retrieval
        self._top_k_final  = top_k_final
        self._qdrant       = QdrantClient(path=QDRANT_PATH)
        self._bm25: Optional[BM25Store] = None
        self._use_patents   = False  # flipped to True if patent collection exists

        # ── Connect to whichever collection is available ──────────────────
        existing = {c.name for c in self._qdrant.get_collections().collections}

        if PATENT_COLLECTION in existing:
            self._collection = PATENT_COLLECTION
            self._use_patents = True
            bm25_path = PATENT_BM25_PATH
        else:
            # Fall back to legal collection filtered by document_type=patent
            self._collection = LEGAL_COLLECTION
            bm25_path = LEGAL_BM25_PATH

        self._vector_store = QdrantVectorStore(
            client         = self._qdrant,
            collection_name= self._collection,
            embedding      = self._emb,
        )

        try:
            self._bm25 = BM25Store.load(bm25_path)
        except FileNotFoundError:
            pass   # BM25 is optional for patent search

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        invention: Invention,
        queries: list[str],
    ) -> list[dict]:
        """
        Run hybrid search over all *queries* and return a ranked list of:
        {
          "chunk"              : PatentChunk | LegalChunk,
          "similarity_score"  : float,
          "matched_components": list[str],
          "matched_use"       : str,
          "matched_technology": str,
          "publication_number": str,
          "title"             : str,
          "classification"    : str,
          "source"            : str,
        }
        """
        if not queries:
            return []

        # ── Collect candidates from all queries ──────────────────────────
        candidate_map: dict[str, tuple] = {}   # chunk_id → (chunk, sem_score)

        # Qdrant filter: if using legal collection, restrict to patent domain
        qdrant_filter = None
        if not self._use_patents and self._collection == LEGAL_COLLECTION:
            qdrant_filter = Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value="patent"))]
            )

        for query in queries[:6]:   # cap queries to avoid rate limits
            # Vector retrieval
            try:
                docs = self._vector_store.similarity_search_with_score(
                    query,
                    k=self._top_k_ret,
                    filter=qdrant_filter,
                )
                for doc, vscore in docs:
                    chunk = _doc_to_patent_chunk(doc)
                    cid   = chunk.chunk_id
                    if cid not in candidate_map or vscore > candidate_map[cid][1]:
                        candidate_map[cid] = (chunk, float(vscore))
            except Exception:
                pass

            # BM25 retrieval
            if self._bm25:
                bm25_results = self._bm25.query(query, top_k=self._top_k_ret)
                for i, c in enumerate(bm25_results):
                    bm25_score = 1.0 / (i + 1)   # rank-based score
                    cid = c.chunk_id
                    # If already found by vector, keep higher score
                    existing_score = candidate_map.get(cid, (None, 0.0))[1]
                    combined = max(existing_score, bm25_score * 0.4)
                    candidate_map[cid] = (c, combined)

        if not candidate_map:
            return []

        # ── Semantic rerank with invention summary ────────────────────────
        inv_text = " ".join(filter(None, [
            invention.title,
            invention.technical_field,
            invention.intended_use,
            " ".join(invention.components),
        ]))
        q_emb = self._emb.embed_query(inv_text)

        results = []
        chunk_texts = [(cid, chunk, score) for cid, (chunk, score)
                       in candidate_map.items()]

        c_embs = self._emb.embed_documents([c.text for _, c, _ in chunk_texts])

        for (cid, chunk, base_score), c_emb in zip(chunk_texts, c_embs):
            sem_score  = _cosine(q_emb, c_emb)
            matched    = _component_overlap(invention, chunk.text)
            final_sim  = _similarity_score(invention, chunk.text, sem_score, matched)

            results.append({
                "chunk"              : chunk,
                "similarity_score"   : final_sim,
                "matched_components" : matched,
                "matched_use"        : invention.intended_use or "",
                "matched_technology" : invention.technical_field or "",
                "publication_number" : getattr(chunk, "publication_number", ""),
                "title"              : chunk.title,
                "classification"     : getattr(chunk, "classification", "") or "",
                "source"             : chunk.source,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Deduplicate by publication_number (keep highest-scoring section)
        seen_pubs: dict[str, int] = {}
        deduped = []
        for r in results:
            pub = r["publication_number"] or r["chunk"].chunk_id
            if pub not in seen_pubs:
                seen_pubs[pub] = len(deduped)
                deduped.append(r)

        return deduped[: self._top_k_final]
