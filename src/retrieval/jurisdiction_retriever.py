"""
Jurisdiction Retriever  (Phase 6)
───────────────────────────────────
Extends DomainRetriever to add a `jurisdiction` dimension to every
Qdrant filter.

Retrieval modes
───────────────
  retrieve_india(query, domain)         → chunks where jurisdiction = "india"
  retrieve_international(query, domain) → chunks where jurisdiction = "international"
  retrieve_both(query, domain)          → { india: [...], international: [...] }
                                          (two independent searches, never merged)

The two result sets are ALWAYS kept separate so the caller (orchestrator)
can generate two independent answers and present them side-by-side.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from retrieval.retriever import HybridRetriever, _confidence, _SUFFICIENCY_THRESHOLD
from ingestion.schema import LegalChunk
from ingestion.bm25_store import BM25Store
from routing.jurisdiction import Jurisdiction

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "ip_sakti_legal"
QDRANT_PATH      = str(Path(__file__).parent.parent.parent / "qdrant_db")
BM25_PATH        = Path(__file__).parent.parent.parent / "bm25_index.pkl"

# Jurisdiction-specific system prompts (injected by orchestrator into LLM)
JURISDICTION_PROMPTS: dict[str, str] = {
    "india": (
        "Jurisdiction: INDIA\n"
        "You must answer ONLY using Indian law and Indian sources.\n"
        "Do NOT cite international treaties or foreign law as Indian law.\n"
        "If evidence is insufficient, abstain."
    ),
    "international": (
        "Jurisdiction: INTERNATIONAL\n"
        "You must answer ONLY using international instruments "
        "(WIPO, PCT, TRIPS, Paris Convention, etc.).\n"
        "Do NOT cite Indian national law as international law.\n"
        "If evidence is insufficient, abstain."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant filter builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_filter(
    domain: Optional[str],
    jurisdiction: Optional[str],
) -> Optional[Filter]:
    """
    Build a Qdrant Filter combining domain AND jurisdiction conditions.
    Returns None when neither is specified.
    """
    conditions = []
    if domain:
        conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    if jurisdiction:
        conditions.append(
            FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction))
        )
    if not conditions:
        return None
    return Filter(must=conditions)


# ─────────────────────────────────────────────────────────────────────────────
# Result builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(
    chunks: list[LegalChunk],
    jurisdiction: str,
) -> dict:
    sufficient = len(chunks) > 0
    if not chunks:
        conf = "low"
    elif len(chunks) >= 4:
        conf = "high"
    elif len(chunks) >= 2:
        conf = "medium"
    else:
        conf = "low"
    return {
        "chunks"      : chunks,
        "confidence"  : conf,
        "sufficient"  : sufficient,
        "jurisdiction": jurisdiction,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Jurisdiction Retriever
# ─────────────────────────────────────────────────────────────────────────────

class JurisdictionRetriever:
    """
    Retrieves chunks with jurisdiction-aware Qdrant filters.

    Wraps the existing HybridRetriever but overrides its filter logic
    to combine domain + jurisdiction into one compound Qdrant filter.

    Parameters
    ----------
    embeddings : shared HuggingFaceEmbeddings
    top_k      : max chunks per jurisdiction per query
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        top_k: int = 6,
    ):
        self._emb   = embeddings
        self._top_k = top_k

        from qdrant_singleton import get_qdrant_client
        self._qdrant = get_qdrant_client()
        existing = {c.name for c in self._qdrant.get_collections().collections}

        if COLLECTION_NAME in existing:
            self._vs = QdrantVectorStore(
                client=self._qdrant,
                collection_name=COLLECTION_NAME,
                embedding=self._emb,
            )
        else:
            self._vs = None

        try:
            self._bm25 = BM25Store.load(BM25_PATH)
        except FileNotFoundError:
            self._bm25 = None

    # ------------------------------------------------------------------
    # Core retrieval — single jurisdiction
    # ------------------------------------------------------------------

    def _retrieve_one(
        self,
        query: str,
        domain: Optional[str],
        jurisdiction: str,
    ) -> list[LegalChunk]:
        """
        Pull up to self._top_k chunks matching domain + jurisdiction.
        BM25 is filtered post-hoc (no jurisdiction field in the in-memory index)
        to chunks whose metadata jurisdiction matches.
        """
        chunks: dict[str, LegalChunk] = {}   # chunk_id → chunk

        # ── Vector search with compound filter ───────────────────────────
        if self._vs:
            filt = _build_filter(domain, jurisdiction)
            try:
                docs = self._vs.similarity_search(
                    query, k=self._top_k * 2, filter=filt
                )
                for doc in docs:
                    chunk = self._doc_to_chunk(doc)
                    if chunk.chunk_id not in chunks:
                        chunks[chunk.chunk_id] = chunk
            except Exception:
                pass

        # ── BM25 search with post-hoc jurisdiction filter ─────────────────
        if self._bm25:
            for chunk in self._bm25.query(query, top_k=self._top_k * 2):
                if chunk.chunk_id in chunks:
                    continue
                chunk_jur = getattr(chunk, "jurisdiction", "india") or "india"
                if chunk_jur != jurisdiction:
                    continue
                if domain:
                    if getattr(chunk, "domain", "") != domain:
                        continue
                chunks[chunk.chunk_id] = chunk

        return list(chunks.values())[: self._top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_india(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> dict:
        """Retrieve India-jurisdiction chunks."""
        chunks = self._retrieve_one(query, domain, "india")
        return _make_result(chunks, "india")

    def retrieve_international(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> dict:
        """Retrieve international-jurisdiction chunks."""
        chunks = self._retrieve_one(query, domain, "international")
        return _make_result(chunks, "international")

    def retrieve_both(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> dict:
        """
        Run two independent searches and return SEPARATE result sets.

        Returns
        -------
        {
          "india"        : { chunks, confidence, sufficient, jurisdiction },
          "international": { chunks, confidence, sufficient, jurisdiction },
        }
        The two sets are NEVER merged.
        """
        india_result = self.retrieve_india(query, domain)
        intl_result  = self.retrieve_international(query, domain)
        return {
            "india"        : india_result,
            "international": intl_result,
        }

    def retrieve_by_jurisdiction(
        self,
        query: str,
        jurisdiction: Jurisdiction,
        domain: Optional[str] = None,
    ) -> dict:
        """
        Dispatch to the correct retrieval method based on jurisdiction enum.
        For BOTH, returns the full both-dict.
        For INDIA/INTERNATIONAL, returns a single-jurisdiction result dict.
        For UNKNOWN, defaults to INDIA.
        """
        if jurisdiction == Jurisdiction.BOTH:
            return self.retrieve_both(query, domain)
        if jurisdiction == Jurisdiction.INTERNATIONAL:
            return self.retrieve_international(query, domain)
        return self.retrieve_india(query, domain)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_to_chunk(doc: Document) -> LegalChunk:
        m = doc.metadata
        return LegalChunk(
            chunk_id      = m.get("chunk_id", "unknown"),
            document_id   = m.get("document_id", "unknown"),
            title         = m.get("title", "Unknown"),
            source        = m.get("source", "Unknown"),
            source_url    = m.get("source_url"),
            document_type = m.get("document_type", "act"),
            domain        = m.get("domain", "general"),
            authority_level=m.get("authority_level", "primary"),
            chapter       = m.get("chapter"),
            section       = m.get("section"),
            subsection    = m.get("subsection"),
            page          = m.get("page"),
            language      = m.get("language", "english"),
            version       = m.get("version"),
            effective_date= m.get("effective_date"),
            last_verified = m.get("last_verified", "2026-08-25"),
            jurisdiction  = m.get("jurisdiction", "india"),
            text          = doc.page_content,
        )
