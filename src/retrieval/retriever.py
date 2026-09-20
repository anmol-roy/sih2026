"""
Phase 2 Hybrid Retriever
─────────────────────────
Query
  │
  ├─ classify ──► exact legal query?
  │                      │
  │              yes     │     no
  │              ▼       │     ▼
  │         BM25 top-10  │  Vector top-10
  │              │       │     │
  │              └───────┴─────┘
  │                      │
  │                merge & deduplicate (up to 20 candidates)
  │                      │
  │               cross-encoder rerank
  │                      │
  │                   top-k (default 5)
  │                      │
  │              evidence sufficiency check
  │                      │
  │            ┌─────────┴──────────┐
  │            │                    │
  │       sufficient            insufficient
  │            │                    │
  │       return chunks        return []
  │
  └─► caller decides whether to call LLM
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion import BM25Store, LegalChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "ip_sakti_legal"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_PATH = str(Path(__file__).parent.parent.parent / "qdrant_db")
BM25_PATH = Path(__file__).parent.parent.parent / "bm25_index.pkl"

# Minimum score threshold below which we declare "insufficient evidence"
_SUFFICIENCY_THRESHOLD = 0.30

# Patterns that suggest the user is asking for an exact legal provision
_EXACT_QUERY_RE = re.compile(
    r"""
    (section|sec\.?|rule|article|clause|sub-?section|schedule|form)\s*
    [\(\[]?\s*\d+[A-Z]?\s*[\)\]]?              # e.g. Section 3, Rule 14
    (?:\s*[\(\[]\s*[a-z]{1,3}\s*[\)\]])?       # optional sub-clause (p)
    | \bIPC\s+[A-Z]\d+[a-z/]+                   # IPC codes like A61K
    | \b[A-Z]\d{2}[A-Z]?\s*\d*/\d+             # patent classification codes
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Domain keywords → Qdrant filter values
_DOMAIN_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpatent\b|\bpatentab", re.I), "patent"),
    (re.compile(r"\btrademark\b|\btrade\s*mark\b", re.I), "trademark"),
    (re.compile(r"\bcopyright\b", re.I), "copyright"),
    (re.compile(r"\bdesign\b", re.I), "design"),
    (re.compile(r"\bgeograph|\bgi\b|\bindication", re.I), "gi"),
    (re.compile(r"\bayush|\bayurved|\bherbal|\btraditional\s+knowledge|\btkdl", re.I), "ayush"),
]


# ---------------------------------------------------------------------------
# Query classification helpers
# ---------------------------------------------------------------------------

def _is_exact_query(question: str) -> bool:
    return bool(_EXACT_QUERY_RE.search(question))


def _detect_domain(question: str) -> Optional[str]:
    for pattern, domain in _DOMAIN_HINTS:
        if pattern.search(question):
            return domain
    return None


def _extract_section(question: str) -> Optional[str]:
    """Try to extract an exact section label like '3(p)' or '3'."""
    m = re.search(
        r"(?:section|sec\.?)\s*(\d+[A-Z]?)\s*(?:\(([a-zA-Z0-9]+)\))?",
        question,
        re.IGNORECASE,
    )
    if m:
        sec = m.group(1)
        sub = m.group(2)
        return f"{sec}({sub})" if sub else sec
    return None


# ---------------------------------------------------------------------------
# Score normalisation & deduplication
# ---------------------------------------------------------------------------

def _normalise(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _deduplicate(
    candidates: list[tuple[LegalChunk, float]]
) -> list[tuple[LegalChunk, float]]:
    seen: set[str] = set()
    out = []
    for chunk, score in candidates:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            out.append((chunk, score))
    return out


# ---------------------------------------------------------------------------
# Cross-encoder reranker (lightweight — uses the embedding model's similarity)
# ---------------------------------------------------------------------------

class _SimpleReranker:
    """
    Score each chunk by computing cosine similarity between
    the query embedding and the chunk embedding.

    A proper cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)
    would be more accurate but requires sentence-transformers[cross-encoder].
    This is a drop-in that works with the same model we already load.
    """

    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self._model = embedding_model

    def rerank(
        self,
        query: str,
        candidates: list[LegalChunk],
        top_k: int = 5,
    ) -> list[tuple[LegalChunk, float]]:
        if not candidates:
            return []

        texts = [c.text for c in candidates]
        # Batch encode
        q_emb = self._model.embed_query(query)
        c_embs = self._model.embed_documents(texts)

        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb + 1e-9)

        scored = [(c, cosine(q_emb, emb)) for c, emb in zip(candidates, c_embs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Evidence sufficiency
# ---------------------------------------------------------------------------

def _confidence(top_score: float) -> str:
    if top_score >= 0.75:
        return "high"
    if top_score >= 0.50:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combine BM25 + vector search, rerank, check sufficiency.

    Parameters
    ----------
    embeddings      : optional shared HuggingFaceEmbeddings instance
    bm25_top_k      : candidates from BM25 per query
    vector_top_k    : candidates from vector search per query
    final_top_k     : chunks returned after reranking
    """

    def __init__(
        self,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        bm25_top_k: int = 10,
        vector_top_k: int = 10,
        final_top_k: int = 5,
    ):
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k

        if embeddings is not None:
            self._embeddings = embeddings
        else:
            print("Loading embedding model …")
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        print("Connecting to Qdrant …")
        from qdrant_singleton import get_qdrant_client
        self._qdrant_client = get_qdrant_client()
        self._vector_store = QdrantVectorStore(
            client=self._qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self._embeddings,
        )

        print("Loading BM25 index …")
        self._bm25 = BM25Store.load(BM25_PATH)

        self._reranker = _SimpleReranker(self._embeddings)
        print("Retriever ready.\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        domain_filter: Optional[str] = None,
    ) -> dict:
        """
        Returns
        -------
        {
            "chunks"     : list[LegalChunk],
            "confidence" : "high" | "medium" | "low",
            "sufficient" : bool,
            "query_type" : "exact" | "semantic",
        }
        """
        is_exact = _is_exact_query(question)
        auto_domain = _detect_domain(question)
        domain = domain_filter or auto_domain
        query_type = "exact" if is_exact else "semantic"

        # ── BM25 retrieval ────────────────────────────────────────────────
        bm25_chunks = self._bm25.query(question, top_k=self.bm25_top_k)

        # ── Vector retrieval ─────────────────────────────────────────────
        qdrant_filter = None
        if domain:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key="domain", match=MatchValue(value=domain))
                ]
            )

        vector_docs: list[Document] = self._vector_store.similarity_search(
            question,
            k=self.vector_top_k,
            filter=qdrant_filter,
        )
        vector_chunks = [self._doc_to_chunk(d) for d in vector_docs]

        # ── Merge & deduplicate ───────────────────────────────────────────
        # Weight exact queries slightly more toward BM25
        if is_exact:
            bm25_weight, vec_weight = 0.6, 0.4
        else:
            bm25_weight, vec_weight = 0.4, 0.6

        bm25_scores = _normalise([1.0 / (i + 1) for i in range(len(bm25_chunks))])
        vec_scores  = _normalise([1.0 / (i + 1) for i in range(len(vector_chunks))])

        candidates: list[tuple[LegalChunk, float]] = (
            [(c, s * bm25_weight) for c, s in zip(bm25_chunks, bm25_scores)]
            + [(c, s * vec_weight) for c, s in zip(vector_chunks, vec_scores)]
        )
        candidates = _deduplicate(candidates)

        # ── Rerank ────────────────────────────────────────────────────────
        raw_chunks = [c for c, _ in candidates]
        reranked = self._reranker.rerank(question, raw_chunks, top_k=self.final_top_k)

        if not reranked:
            return {
                "chunks": [],
                "confidence": "low",
                "sufficient": False,
                "query_type": query_type,
            }

        top_score = reranked[0][1]
        conf = _confidence(top_score)
        sufficient = top_score >= _SUFFICIENCY_THRESHOLD

        return {
            "chunks": [c for c, _ in reranked],
            "confidence": conf,
            "sufficient": sufficient,
            "query_type": query_type,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_to_chunk(doc: Document) -> LegalChunk:
        """Reconstruct a LegalChunk from a Qdrant-retrieved Document."""
        m = doc.metadata
        return LegalChunk(
            chunk_id=m.get("chunk_id", "unknown"),
            document_id=m.get("document_id", "unknown"),
            title=m.get("title", "Unknown Document"),
            source=m.get("source", "Unknown"),
            source_url=m.get("source_url"),
            document_type=m.get("document_type", "act"),
            domain=m.get("domain", "general"),
            authority_level=m.get("authority_level", "primary"),
            chapter=m.get("chapter"),
            section=m.get("section"),
            subsection=m.get("subsection"),
            page=m.get("page"),
            language=m.get("language", "english"),
            version=m.get("version"),
            effective_date=m.get("effective_date"),
            last_verified=m.get("last_verified", "2026-08-25"),
            text=doc.page_content,
        )
