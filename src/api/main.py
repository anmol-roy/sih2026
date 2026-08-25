"""
IP-SAKTI Sahayak — Phase 3 API
────────────────────────────────
Endpoints
─────────
GET  /                    health check
GET  /health              health check

POST /query               Phase 2: legal Q&A with citations
POST /analyze-invention   Phase 3: invention analysis + preliminary report

Run with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

# ── Make src/ importable ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.schema import Invention
from retrieval.retriever import HybridRetriever
from analysis.invention_extractor import InventionExtractor
from analysis.query_generator import generate_queries
from analysis.evidence_fusion import fuse_evidence
from patents.patent_search import PatentSearcher
from tk.tk_matcher import TKMatcher
from generation.report_generator import ReportGenerator

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IP-SAKTI Sahayak",
    description="Indian IP legal RAG + Invention analysis (Phase 3)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons  (created once on first request)
# ─────────────────────────────────────────────────────────────────────────────

_embeddings : Optional[HuggingFaceEmbeddings] = None
_llm        : Optional[ChatGroq]              = None
_retriever  : Optional[HybridRetriever]       = None
_extractor  : Optional[InventionExtractor]    = None
_pat_searcher: Optional[PatentSearcher]       = None
_tk_matcher : Optional[TKMatcher]             = None
_reporter   : Optional[ReportGenerator]       = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            embeddings   = _get_embeddings(),
            bm25_top_k   = 10,
            vector_top_k = 10,
            final_top_k  = 5,
        )
    return _retriever


def _get_extractor() -> InventionExtractor:
    global _extractor
    if _extractor is None:
        _extractor = InventionExtractor(llm=_get_llm())
    return _extractor


def _get_pat_searcher() -> PatentSearcher:
    global _pat_searcher
    if _pat_searcher is None:
        _pat_searcher = PatentSearcher(
            embeddings     = _get_embeddings(),
            top_k_retrieval= 10,
            top_k_final    = 5,
        )
    return _pat_searcher


def _get_tk_matcher() -> TKMatcher:
    global _tk_matcher
    if _tk_matcher is None:
        _tk_matcher = TKMatcher(embeddings=_get_embeddings(), top_k=5)
    return _tk_matcher


def _get_reporter() -> ReportGenerator:
    global _reporter
    if _reporter is None:
        _reporter = ReportGenerator(llm=_get_llm())
    return _reporter


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    domain  : Optional[str] = None


class CitationOut(BaseModel):
    document   : str
    chapter    : Optional[str] = None
    section    : Optional[str] = None
    subsection : Optional[str] = None
    page       : Optional[int] = None
    source     : str
    source_url : Optional[str] = None
    domain     : str
    chunk_id   : str


class QueryResponse(BaseModel):
    answer     : str
    confidence : str
    citations  : list[CitationOut]
    query_type : str
    sufficient : bool


class InventionRequest(BaseModel):
    description: str


class PatentMatchOut(BaseModel):
    label           : str
    similarity_score: Optional[float] = None
    source          : str
    chunk_id        : str


class TKMatchOut(BaseModel):
    label  : str
    source : str
    page   : Optional[int] = None
    score  : Optional[float] = None


class EvidenceCitationOut(BaseModel):
    type     : str
    label    : str
    source   : str
    section  : Optional[str] = None
    page     : Optional[int] = None
    chunk_id : str


class InventionOut(BaseModel):
    title          : str
    technical_field: Optional[str] = None
    problem        : Optional[str] = None
    solution       : Optional[str] = None
    components     : list[str]
    intended_use   : Optional[str] = None
    keywords       : list[str]
    novelty_claim  : Optional[str] = None


class AnalysisResponse(BaseModel):
    invention            : InventionOut
    relevant_provisions  : list[str]
    similar_patents      : list[PatentMatchOut]
    tk_matches           : list[TKMatchOut]
    issues               : list[str]
    assessment           : str
    confidence           : str
    citations            : list[EvidenceCitationOut]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 query handler (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_context_block(chunk) -> str:
    parts = [f"[{chunk.title}"]
    if chunk.chapter:
        parts.append(f"  {chunk.chapter}")
    if chunk.section:
        parts.append(f"  {chunk.section}")
    if chunk.subsection:
        parts.append(f"  § {chunk.subsection}")
    if chunk.page:
        parts.append(f"  p. {chunk.page}")
    parts.append(f"  Source: {chunk.source}]")
    return "\n".join(parts) + f"\n\n{chunk.text}"


_LEGAL_SYSTEM = """\
You are an Indian intellectual-property legal information assistant.
Answer ONLY from the provided document excerpts.
If the answer is not present, reply exactly:
"I could not find sufficient information in the provided authoritative documents."
Never invent legal provisions, sections, or case-law.
Cite every legal provision you reference (Act name, Section, Subsection).
Be concise and precise."""


def _handle_query(question: str, domain: Optional[str] = None) -> QueryResponse:
    result    = _get_retriever().retrieve(question, domain_filter=domain)
    chunks    = result["chunks"]
    confidence= result["confidence"]
    sufficient= result["sufficient"]
    query_type= result["query_type"]

    if not sufficient or not chunks:
        return QueryResponse(
            answer=(
                "I could not find sufficient information in the "
                "provided authoritative documents."
            ),
            confidence="low",
            citations=[],
            query_type=query_type,
            sufficient=False,
        )

    context = "\n\n---\n\n".join(_chunk_to_context_block(c) for c in chunks)
    messages = [
        {"role": "system", "content": _LEGAL_SYSTEM},
        {"role": "user",   "content": f"Documents:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"},
    ]
    answer = _get_llm().invoke(messages).content.strip()

    citations = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        c = chunk.citation()
        citations.append(CitationOut(
            document   = c["document"],
            chapter    = c["chapter"]    or None,
            section    = c["section"]    or None,
            subsection = c["subsection"] or None,
            page       = c["page"],
            source     = c["source"],
            source_url = c["source_url"] or None,
            domain     = c["domain"],
            chunk_id   = c["chunk_id"],
        ))

    return QueryResponse(
        answer=answer, confidence=confidence,
        citations=citations, query_type=query_type, sufficient=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 invention analysis handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_analysis(description: str) -> AnalysisResponse:
    # 3.1 — Extract structured invention
    invention: Invention = _get_extractor().extract(description)

    # 3.2 — Generate search queries
    queries = generate_queries(invention)

    # 3.3 — Legal retrieval (multi-query, patent domain priority)
    legal_retriever = _get_retriever()
    all_legal_chunks: dict[str, tuple] = {}   # chunk_id → (chunk, score)

    for q in queries[:6]:
        result = legal_retriever.retrieve(q, domain_filter="patent")
        for chunk in result["chunks"]:
            cid = chunk.chunk_id
            if cid not in all_legal_chunks:
                # Use score proxy: position rank
                all_legal_chunks[cid] = (chunk, result.get("confidence_score", 0.5))

    # Also run without domain filter to capture cross-domain law
    for q in queries[:3]:
        result = legal_retriever.retrieve(q)
        for chunk in result["chunks"]:
            cid = chunk.chunk_id
            if cid not in all_legal_chunks:
                all_legal_chunks[cid] = (chunk, 0.4)

    legal_chunks = list(all_legal_chunks.values())   # list of (chunk, score)

    # 3.4 — Patent search
    patent_results = _get_pat_searcher().search(invention, queries)

    # 3.5 — TK / AYUSH match
    tk_result = _get_tk_matcher().match(invention)
    tk_chunks = [
        (m["chunk"], m["score"])
        for m in tk_result["matches"]
    ]

    # 3.6 — Fuse evidence
    evidence_package = fuse_evidence(
        invention     = invention,
        legal_chunks  = legal_chunks,
        patent_results= patent_results,
        tk_chunks     = tk_chunks,
    )

    # 3.7 — Generate report
    report = _get_reporter().generate(evidence_package)

    # ── Build response ────────────────────────────────────────────────────
    return AnalysisResponse(
        invention=InventionOut(
            title          = invention.title,
            technical_field= invention.technical_field,
            problem        = invention.problem,
            solution       = invention.solution,
            components     = invention.components,
            intended_use   = invention.intended_use,
            keywords       = invention.keywords,
            novelty_claim  = invention.novelty_claim,
        ),
        relevant_provisions = report["relevant_provisions"],
        similar_patents = [
            PatentMatchOut(
                label            = p["label"],
                similarity_score = p.get("similarity_score"),
                source           = p["source"],
                chunk_id         = p["chunk_id"],
            )
            for p in report["similar_patents"]
        ],
        tk_matches = [
            TKMatchOut(
                label  = t["label"],
                source = t["source"],
                page   = t.get("page"),
                score  = t.get("score"),
            )
            for t in report["tk_matches"]
        ],
        issues     = report["issues"],
        assessment = report["assessment"],
        confidence = report["confidence"],
        citations  = [
            EvidenceCitationOut(
                type    = c["type"],
                label   = c["label"],
                source  = c["source"],
                section = c.get("section"),
                page    = c.get("page"),
                chunk_id= c["chunk_id"],
            )
            for c in report["citations"]
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "IP-SAKTI Sahayak", "version": "3.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["legal-qa"])
def query(req: QueryRequest):
    """Phase 2 endpoint: legal Q&A with section-level citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return _handle_query(req.question.strip(), req.domain)


@app.post("/analyze-invention", response_model=AnalysisResponse, tags=["invention"])
def analyze_invention(req: InventionRequest):
    """
    Phase 3 endpoint: given a free-text invention description, return a
    preliminary IP assessment with legal provisions, similar patents,
    TK matches, and an evidence-backed report.
    """
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")
    return _handle_analysis(req.description.strip())
