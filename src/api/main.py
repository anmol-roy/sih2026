"""
IP-SAKTI Sahayak — Phase 4 API
────────────────────────────────
Endpoints
─────────
GET  /                        health check
GET  /health                  health check

POST /query                   Phase 2 — legal Q&A with section citations
POST /analyze-invention       Phase 3 — invention + legal/patent/TK report
POST /patentability-check     Phase 4 — feature-level prior-art + patentability report

Run with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ingestion.schema import Invention
from retrieval.retriever import HybridRetriever
from analysis.invention_extractor import InventionExtractor
from analysis.feature_extractor import FeatureExtractor
from analysis.claim_representation import ClaimBuilder
from analysis.query_generator import generate_queries
from analysis.evidence_fusion import fuse_evidence, EvidenceItem
from analysis.novelty import NoveltyAnalyzer
from analysis.inventive_step import InventiveStepAnalyzer
from patents.patent_search import PatentSearcher
from patents.prior_art_search import PriorArtSearcher, generate_prior_art_queries
from patents.patent_matcher import PatentMatcher
from tk.tk_matcher import TKMatcher
from generation.report_generator import ReportGenerator

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IP-SAKTI Sahayak",
    description="Indian IP legal RAG + Invention analysis + Patentability check",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────────────────────────────────────

_embeddings    : Optional[HuggingFaceEmbeddings] = None
_llm           : Optional[ChatGroq]              = None
_retriever     : Optional[HybridRetriever]       = None
_extractor     : Optional[InventionExtractor]    = None
_feat_extractor: Optional[FeatureExtractor]      = None
_claim_builder : Optional[ClaimBuilder]          = None
_pat_searcher  : Optional[PatentSearcher]        = None
_pa_searcher   : Optional[PriorArtSearcher]      = None
_matcher       : Optional[PatentMatcher]         = None
_tk_matcher    : Optional[TKMatcher]             = None
_reporter      : Optional[ReportGenerator]       = None
_novelty_az    : Optional[NoveltyAnalyzer]       = None
_invstep_az    : Optional[InventiveStepAnalyzer] = None


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
            embeddings=_get_embeddings(),
            bm25_top_k=10, vector_top_k=10, final_top_k=5,
        )
    return _retriever


def _get_extractor() -> InventionExtractor:
    global _extractor
    if _extractor is None:
        _extractor = InventionExtractor(llm=_get_llm())
    return _extractor


def _get_feat_extractor() -> FeatureExtractor:
    global _feat_extractor
    if _feat_extractor is None:
        _feat_extractor = FeatureExtractor(llm=_get_llm())
    return _feat_extractor


def _get_claim_builder() -> ClaimBuilder:
    global _claim_builder
    if _claim_builder is None:
        _claim_builder = ClaimBuilder(llm=_get_llm())
    return _claim_builder


def _get_pat_searcher() -> PatentSearcher:
    global _pat_searcher
    if _pat_searcher is None:
        _pat_searcher = PatentSearcher(
            embeddings=_get_embeddings(),
            top_k_retrieval=10, top_k_final=5,
        )
    return _pat_searcher


def _get_pa_searcher(cutoff_date: Optional[str] = None) -> PriorArtSearcher:
    global _pa_searcher
    # Always recreate if cutoff changes; otherwise reuse
    if _pa_searcher is None or _pa_searcher._cutoff != cutoff_date:
        _pa_searcher = PriorArtSearcher(
            embeddings=_get_embeddings(),
            top_k_per_query=10,
            final_pool_size=20,
            cutoff_date=cutoff_date,
        )
    return _pa_searcher


def _get_matcher() -> PatentMatcher:
    global _matcher
    if _matcher is None:
        _matcher = PatentMatcher(embeddings=_get_embeddings())
    return _matcher


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


def _get_novelty_az() -> NoveltyAnalyzer:
    global _novelty_az
    if _novelty_az is None:
        _novelty_az = NoveltyAnalyzer()
    return _novelty_az


def _get_invstep_az() -> InventiveStepAnalyzer:
    global _invstep_az
    if _invstep_az is None:
        _invstep_az = InventiveStepAnalyzer()
    return _invstep_az


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic I/O models
# ─────────────────────────────────────────────────────────────────────────────

# ── Phase 2 ──────────────────────────────────────────────────────────────────

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


# ── Phase 3 ──────────────────────────────────────────────────────────────────

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
    invention           : InventionOut
    relevant_provisions : list[str]
    similar_patents     : list[PatentMatchOut]
    tk_matches          : list[TKMatchOut]
    issues              : list[str]
    assessment          : str
    confidence          : str
    citations           : list[EvidenceCitationOut]


# ── Phase 4 ──────────────────────────────────────────────────────────────────

class PatentabilityRequest(BaseModel):
    description : str
    cutoff_date : Optional[str] = None   # ISO "YYYY-MM-DD" — only prior art before this date


class FeatureOut(BaseModel):
    id      : str
    feature : str
    category: str


class InventionDetailOut(BaseModel):
    title            : str
    technical_field  : Optional[str] = None
    features         : list[FeatureOut]
    claim_type       : str
    ipc_suggested    : list[str]
    independent_claim: str


class LegalBasisOut(BaseModel):
    label  : str
    source : str
    section: Optional[str] = None
    page   : Optional[int] = None


class TKEvidenceOut(BaseModel):
    label  : str
    source : str
    page   : Optional[int] = None
    score  : Optional[float] = None
    section: Optional[str] = None


class AssessmentStatusOut(BaseModel):
    status: str
    reason: str


class PatentabilityResponse(BaseModel):
    invention            : InventionDetailOut
    legal_basis          : list[LegalBasisOut]
    prior_art            : list[dict]            # PriorArtCandidate summaries
    traditional_knowledge: list[TKEvidenceOut]
    novelty_analysis     : dict
    inventive_step_analysis: dict
    assessment           : AssessmentStatusOut
    report               : str
    confidence           : str
    citations            : list[EvidenceCitationOut]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 handler
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_context_block(chunk) -> str:
    parts = [f"[{chunk.title}"]
    for attr, prefix in [("chapter", ""), ("section", ""), ("subsection", "§ "), ("page", "p. ")]:
        val = getattr(chunk, attr, None)
        if val:
            parts.append(f"  {prefix}{val}")
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
    result     = _get_retriever().retrieve(question, domain_filter=domain)
    chunks     = result["chunks"]
    confidence = result["confidence"]
    sufficient = result["sufficient"]
    query_type = result["query_type"]

    if not sufficient or not chunks:
        return QueryResponse(
            answer="I could not find sufficient information in the provided authoritative documents.",
            confidence="low", citations=[], query_type=query_type, sufficient=False,
        )

    context  = "\n\n---\n\n".join(_chunk_to_context_block(c) for c in chunks)
    messages = [
        {"role": "system", "content": _LEGAL_SYSTEM},
        {"role": "user",   "content": f"Documents:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"},
    ]
    answer = _get_llm().invoke(messages).content.strip()

    citations, seen = [], set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        c = chunk.citation()
        citations.append(CitationOut(
            document=c["document"], chapter=c["chapter"] or None,
            section=c["section"] or None, subsection=c["subsection"] or None,
            page=c["page"], source=c["source"],
            source_url=c["source_url"] or None, domain=c["domain"],
            chunk_id=c["chunk_id"],
        ))
    return QueryResponse(
        answer=answer, confidence=confidence,
        citations=citations, query_type=query_type, sufficient=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 handler
# ─────────────────────────────────────────────────────────────────────────────

def _collect_legal_chunks(queries: list[str]) -> list[tuple]:
    legal_map: dict[str, tuple] = {}
    retriever = _get_retriever()
    for q in queries[:6]:
        for chunk in retriever.retrieve(q, domain_filter="patent")["chunks"]:
            if chunk.chunk_id not in legal_map:
                legal_map[chunk.chunk_id] = (chunk, 0.5)
    for q in queries[:3]:
        for chunk in retriever.retrieve(q)["chunks"]:
            if chunk.chunk_id not in legal_map:
                legal_map[chunk.chunk_id] = (chunk, 0.4)
    return list(legal_map.values())


def _handle_analysis(description: str) -> AnalysisResponse:
    invention = _get_extractor().extract(description)
    queries   = generate_queries(invention)

    legal_chunks   = _collect_legal_chunks(queries)
    patent_results = _get_pat_searcher().search(invention, queries)
    tk_result      = _get_tk_matcher().match(invention)
    tk_chunks      = [(m["chunk"], m["score"]) for m in tk_result["matches"]]

    evidence_pkg = fuse_evidence(
        invention=invention, legal_chunks=legal_chunks,
        patent_results=patent_results, tk_chunks=tk_chunks,
    )
    report = _get_reporter().generate(evidence_pkg)

    return AnalysisResponse(
        invention=InventionOut(
            title=invention.title, technical_field=invention.technical_field,
            problem=invention.problem, solution=invention.solution,
            components=invention.components, intended_use=invention.intended_use,
            keywords=invention.keywords, novelty_claim=invention.novelty_claim,
        ),
        relevant_provisions=report["relevant_provisions"],
        similar_patents=[
            PatentMatchOut(label=p["label"], similarity_score=p.get("similarity_score"),
                           source=p["source"], chunk_id=p["chunk_id"])
            for p in report["similar_patents"]
        ],
        tk_matches=[
            TKMatchOut(label=t["label"], source=t["source"],
                       page=t.get("page"), score=t.get("score"))
            for t in report["tk_matches"]
        ],
        issues=report["issues"], assessment=report["assessment"],
        confidence=report["confidence"],
        citations=[
            EvidenceCitationOut(type=c["type"], label=c["label"], source=c["source"],
                                section=c.get("section"), page=c.get("page"),
                                chunk_id=c["chunk_id"])
            for c in report["citations"]
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_patentability(
    description: str,
    cutoff_date: Optional[str],
) -> PatentabilityResponse:

    # 4.1 — Extract structured invention
    invention = _get_extractor().extract(description)

    # 4.2 — Extract discrete features
    features = _get_feat_extractor().extract_from_invention(invention)

    # 4.3 — Build claim representation + suggested IPC codes
    claim = _get_claim_builder().build(invention, features)

    # 4.4 — Generate combinatorial prior-art queries
    pa_queries = generate_prior_art_queries(features, claim, invention, max_queries=20)

    # 4.5 — Build candidate pool (date-filtered)
    pa_searcher = _get_pa_searcher(cutoff_date)
    candidate_pool = pa_searcher.build_candidate_pool(pa_queries, invention, claim)

    # 4.6 — Feature-to-patent matching matrix
    pa_candidates = _get_matcher().match_candidates(features, candidate_pool)

    # 4.7 — Novelty analysis
    novelty_result = _get_novelty_az().analyze(features, pa_candidates)

    # 4.8 — Inventive-step analysis
    invstep_result = _get_invstep_az().analyze(features, pa_candidates, novelty_result)

    # 4.9 — Legal retrieval (multi-query)
    legal_chunks = _collect_legal_chunks(pa_queries[:6])

    # 4.10 — TK / AYUSH matching
    tk_result  = _get_tk_matcher().match(invention)
    tk_chunks  = [(m["chunk"], m["score"]) for m in tk_result["matches"]]

    # 4.11 — Convert legal/TK chunks into EvidenceItem lists for the report
    from analysis.evidence_fusion import (
        _legal_chunk_to_evidence, _tk_chunk_to_evidence
    )
    legal_evidence: list[EvidenceItem] = [
        _legal_chunk_to_evidence(c, s) for c, s in legal_chunks
    ]
    tk_evidence: list[EvidenceItem] = [
        _tk_chunk_to_evidence(c, s) for c, s in tk_chunks
    ]

    # 4.12 — Compute overall confidence from candidate coverage
    top_coverage = pa_candidates[0].coverage() if pa_candidates else 0.0
    if top_coverage >= 0.75:
        overall_confidence = "high"
    elif top_coverage >= 0.45:
        overall_confidence = "medium"
    else:
        overall_confidence = "low"

    # 4.13 — Build report package
    pat4_package = {
        "invention"            : invention,
        "features"             : features,
        "claim"                : claim,
        "novelty_result"       : novelty_result,
        "inventive_step_result": invstep_result,
        "legal_evidence"       : legal_evidence,
        "tk_evidence"          : tk_evidence,
        "prior_art_summaries"  : [c.to_summary() for c in pa_candidates[:10]],
        "overall_confidence"   : overall_confidence,
    }

    # 4.14 — Generate patentability report
    report = _get_reporter().generate_patentability(pat4_package)

    # ── Build response ─────────────────────────────────────────────────────
    return PatentabilityResponse(
        invention=InventionDetailOut(
            title            = report["invention"]["title"],
            technical_field  = report["invention"].get("technical_field"),
            features         = [
                FeatureOut(id=f["id"], feature=f["feature"], category=f["category"])
                for f in report["invention"]["features"]
            ],
            claim_type       = report["invention"]["claim_type"],
            ipc_suggested    = report["invention"]["ipc_suggested"],
            independent_claim= report["invention"]["independent_claim"],
        ),
        legal_basis=[
            LegalBasisOut(
                label=lb["label"], source=lb["source"],
                section=lb.get("section"), page=lb.get("page"),
            )
            for lb in report["legal_basis"]
        ],
        prior_art=report["prior_art"],
        traditional_knowledge=[
            TKEvidenceOut(
                label=t["label"], source=t["source"], page=t.get("page"),
                score=t.get("score"), section=t.get("section"),
            )
            for t in report["traditional_knowledge"]
        ],
        novelty_analysis        = report["novelty_analysis"],
        inventive_step_analysis = report["inventive_step_analysis"],
        assessment=AssessmentStatusOut(
            status=report["assessment"]["status"],
            reason=report["assessment"]["reason"],
        ),
        report    = report["report"],
        confidence= report["confidence"],
        citations=[
            EvidenceCitationOut(
                type=c["type"], label=c["label"], source=c["source"],
                section=c.get("section"), page=c.get("page"),
                chunk_id=c["chunk_id"],
            )
            for c in report["citations"]
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "IP-SAKTI Sahayak", "version": "4.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["phase-2"])
def query(req: QueryRequest):
    """Phase 2 — Legal Q&A with section-level citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return _handle_query(req.question.strip(), req.domain)


@app.post("/analyze-invention", response_model=AnalysisResponse, tags=["phase-3"])
def analyze_invention(req: InventionRequest):
    """
    Phase 3 — Preliminary IP assessment.
    Returns relevant legal provisions, similar patents, TK matches,
    and an evidence-backed report.
    """
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")
    return _handle_analysis(req.description.strip())


@app.post("/patentability-check", response_model=PatentabilityResponse, tags=["phase-4"])
def patentability_check(req: PatentabilityRequest):
    """
    Phase 4 — Feature-level patentability analysis.

    Steps:
      1. Extract discrete technical features (F1, F2, …)
      2. Build a claim-like representation + IPC codes
      3. Generate combinatorial prior-art search queries
      4. Build a date-filtered candidate pool (BM25 + vector)
      5. Match each feature against each candidate (exact / semantic / no-match)
      6. Novelty analysis  — single-reference full coverage?
      7. Inventive-step analysis  — multi-reference combination coverage?
      8. Legal retrieval  — relevant Patents Act provisions
      9. TK / AYUSH match
     10. Evidence-backed preliminary patentability report

    Supply `cutoff_date` (ISO "YYYY-MM-DD") to restrict prior art to
    documents filed/published before that date.
    """
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")
    return _handle_patentability(req.description.strip(), req.cutoff_date)
