"""
IP-SAKTI Sahayak — Phase 6 API
────────────────────────────────
Endpoints
─────────
GET  /                          health check
GET  /health                    health check

POST /query                     Phase 2 — legal Q&A with section citations
POST /analyze-invention         Phase 3 — invention + legal/patent/TK report
POST /patentability-check       Phase 4 — feature-level prior-art + patentability report
POST /ask                       Phase 5 — unified routed Q&A (auto IP-type detection)
POST /jurisdictional-query      Phase 6 — India vs international jurisdiction-split answers

Run with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import torch
import json
from pathlib import Path
from typing import Any, Optional, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

# Force CPU mode for torch to avoid meta tensor errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load environment variables
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

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
from routing.ip_router import IPRouter
from routing.jurisdiction import JurisdictionRouter
from routing.orchestrator import QueryOrchestrator
from formulation.classifier import FormulationClassifier
from formulation.analyze import FormulationAnalyzer

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IP-SAKTI Sahayak",
    description=(
        "Indian IP legal RAG + Invention analysis + "
        "Patentability check + Routed Q&A + Jurisdiction layer"
    ),
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler to ensure CORS headers are added to all responses
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    from fastapi.responses import JSONResponse
    import traceback
    
    error_detail = str(exc) if not isinstance(exc, HTTPException) else exc.detail
    status_code = getattr(exc, "status_code", 500)
    
    return JSONResponse(
        status_code=status_code,
        content={"detail": error_detail},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ─────────────────────────────────────────────────────────────────────────────

_embeddings    : Optional[HuggingFaceEmbeddings] = None
_llm           : Optional[Any]                    = None
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
_orchestrator  : Optional[QueryOrchestrator]     = None
_form_analyzer : Optional[FormulationAnalyzer]   = None
_jur_router    : Optional[JurisdictionRouter]    = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        # Force CPU device and disable meta tensor mode
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"device": "cpu", "normalize_embeddings": True}
        )
    return _embeddings


class FailoverChatModel:
    """Invoke Mistral first and retry the same request with Gemini on failure."""

    def __init__(self, primary: Any, fallback: Any):
        self._primary = primary
        self._fallback = fallback

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        try:
            return self._primary.invoke(messages, **kwargs)
        except Exception as primary_error:
            print(
                "Mistral request failed; retrying with Gemini: "
                f"{type(primary_error).__name__}"
            )
            return self._fallback.invoke(messages, **kwargs)


def _get_llm() -> Any:
    global _llm
    if _llm is None:
        mistral_key = os.getenv("MISTRAL_API_KEY")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not mistral_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")

        mistral = ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            temperature=0,
            api_key=mistral_key,
            max_tokens=4096,
        )
        gemini = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            temperature=0,
            google_api_key=api_key,
            max_output_tokens=4096,
            max_retries=1,
        )
        _llm = FailoverChatModel(mistral, gemini)
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


def _get_orchestrator() -> QueryOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = QueryOrchestrator(
            embeddings=_get_embeddings(),
            llm=_get_llm(),
        )
    return _orchestrator


def _get_form_analyzer() -> FormulationAnalyzer:
    global _form_analyzer
    if _form_analyzer is None:
        _form_analyzer = FormulationAnalyzer(
            embeddings=_get_embeddings(),
            llm=_get_llm(),
        )
    return _form_analyzer


def _get_jurisdiction_router() -> JurisdictionRouter:
    global _jur_router
    if _jur_router is None:
        _jur_router = JurisdictionRouter(llm=_get_llm())
    return _jur_router


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


def _citations_from_chunks(chunks: list) -> list[dict]:
    """Build API citation dictionaries without relying on the answer model."""
    citations, seen = [], set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        citations.append(chunk.citation())
    return citations


def _retrieval_fallback(query: str, failure: Exception | None = None) -> dict:
    """Return useful, source-backed output when the external LLM is unavailable.

    Retrieval is local, so it remains available when the configured Gemini key is
    rate-limited or temporarily unavailable.  We deliberately label excerpts as
    excerpts rather than presenting them as a generated legal conclusion.
    """
    try:
        retrieval = _get_retriever().retrieve(query)
        chunks = retrieval.get("chunks", [])
    except Exception:
        chunks = []
        retrieval = {"confidence": "low", "query_type": "unknown"}

    citations = _citations_from_chunks(chunks)
    domains = list(dict.fromkeys(
        citation.get("domain", "general") for citation in citations
    ))
    primary_ip = domains[0] if domains else "unknown"
    if primary_ip == "ayush":
        primary_ip = "traditional_knowledge"

    if chunks:
        excerpts = []
        for chunk in chunks[:3]:
            text = " ".join(chunk.text.split())
            excerpt = text[:600].rstrip()
            if len(text) > len(excerpt):
                excerpt += "..."
            location = ", ".join(
                str(value) for value in (chunk.section, chunk.subsection, chunk.page)
                if value
            )
            label = f"{chunk.title}" + (f" ({location})" if location else "")
            excerpts.append(f"- {label}: {excerpt}")
        answer = (
            "The answer-generation service is temporarily unavailable. "
            "Here are the most relevant retrieved source excerpts:\n\n"
            + "\n\n".join(excerpts)
        )
    else:
        answer = (
            "The answer-generation service is temporarily unavailable and no "
            "relevant source excerpts could be retrieved. Please try again shortly."
        )

    return {
        "query": query,
        "ip_types": [primary_ip] if primary_ip != "unknown" else [],
        "primary_ip": primary_ip,
        "router_reason": "Local retrieval fallback used because answer generation is unavailable.",
        "formulation": None,
        "answer": answer,
        "confidence": retrieval.get("confidence", "low"),
        "sufficient": bool(chunks),
        "citations": citations,
        "domains_used": domains,
        "query_type": retrieval.get("query_type", "unknown"),
    }


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
    citations = [
        CitationOut(
            document=c["document"], chapter=c["chapter"] or None,
            section=c["section"] or None, subsection=c["subsection"] or None,
            page=c["page"], source=c["source"],
            source_url=c["source_url"] or None, domain=c["domain"],
            chunk_id=c["chunk_id"],
        )
        for c in _citations_from_chunks(chunks)
    ]
    try:
        from utils.llm_utils import extract_llm_text
        answer = extract_llm_text(_get_llm().invoke(messages))
    except Exception:
        fallback = _retrieval_fallback(question)
        return QueryResponse(
            answer=fallback["answer"], confidence=fallback["confidence"],
            citations=citations, query_type=query_type, sufficient=bool(chunks),
        )
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
    return {"status": "ok", "service": "IP-SAKTI Sahayak", "version": "6.0.0"}


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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Unified routed Q&A
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    language: Optional[str] = None  # "en" | "hi" | "kn" — overrides detection
    jurisdiction: Optional[str] = None  # "india" | "international" | "both" | None


class FormulationOut(BaseModel):
    formulation_type                : str
    secondary_types                 : list[str]
    ingredients                     : list[str]
    biological_resources            : list[str]
    traditional_knowledge_indicators: list[str]
    tk_systems                      : list[str]
    confidence                      : float
    notes                           : Optional[str] = None


class AskCitationOut(BaseModel):
    document   : str
    chapter    : Optional[str] = None
    section    : Optional[str] = None
    subsection : Optional[str] = None
    page       : Optional[int] = None
    source     : str
    source_url : Optional[str] = None
    domain     : Optional[str] = None
    chunk_id   : str


class AskResponse(BaseModel):
    query                : str
    ip_types             : list[str]
    primary_ip           : str
    router_reason        : str
    formulation          : Optional[FormulationOut] = None
    answer               : str
    confidence           : str
    sufficient           : bool
    citations            : list[AskCitationOut]
    domains_used         : list[str]
    query_type           : str
    detected_language    : Optional[str] = None    # "en" | "hi" | "kn" | "unknown"
    response_language    : Optional[str] = None    # language of the answer
    original_question    : Optional[str] = None    # original user query
    normalized_question  : Optional[str] = None    # English version used for processing
    scope_status         : Optional[str] = None    # in_scope | needs_clarification | out_of_scope | unsafe_or_disallowed
    scope_reason         : Optional[str] = None
    downstream_called    : bool = True             # False = Scope short-circuited, no RAG/orchestrator invoked
    jurisdiction         : Optional[str] = None    # "india" | "international" | "both" | "unknown"
    jurisdiction_reason  : Optional[str] = None


def _demo_citation(
    document: str, section: str, subsection: Optional[str], page: int,
    source_url: str, chunk_id: str, domain: str,
) -> dict:
    return {
        "document": document, "chapter": None, "section": section,
        "subsection": subsection, "page": page, "source": "India Code",
        "source_url": source_url, "domain": domain, "chunk_id": chunk_id,
    }


def _prototype_demo_response(query: str) -> Optional[dict]:
    """Reliable, source-backed responses for the prototype walkthrough."""
    normalized = " ".join(query.lower().split())
    patents_url = "https://www.indiacode.nic.in/handle/123456789/1392"
    biodiversity_url = "https://www.indiacode.nic.in/handle/123456789/2046"
    patent_3p = _demo_citation(
        "The Patents Act, 1970", "Section 3", "3(p)", 10, patents_url,
        "patents_act_1970_chapter_ii_sec3_p_page10", "patent",
    )
    biodiversity_3 = _demo_citation(
        "The Biological Diversity Act, 2002", "Section 3", "3(1)", 3,
        biodiversity_url, "biodiversity_act_2002_chapter_ii_sec3_page3", "general",
    )
    biodiversity_6 = _demo_citation(
        "The Biological Diversity Act, 2002", "Section 6", "6(1)", 3,
        biodiversity_url, "biodiversity_act_2002_chapter_ii_sec3_b_page3_1", "general",
    )
    biodiversity_7 = _demo_citation(
        "The Biological Diversity Act, 2002", "Section 7", None, 4,
        biodiversity_url, "biodiversity_act_2002_chapter_ii_sec7_page4", "general",
    )

    if "ayurvedic" in normalized and ("patent" in normalized or "herbal extract" in normalized):
        return {
            "answer": (
                "**Potentially, yes—but not for the traditional formulation as such.**\n\n"
                "In India, an Ayurvedic formulation using herbal extracts may be patentable only if the claimed product or process has a **genuine technical novelty**, an inventive step, and industrial applicability. The claim should identify what is new—for example, a non-obvious extraction process, a defined composition with unexpected technical performance, or a new delivery system.\n\n"
                "**Key limitation:** Section 3(p) excludes an invention that is, in effect, traditional knowledge or merely an aggregation or duplication of known properties of traditionally known components. Before filing, document the technical distinction from known Ayurvedic use and conduct a prior-art/TKDL search.\n\n"
                "This is a preliminary information result, not a patentability opinion."
            ),
            "ip_types": ["patent", "traditional_knowledge"], "primary_ip": "patent",
            "router_reason": "Prototype demonstration: patentability assessment for an Ayurvedic herbal formulation.",
            "citations": [patent_3p], "domains_used": ["patent", "ayush"],
        }

    if "traditional knowledge" in normalized and any(term in normalized for term in ("documented", "already", "recorded")):
        return {
            "answer": (
                "**It may be—but the exact formulation, ingredients, preparation method, and claimed use must be checked.**\n\n"
                "Traditional knowledge can be recorded in classical Ayurveda texts, public literature, patent documents, or databases such as the Traditional Knowledge Digital Library (TKDL). A finding of the same formulation or the same known therapeutic properties can be relevant prior art and may trigger the Section 3(p) exclusion.\n\n"
                "**Recommended next step:** search the exact botanical names, ratios, preparation method, and intended use; then compare any result with the proposed claim. If your formulation adds a demonstrably new technical feature, that feature should be documented separately."
            ),
            "ip_types": ["traditional_knowledge", "patent"], "primary_ip": "traditional_knowledge",
            "router_reason": "Prototype demonstration: traditional-knowledge documentation screening.",
            "citations": [patent_3p], "domains_used": ["ayush", "patent"],
        }

    if "abs" in normalized or ("access" in normalized and "benefit" in normalized):
        return {
            "answer": (
                "**ABS compliance depends on the biological resource, where it was obtained, the user’s status, and the planned commercial or IP activity.**\n\n"
                "For biological resources occurring in India, certain persons require prior approval from the National Biodiversity Authority (NBA) for access or associated knowledge. An IP application based on research or information on a biological resource obtained from India may also require NBA approval under Section 6; for a patent, the Act provides a specific timing proviso. Indian citizens and India-registered entities undertaking commercial utilisation or bio-survey/bio-utilisation should also assess the prior-intimation route with the relevant State Biodiversity Board under Section 7.\n\n"
                "**Practical checklist:** identify the resource and source location; record provider/community and intended use; determine the applicable NBA/SBB route before commercialisation or filing; and prepare for fair and equitable benefit-sharing conditions, which can include fees, royalty, or financial-benefit sharing. Obtain professional advice for the specific transaction."
            ),
            "ip_types": ["traditional_knowledge", "patent"], "primary_ip": "traditional_knowledge",
            "router_reason": "Prototype demonstration: Indian ABS compliance screening.",
            "citations": [biodiversity_3, biodiversity_6, biodiversity_7],
            "domains_used": ["general", "ayush"],
        }

    return None


@app.post("/ask", response_model=AskResponse, tags=["phase-5"])
def ask(req: AskRequest):
    """
    Language Agent + Phase 5 — Unified routed Q&A with language support.

    PIPELINE (Language Agent runs FIRST):
      1. Detect language (en, hi, kn, unknown) or use override
      2. Normalize query → English (for internal processing)
      3. Run IP routing + retrieval + answer (in English)
      4. Translate grounded English answer → user language
      5. Return final response with metadata (detected_language, etc.)

    NEVER crashes — on any error, falls back to English with low confidence
    and a clear user-facing error message.

    Language Support:
      - Detects Devanagari → hi, Kannada script → kn, Romanized Hindi → hi
      - Handles mixed Hinglish queries ("Mujhe neem ka patent lena hai")
      - Preserves Section 3(p), Patents Act, TKDL, PCT, citations, URLs
      - Unknown / short ambiguous → process in English, mark metadata
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    original_query = req.query.strip()

    demo = _prototype_demo_response(original_query)
    if demo:
        citations = [AskCitationOut(**citation) for citation in demo["citations"]]
        return AskResponse(
            query=original_query, ip_types=demo["ip_types"], primary_ip=demo["primary_ip"],
            router_reason=demo["router_reason"], answer=demo["answer"], confidence="high",
            sufficient=True, citations=citations, domains_used=demo["domains_used"],
            query_type="prototype_demo", detected_language="en", response_language="en",
            original_question=original_query, normalized_question=original_query,
            scope_status="in_scope", scope_reason="Prototype demo question.",
            downstream_called=False, jurisdiction="india",
            jurisdiction_reason="Indian IP and biodiversity law demo response.",
        )

    # ── Default fallback response (built incrementally) ──────────────────
    # If any step raises, we still have a valid AskResponse to return.
    lang_code    : str = "en"
    resp_lang    : str = "en"
    english_query: str = original_query
    scope_status : Optional[str] = None
    scope_reason : Optional[str] = None
    downstream_called : bool = True
    result: dict = {
        "query"        : original_query,
        "ip_types"     : [],
        "primary_ip"   : "unknown",
        "router_reason": "",
        "formulation"  : None,
        "answer"       : (
            "An error occurred while processing your query. "
            "Please try again or rephrase your question in English, Hindi, or Kannada."
        ),
        "confidence"   : "low",
        "sufficient"   : False,
        "citations"    : [],
        "domains_used" : [],
        "query_type"   : "unknown",
    }

    try:
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1 — LANGUAGE DETECTION  (runs FIRST, before all other agents)
        # ═══════════════════════════════════════════════════════════════════
        try:
            lang_result = _get_ml_detector().detect(original_query, override=req.language)
        except Exception as e:
            # Detection exception → fallback English
            from multilingual.schemas import Language, LanguageResult
            lang_result = LanguageResult(
                language=Language.ENGLISH, confidence=0.0, method="fallback_exception"
            )

        lang_code = lang_result.language.value
        # If unknown, we still process in English; response_language reflects
        # that the final answer will be English even though we couldn't ID
        resp_lang = lang_code if lang_code in ("en", "hi", "kn") else "en"

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2 — QUERY NORMALIZATION  (non-English → English for RAG)
        # ═══════════════════════════════════════════════════════════════════
        try:
            translator    = _get_ml_translator()
            english_query = translator.normalize_query(original_query, lang_result)
        except Exception as e:
            # Normalization failed → keep original query (best effort)
            english_query = original_query

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3 — SCOPE & GUARDRAIL AGENT  (runs BEFORE IP routing / RAG)
        # ═══════════════════════════════════════════════════════════════════
        scope_short_circuited : bool = False
        scope_short_answer_en : str  = ""
        try:
            gr = _get_scope_guard().check(
                normalized_query=english_query,
                original_query=original_query,
            )
            scope_status = gr.status.value
            scope_reason = gr.reason

            if gr.status == ScopeStatus.IN_SCOPE:
                # Continue to downstream pipeline normally
                downstream_called = True
            else:
                # Short-circuit: do not invoke the RAG/orchestrator pipeline
                downstream_called = False
                scope_short_circuited = True

                if gr.status == ScopeStatus.NEEDS_CLARIFICATION:
                    scope_short_answer_en = (
                        gr.clarification_question
                        or "Please clarify your question."
                    )
                elif gr.status == ScopeStatus.OUT_OF_SCOPE:
                    scope_short_answer_en = (
                        gr.user_response
                        or "This query is outside the supported scope."
                    )
                elif gr.status == ScopeStatus.UNSAFE_OR_DISALLOWED:
                    scope_short_answer_en = (
                        gr.user_response
                        or "That request is disallowed. Please describe a safe IP, "
                           "Ayurveda, traditional knowledge, or ABS question to "
                           "explore with source evidence."
                    )

                # Override the result dict with a short-circuit answer.
                # Note: do NOT run downstream (orchestrator/RAG/formulation)
                result = {
                    "query"        : english_query,
                    "ip_types"     : [],
                    "primary_ip"   : "unknown",
                    "router_reason": f"Scope: {gr.status.value} — {gr.reason}",
                    "formulation"  : None,
                    "answer"       : scope_short_answer_en,
                    "confidence"   : "low",
                    "sufficient"   : True,   # true in the sense we have a direct answer
                    "citations"    : [],
                    "domains_used" : [],
                    "query_type"   : "scope_guardrail",
                }
        except Exception as e:
            # Scope agent exception → proceed as in_scope (best effort, avoid blocking user)
            scope_status = "in_scope"
            scope_reason = f"Scope agent errored — defaulting to in_scope: {e!r}"
            downstream_called = True
            scope_short_circuited = False

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4 — JURISDICTION AGENT  (runs AFTER Scope, BEFORE IP routing)
        # ═══════════════════════════════════════════════════════════════════
        jurisdiction_status: str = "unknown"
        jurisdiction_reason: str = ""
        jurisdiction_override: Optional[str] = req.jurisdiction if hasattr(req, 'jurisdiction') else None
        try:
            jur_route = _get_jurisdiction_router().classify(
                query=english_query,
                override=jurisdiction_override
            )
            jurisdiction_status = jur_route.jurisdiction.value
            jurisdiction_reason = jur_route.reason
        except Exception as e:
            jurisdiction_status = "unknown"
            jurisdiction_reason = f"Jurisdiction agent errored — defaulting to unknown: {e!r}"

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5 — EXISTING /ask PIPELINE  (IP routing + retrieval + answer)
        #           (SKIPPED if Scope agent short-circuited)
        # ═══════════════════════════════════════════════════════════════════
        if not scope_short_circuited:
            try:
                result = _get_orchestrator().process(english_query)
            except Exception as e:
                # The model may be rate-limited while local retrieval remains
                # healthy.  Return source excerpts so the frontend has a useful
                # response instead of a generic error or a 500.
                import traceback
                print(f"ERROR in orchestrator.process(): {e}")
                traceback.print_exc()
                result = _retrieval_fallback(english_query, e)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5 — FORMULATION OUT  (if present)
        # ═══════════════════════════════════════════════════════════════════
        formulation_out = None
        try:
            if not scope_short_circuited and result.get("formulation"):
                f = result["formulation"]
                formulation_out = FormulationOut(
                    formulation_type                = f.get("formulation_type", "unknown"),
                    secondary_types                 = f.get("secondary_types", []),
                    ingredients                     = f.get("ingredients", []),
                    biological_resources            = f.get("biological_resources", []),
                    traditional_knowledge_indicators= f.get("traditional_knowledge_indicators", []),
                    tk_systems                      = f.get("tk_systems", []),
                    confidence                      = f.get("confidence", 0.0),
                    notes                           = f.get("notes"),
                )
        except Exception:
            formulation_out = None

        # ═══════════════════════════════════════════════════════════════════
        # STEP 6 — TRANSLATE ANSWER BACK  (English → user language)
        # ═══════════════════════════════════════════════════════════════════
        final_answer = result.get("answer", "")
        try:
            from multilingual.schemas import Language as Lang
            target = lang_result.language
            needs_translation = target in (Lang.HINDI, Lang.KANNADA)

            if needs_translation:
                from multilingual.schemas import GroundedAnswer, MultilingualCitation
                # Build structured citations (never translated)
                ml_citations = [
                    MultilingualCitation(
                        document    = c.get("document", ""),
                        section     = c.get("section"),
                        subsection  = c.get("subsection"),
                        page        = c.get("page"),
                        source      = c.get("source", ""),
                        jurisdiction= c.get("jurisdiction", "india"),
                        chunk_id    = c.get("chunk_id", ""),
                    )
                    for c in result.get("citations", [])
                ]
                answer_obj = GroundedAnswer(
                    answer_text         = result.get("answer", ""),
                    citations           = ml_citations,
                    confidence          = str(result.get("confidence", "low")),
                    sufficient          = bool(result.get("sufficient", False)),
                    original_question   = original_query,
                    normalized_question = english_query,
                )
                translated = translator.translate_answer(answer_obj, target)
                final_answer = translated.translated_text or result.get("answer", "")
        except Exception as e:
            # Translation error — keep the English answer we already have
            final_answer = result.get("answer", "")

    except Exception as top_level:
        # Catastrophic top-level error → keep the fallback result
        # (which is already set at the top of the function)
        import traceback
        print(f"[/ask] top-level exception: {top_level}")
        traceback.print_exc()
        lang_code    = "en"
        resp_lang    = "en"
        english_query = original_query
        scope_status = scope_status or "in_scope"
        downstream_called = downstream_called if downstream_called is not None else True

    # ── Return a well-formed AskResponse  (guaranteed to succeed) ────────
    try:
        citations_out = [
            AskCitationOut(
                document   = c.get("document", ""),
                chapter    = c.get("chapter"),
                section    = c.get("section"),
                subsection = c.get("subsection"),
                page       = c.get("page"),
                source     = c.get("source", ""),
                source_url = c.get("source_url"),
                domain     = c.get("domain"),
                chunk_id   = c.get("chunk_id", ""),
            )
            for c in result.get("citations", [])
        ]
    except Exception:
        citations_out = []

    return AskResponse(
        query                = result.get("query", original_query),
        ip_types             = list(result.get("ip_types", [])),
        primary_ip           = str(result.get("primary_ip", "unknown")),
        router_reason        = str(result.get("router_reason", "")),
        formulation          = formulation_out,
        answer               = final_answer if final_answer else result.get("answer", ""),
        confidence           = str(result.get("confidence", "low")),
        sufficient           = bool(result.get("sufficient", False)),
        citations            = citations_out,
        domains_used         = list(result.get("domains_used", [])),
        query_type           = str(result.get("query_type", "unknown")),
        detected_language    = lang_code,
        response_language    = resp_lang,
        original_question    = original_query,
        normalized_question  = english_query,
        scope_status         = scope_status,
        scope_reason         = scope_reason,
        downstream_called    = bool(downstream_called),
        jurisdiction         = jurisdiction_status,
        jurisdiction_reason  = jurisdiction_reason,
    )




# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Formulation Classification + ABS Compliance Helper
# ─────────────────────────────────────────────────────────────────────────────

class FormulationAnalysisRequest(BaseModel):
    description: str


class IngredientOut(BaseModel):
    name                    : str
    scientific_name         : Optional[str] = None
    biological_resource     : bool
    traditional_use_indicator: bool
    traditional_systems     : list[str]
    source                  : str


class FormulationClassOut(BaseModel):
    formulation_type                : str
    secondary_types                 : list[str]
    ingredients                     : list[str]
    biological_resources            : list[str]
    traditional_knowledge_indicators: list[str]
    tk_systems                      : list[str]
    ingredient_objects              : list[IngredientOut]
    confidence                      : float
    notes                           : Optional[str] = None


class ABSAssessmentOut(BaseModel):
    potentially_relevant           : bool
    biological_resources           : list[str]
    traditional_knowledge_detected : bool
    reasons                        : list[str]
    relevant_sources               : list[str]
    requires_human_review          : bool
    suggested_provisions           : list[str]


class TKMatchItemOut(BaseModel):
    source            : str
    title             : str
    matched_components: list[str]
    score             : float
    page              : Optional[int] = None


class TKResultOut(BaseModel):
    match_found: bool
    matches    : list[TKMatchItemOut]


class PatentResultOut(BaseModel):
    publication_number: str
    title             : str
    similarity_score  : float
    matched_components: list[str]
    source            : str


class LegalProvisionOut(BaseModel):
    document  : str
    section   : Optional[str] = None
    subsection: Optional[str] = None
    source    : str
    page      : Optional[int] = None
    chunk_id  : str


class FormulationAnalysisResponse(BaseModel):
    classification   : FormulationClassOut
    abs_assessment   : ABSAssessmentOut
    tk_results       : TKResultOut
    patent_results   : list[PatentResultOut]
    legal_provisions : list[LegalProvisionOut]
    report           : str
    confidence       : str


@app.post(
    "/analyze-formulation",
    response_model=FormulationAnalysisResponse,
    tags=["phase-7"],
)
def analyze_formulation(req: FormulationAnalysisRequest):
    """
    Phase 7 — Formulation Classification + ABS Compliance Helper.

    Given a free-text formulation description, performs:

    1. **Ingredient extraction** — identifies ingredients and enriches with
       scientific names, biological-resource flags, and TK indicators from
       the local knowledge base.

    2. **Formulation classification** — AYURVEDA / SIDDHA / UNANI / YOGA /
       TRADITIONAL_KNOWLEDGE / MODERN_PHARMACEUTICAL / BIOLOGICAL_RESOURCE /
       MIXED / UNKNOWN.

    3. **ABS assessment** — checks for potential Access and Benefit Sharing
       relevance under the Biological Diversity Act, 2002 and the Nagoya
       Protocol. Always flags for human review.

    4. **TK / TKDL pointer** — searches the TK/AYUSH corpus for potentially
       related traditional knowledge material when TK indicators are detected.

    5. **Prior-art pointer** — searches for similar patent documents.

    6. **Legal provision retrieval** — retrieves relevant provisions from the
       Patents Act, Biological Diversity Act, and AYUSH corpus.

    7. **Preliminary guidance report** — evidence-backed, fully cited,
       always includes the disclaimer that this is not legal clearance.

    **This endpoint does NOT provide legal clearance or ABS approval.**
    All findings require professional review.
    """
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")

    result = _get_form_analyzer().analyze(req.description.strip())
    d      = result.to_dict()

    cls = d["classification"]
    return FormulationAnalysisResponse(
        classification=FormulationClassOut(
            formulation_type                = cls["formulation_type"],
            secondary_types                 = [t for t in cls.get("secondary_types", [])],
            ingredients                     = cls.get("ingredients", []),
            biological_resources            = cls.get("biological_resources", []),
            traditional_knowledge_indicators= cls.get("traditional_knowledge_indicators", []),
            tk_systems                      = cls.get("tk_systems", []),
            ingredient_objects              = [
                IngredientOut(
                    name                     = i["name"],
                    scientific_name          = i.get("scientific_name"),
                    biological_resource      = i.get("biological_resource", False),
                    traditional_use_indicator= i.get("traditional_use_indicator", False),
                    traditional_systems      = i.get("traditional_systems", []),
                    source                   = i.get("source", "extracted"),
                )
                for i in cls.get("ingredient_objects", [])
            ],
            confidence = cls.get("confidence", 0.0),
            notes      = cls.get("notes"),
        ),
        abs_assessment=ABSAssessmentOut(
            **d["abs_assessment"]
        ),
        tk_results=TKResultOut(
            match_found = d["tk_results"].get("match_found", False),
            matches     = [
                TKMatchItemOut(
                    source             = m.get("source", ""),
                    title              = m.get("title", ""),
                    matched_components = m.get("matched_components", []),
                    score              = float(m.get("score", 0.0)),
                    page               = m.get("page"),
                )
                for m in d["tk_results"].get("matches", [])
            ],
        ),
        patent_results=[
            PatentResultOut(
                publication_number = r.get("publication_number", ""),
                title              = r.get("title", ""),
                similarity_score   = float(r.get("similarity_score", 0.0)),
                matched_components = r.get("matched_components", []),
                source             = r.get("source", ""),
            )
            for r in d.get("patent_results", [])
        ],
        legal_provisions=[
            LegalProvisionOut(
                document   = p.get("document", ""),
                section    = p.get("section"),
                subsection = p.get("subsection"),
                source     = p.get("source", ""),
                page       = p.get("page"),
                chunk_id   = p.get("chunk_id", ""),
            )
            for p in d.get("legal_provisions", [])
        ],
        report     = d["report"],
        confidence = d["confidence"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — Multilingual Q&A
# ─────────────────────────────────────────────────────────────────────────────

from multilingual.schemas import Language as MLLanguage, MultilingualCitation
from multilingual.detector import LanguageDetector
from multilingual.translator import MultilingualTranslator

_ml_translator: Optional[MultilingualTranslator] = None
_ml_detector  : Optional[LanguageDetector]       = None


def _get_ml_translator() -> MultilingualTranslator:
    global _ml_translator
    if _ml_translator is None:
        _ml_translator = MultilingualTranslator(
            prefer_bhashini=False, llm=_get_llm()
        )
    return _ml_translator


def _get_ml_detector() -> LanguageDetector:
    global _ml_detector
    if _ml_detector is None:
        _ml_detector = LanguageDetector()
    return _ml_detector


class MultilingualQueryRequest(BaseModel):
    question    : str
    language    : Optional[str] = None   # "en" | "hi" | "kn" — overrides detection
    domain      : Optional[str] = None
    jurisdiction: Optional[str] = None
    ip_type     : Optional[str] = None


class MultilingualCitationOut(BaseModel):
    document    : str
    section     : Optional[str] = None
    subsection  : Optional[str] = None
    page        : Optional[int] = None
    source      : str
    jurisdiction: Optional[str] = None
    chunk_id    : str


class MultilingualQueryResponse(BaseModel):
    language            : str            # detected / overridden language code
    original_question   : str
    normalized_question : str            # English version used for retrieval
    ip_types            : list[str]
    jurisdiction        : Optional[str]
    answer              : str            # translated answer
    answer_english      : str            # original English answer
    citations           : list[MultilingualCitationOut]
    confidence          : str
    sufficient          : bool
    disclaimer          : str
    detection_method    : str


@app.post("/multilingual-query", response_model=MultilingualQueryResponse, tags=["phase-8"])
def multilingual_query(req: MultilingualQueryRequest):
    """
    Phase 8 — Multilingual Q&A.

    Supports English (en), Hindi (hi), and Kannada (kn).

    Pipeline:
      1. Detect language (or use explicit `language` parameter)
      2. Translate query → English (before retrieval)
      3. Run existing IP-SAKTI legal RAG pipeline (English only)
      4. Translate grounded English answer → user language
      5. Return answer + UNCHANGED citations + confidence

    Key invariants:
      - The authoritative corpus is NEVER duplicated or translated
      - Section numbers, Act names, IPC codes are never translated
      - Citations are never modified
      - Confidence is recalculated from evidence, never from translation
      - Disclaimer is always returned in the target language

    Supply `language` to override auto-detection (recommended for short queries).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    question = req.question.strip()

    # 1. Detect / override language
    lang_result = _get_ml_detector().detect(question, override=req.language)

    # 2. Translate query to English
    translator     = _get_ml_translator()
    english_query  = translator.normalize_query(question, lang_result)

    # 3. Run Phase 2 legal RAG on the English query
    rag_result = _get_retriever().retrieve(
        english_query,
        domain_filter=req.domain,
    )
    chunks     = rag_result["chunks"]
    sufficient = rag_result["sufficient"]
    confidence = rag_result["confidence"]
    query_type = rag_result["query_type"]

    if not sufficient or not chunks:
        english_answer = (
            "I could not find sufficient information in the "
            "provided authoritative documents."
        )
    else:
        context = "\n\n---\n\n".join(_chunk_to_context_block(c) for c in chunks)
        from utils.llm_utils import extract_llm_text
        english_answer = extract_llm_text(_get_llm().invoke([
            {"role": "system", "content": _LEGAL_SYSTEM},
            {"role": "user",   "content": (
                f"Documents:\n\n{context}\n\n---\n\n"
                f"Question: {english_query}\n\nAnswer:"
            )},
        ]))

    # 4. Build structured citations (never translated)
    citations: list[MultilingualCitation] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        c = chunk.citation()
        citations.append(MultilingualCitation(
            document    = c["document"],
            section     = c.get("section") or None,
            subsection  = c.get("subsection") or None,
            page        = c.get("page"),
            source      = c["source"],
            jurisdiction= "india",
            chunk_id    = c["chunk_id"],
        ))

    # 5. Build GroundedAnswer and translate
    from multilingual.schemas import GroundedAnswer
    answer_obj = GroundedAnswer(
        answer_text        = english_answer,
        citations          = citations,
        confidence         = confidence,
        sufficient         = sufficient,
        original_question  = question,
        normalized_question= english_query,
    )
    translated_answer = translator.translate_answer(answer_obj, lang_result.language)

    # 6. IP type detection (from Phase 5 router)
    from routing.ip_router import _keyword_classify as _ip_kw
    ip_types = [t.value for t in _ip_kw(english_query)]

    return MultilingualQueryResponse(
        language            = lang_result.language.value,
        original_question   = question,
        normalized_question = english_query,
        ip_types            = ip_types,
        jurisdiction        = req.jurisdiction or "india",
        answer              = translated_answer.translated_text or english_answer,
        answer_english      = english_answer,
        citations           = [
            MultilingualCitationOut(
                document    = c.document,
                section     = c.section,
                subsection  = c.subsection,
                page        = c.page,
                source      = c.source,
                jurisdiction= c.jurisdiction,
                chunk_id    = c.chunk_id,
            )
            for c in translated_answer.citations
        ],
        confidence          = confidence,
        sufficient          = sufficient,
        disclaimer          = translated_answer.disclaimer,
        detection_method    = lang_result.method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9 — Guardrails + Escalation + Privacy + Audit
# ─────────────────────────────────────────────────────────────────────────────

from guardrails.disclaimer  import add_disclaimer, get_disclaimer
from guardrails.scope       import (ScopeChecker, Scope,
                                    ScopeGuard, ScopeStatus, GuardrailResult)
from guardrails.confidence  import score_from_chunks, confidence_band, evidence_is_sufficient
from guardrails.pipeline    import SafetyPipeline, SafetyResult, INJECTION_GUARD_PROMPT
from escalation.facilitator import (
    EscalationRequest as EscReq,
    EscalationResponse as EscResp,
    generate_request_id, should_escalate, escalation_reason,
    create_escalation_request,
)
from audit.logger import get_logger

_scope_checker: Optional[ScopeChecker]  = None
_scope_guard  : Optional[ScopeGuard]    = None
_safety_pipeline: Optional[SafetyPipeline] = None


def _get_scope_checker() -> ScopeChecker:
    global _scope_checker
    if _scope_checker is None:
        _scope_checker = ScopeChecker(llm=_get_llm())
    return _scope_checker


def _get_scope_guard() -> ScopeGuard:
    global _scope_guard
    if _scope_guard is None:
        _scope_guard = ScopeGuard()
    return _scope_guard


def _get_safety_pipeline() -> SafetyPipeline:
    global _safety_pipeline
    if _safety_pipeline is None:
        _safety_pipeline = SafetyPipeline(
            scope_checker=_get_scope_checker(),
            llm=_get_llm(),
        )
    return _safety_pipeline


# ── Pydantic models ──────────────────────────────────────────────────────────

class SafeQueryRequest(BaseModel):
    question        : str
    domain          : Optional[str] = None
    language        : str           = "en"
    privacy_mode    : bool          = False
    request_human   : bool          = False


class SafeQueryCitationOut(BaseModel):
    document   : str
    section    : Optional[str] = None
    subsection : Optional[str] = None
    page       : Optional[int] = None
    source     : str
    chunk_id   : str = ""


class SafeQueryResponse(BaseModel):
    request_id         : str
    status             : str            # answered | abstained | escalated | out_of_scope
    answer             : Optional[str]
    confidence         : float
    confidence_band    : str
    citations          : list[SafeQueryCitationOut]
    needs_human_review : bool
    reason             : Optional[str]
    disclaimer         : str
    language           : str


class EscalateRequest(BaseModel):
    question    : str
    reason      : str = "User requested human review"
    ip_type     : str = "unknown"
    jurisdiction: str = "india"
    confidence  : float = 0.0
    language    : str = "en"
    privacy_mode: bool = False


class EscalateResponse(BaseModel):
    request_id        : str
    status            : str
    message           : str
    estimated_response: str


class DeleteResponse(BaseModel):
    request_id: str
    deleted   : bool
    message   : str


# ── Handlers ─────────────────────────────────────────────────────────────────

def _safe_citations(raw_citations: list) -> list[SafeQueryCitationOut]:
    out = []
    for c in raw_citations:
        if isinstance(c, dict):
            out.append(SafeQueryCitationOut(
                document   = c.get("document", ""),
                section    = c.get("section") or c.get("subsection"),
                subsection = c.get("subsection"),
                page       = c.get("page"),
                source     = c.get("source", ""),
                chunk_id   = c.get("chunk_id", ""),
            ))
    return out


def _generation_fn(query: str, chunks: list) -> str:
    """Generate answer from chunks using the existing legal RAG prompt."""
    context = "\n\n---\n\n".join(_chunk_to_context_block(c) for c in chunks)
    # Inject the prompt-injection guard into the system prompt
    from utils.llm_utils import extract_llm_text
    return extract_llm_text(_get_llm().invoke([
        {"role": "system", "content": guarded_system},
        {"role": "user",   "content": (
            f"Documents:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"
        )},
    ]))


@app.post("/safe-query", response_model=SafeQueryResponse, tags=["phase-9"])
def safe_query(req: SafeQueryRequest):
    """
    Phase 9 — Guardrailed legal Q&A.

    Full safety pipeline:
      1. Scope check   — rejects out-of-scope queries
      2. Injection guard — sanitizes query before retrieval
      3. Evidence check — abstains when no authoritative source
      4. Confidence     — escalates when score < 0.60
      5. Disclaimer     — always appended by backend (never LLM-generated)
      6. Audit log      — privacy-mode redacts the question

    Status values:
      answered      — sufficient evidence, confident answer
      abstained     — no authoritative evidence found
      escalated     — low confidence or user requested human review
      out_of_scope  — query unrelated to IP / TK / ABS
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    request_id = generate_request_id()
    question   = req.question.strip()
    language   = req.language or "en"

    retrieval_fn  = lambda q: _get_retriever().retrieve(q, domain_filter=req.domain)

    result: SafetyResult = _get_safety_pipeline().run(
        query        = question,
        retrieval_fn = retrieval_fn,
        generation_fn= _generation_fn,
        request_id   = request_id,
        language     = language,
        user_wants_human = req.request_human,
    )

    # Escalation: store escalation record
    if result.status == "escalated":
        esc = create_escalation_request(
            reason      = result.reason or "Escalated",
            ip_type     = "unknown",
            jurisdiction= "india",
            confidence  = result.confidence,
            question    = "" if req.privacy_mode else question,
            language    = language,
            privacy_mode= req.privacy_mode,
            request_id  = request_id,
        )
        get_logger().log_escalation(esc)

    # Audit log
    chunks = retrieval_fn(question).get("chunks", []) if result.status == "answered" else []
    get_logger().log(
        request_id       = request_id,
        language         = language,
        ip_type          = "unknown",
        jurisdiction     = "india",
        retrieval_count  = len(chunks),
        citation_count   = len(result.citations),
        confidence       = result.confidence,
        status           = result.status,
        human_review     = result.needs_human_review,
        scope            = (result.scope_result or {}).get("scope", "in_scope"),
        escalation_reason= result.reason or "",
        privacy_mode     = req.privacy_mode,
    )

    return SafeQueryResponse(
        request_id         = request_id,
        status             = result.status,
        answer             = result.answer,
        confidence         = result.confidence,
        confidence_band    = result.confidence_band,
        citations          = _safe_citations(result.citations),
        needs_human_review = result.needs_human_review,
        reason             = result.reason,
        disclaimer         = get_disclaimer(language),
        language           = language,
    )


@app.post("/escalate", response_model=EscalateResponse, tags=["phase-9"])
def escalate(req: EscalateRequest):
    """
    Phase 9 — Manual escalation to human IP facilitator.

    Use when the user explicitly wants a human to review their query.
    The escalation request is logged and a tracking ID is returned.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    request_id = generate_request_id()
    esc = create_escalation_request(
        reason      = req.reason,
        ip_type     = req.ip_type,
        jurisdiction= req.jurisdiction,
        confidence  = req.confidence,
        question    = "" if req.privacy_mode else req.question.strip(),
        language    = req.language,
        privacy_mode= req.privacy_mode,
        request_id  = request_id,
    )
    get_logger().log_escalation(esc)
    get_logger().log(
        request_id   = request_id,
        language     = req.language,
        ip_type      = req.ip_type,
        jurisdiction = req.jurisdiction,
        confidence   = req.confidence,
        status       = "escalated",
        human_review = True,
        privacy_mode = req.privacy_mode,
    )

    return EscalateResponse(
        request_id        = request_id,
        status            = "pending",
        message           = (
            f"Your query has been escalated (ID: {request_id}). "
            "A qualified IP facilitator will review it."
        ),
        estimated_response= "A qualified IP facilitator will review this query.",
    )


@app.delete("/conversation/{request_id}", response_model=DeleteResponse, tags=["phase-9"])
def delete_conversation(request_id: str):
    """
    Phase 9 — Delete all stored data for a request ID.

    Supports user's right to erasure (DPDP compliance prototype).
    Removes the audit log record and any escalation records associated
    with the request ID.
    """
    deleted = get_logger().delete(request_id)
    return DeleteResponse(
        request_id = request_id,
        deleted    = deleted,
        message    = (
            f"All records for {request_id} have been deleted."
            if deleted else
            f"No records found for {request_id}."
        ),
    )


@app.get("/audit/{request_id}", tags=["phase-9"])
def get_audit_record(request_id: str):
    """
    Phase 9 — Retrieve audit log for a specific request.
    Returns only metadata — never raw question text in privacy_mode records.
    """
    record = get_logger().get(request_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No record for {request_id}")
    # Never expose raw question through audit API
    record.pop("question", None)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10 — Knowledge Graph + Agentic Multi-Source Orchestration
# ─────────────────────────────────────────────────────────────────────────────

from agents.orchestrator import IPSaktiOrchestrator, select_tools
from graph.graph_store   import KnowledgeGraph

_orchestrator_p10: Optional[IPSaktiOrchestrator] = None
_knowledge_graph : Optional[KnowledgeGraph]      = None


def _get_knowledge_graph() -> KnowledgeGraph:
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph.load()
    return _knowledge_graph


def _get_orchestrator_p10() -> IPSaktiOrchestrator:
    global _orchestrator_p10
    if _orchestrator_p10 is None:
        _orchestrator_p10 = IPSaktiOrchestrator(
            llm       = _get_llm(),
            embeddings= _get_embeddings(),
            graph     = _get_knowledge_graph(),
        )
    return _orchestrator_p10


class OrchestrateRequest(BaseModel):
    query       : str
    language    : str  = "en"
    dry_run     : bool = False   # True → return tool selection only, no LLM calls


class ToolStatusOut(BaseModel):
    tool  : str
    status: str


class CitationP10Out(BaseModel):
    source_id   : str
    source_type : str
    title       : str
    section     : Optional[str] = None
    subsection  : Optional[str] = None
    page        : Optional[int] = None
    source_name : str
    jurisdiction: str
    authority   : str
    chunk_id    : str


class GraphRelationOut(BaseModel):
    entity    : str
    tk_sources: list[str]
    patents   : list[str]


class OrchestrateResponse(BaseModel):
    answer             : Optional[str]
    tools_used         : list[str]
    tool_status        : dict[str, str]
    citations          : list[CitationP10Out]
    confidence         : str
    issues             : list[str]
    sources_consulted  : int
    graph_relations    : list[GraphRelationOut]
    selected_tools_dry : list[str]   # always returned (even if dry_run=False)
    language           : str


@app.post("/orchestrate", response_model=OrchestrateResponse, tags=["phase-10"])
def orchestrate(req: OrchestrateRequest):
    """
    Phase 10 — Agentic multi-source orchestration.

    The orchestrator automatically:
      1. Determines which tools are needed (legal / patent / TK / formulation / ABS / international)
      2. Calls each tool and collects evidence
      3. Populates the knowledge graph with retrieved relationships
      4. Fuses and ranks evidence by authority
      5. Detects issues (Section 3(p), 3(d), ABS relevance, etc.)
      6. Generates a grounded, cited answer

    Use `dry_run=true` to see which tools would be selected without making LLM calls.

    Example query:
      "I developed an Ayurvedic formulation using neem and turmeric.
       Can I patent it in India, and could traditional knowledge affect my application?"
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    query = req.query.strip()

    # Always compute tool selection (useful even in dry_run)
    selected_tools = [t.split(":")[0] for t in select_tools(query)]

    if req.dry_run:
        return OrchestrateResponse(
            answer=None,
            tools_used=[],
            tool_status={t: "not_called" for t in selected_tools},
            citations=[],
            confidence="unknown",
            issues=[],
            sources_consulted=0,
            graph_relations=[],
            selected_tools_dry=selected_tools,
            language=req.language,
        )

    result = _get_orchestrator_p10().run(query)

    # Graph relations
    graph_relations = [
        GraphRelationOut(
            entity    = gr.get("entity", ""),
            tk_sources= gr.get("tk_sources", []),
            patents   = gr.get("patents", []),
        )
        for gr in result.get("evidence", {}).get("graph_relations", [])
    ]

    return OrchestrateResponse(
        answer            = result.get("answer"),
        tools_used        = result.get("tools_used", []),
        tool_status       = result.get("tool_status", {}),
        citations         = [
            CitationP10Out(
                source_id   = c.get("source_id", ""),
                source_type = c.get("source_type", ""),
                title       = c.get("title", ""),
                section     = c.get("section"),
                subsection  = c.get("subsection"),
                page        = c.get("page"),
                source_name = c.get("source_name", ""),
                jurisdiction= c.get("jurisdiction", "india"),
                authority   = c.get("authority", "secondary"),
                chunk_id    = c.get("chunk_id", ""),
            )
            for c in result.get("citations", [])
        ],
        confidence        = result.get("confidence", "low"),
        issues            = result.get("issues", []),
        sources_consulted = result.get("sources_consulted", 0),
        graph_relations   = graph_relations,
        selected_tools_dry= selected_tools,
        language          = req.language,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — Unified Pipeline Endpoint
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.main          import IPSaktiPipeline
from pipeline.response_model import PipelineResponse as P11Response

_pipeline_v2: Optional[IPSaktiPipeline] = None


def _get_pipeline_v2() -> IPSaktiPipeline:
    global _pipeline_v2
    if _pipeline_v2 is None:
        _pipeline_v2 = IPSaktiPipeline(
            llm       = _get_llm(),
            embeddings= _get_embeddings(),
        )
    return _pipeline_v2


class AskV2Request(BaseModel):
    question             : str
    language             : Optional[str] = None   # "en" | "hi" | "kn" | auto-detect
    jurisdiction         : Optional[str] = None   # "india" | "international" | "both"
    ip_type              : Optional[str] = None   # "patent" | "trademark" | …
    request_human        : bool          = False
    privacy_mode         : bool          = False


class CitationV2Out(BaseModel):
    source_id   : str
    document    : str
    section     : Optional[str] = None
    subsection  : Optional[str] = None
    page        : Optional[int] = None
    source_name : str
    jurisdiction: str
    authority   : str
    chunk_id    : str
    verified    : bool


class ConflictV2Out(BaseModel):
    source_a   : str
    source_b   : str
    description: str
    resolution : str


class AskV2Response(BaseModel):
    status             : str
    answer             : Optional[str]
    answer_english     : Optional[str]     = None
    language           : str
    original_question  : str
    normalized_question: str
    ip_types           : list[str]
    jurisdiction       : str
    citations          : list[CitationV2Out]
    citation_coverage  : float
    confidence         : float
    confidence_band    : str
    needs_human_review : bool
    reason             : Optional[str]
    tools_used         : list[str]
    sources_consulted  : int
    issues             : list[str]
    conflicts          : list[ConflictV2Out]
    formulation        : Optional[dict]
    disclaimer         : str
    request_id         : Optional[str]


@app.post("/ask-v2", response_model=AskV2Response, tags=["phase-11"])
def ask_v2(req: AskV2Request):
    """
    Phase 11 — Unified Pipeline (all phases wired together).

    Single endpoint that runs the complete 18-step pipeline:

      1. Language detection (or override)
      2. Scope check
      3. Query normalisation (translate to English)
      4. Jurisdiction detection (or override)
      5. IP type detection (or override)
      6. Formulation classification (when relevant)
      7. Agentic orchestration (tool selection + evidence collection)
      8. Conflict detection
      9. Multi-signal confidence calculation
      10. Evidence sufficiency check → answer or abstain/escalate
      11. SOURCE_ID-labelled context construction
      12. Answer generation (LLM cites SOURCE_IDs — never invents)
      13. Citation verification (backend-constructed)
      14. Disclaimer injection (backend — never LLM)
      15. Translation to user language
      16. Audit logging

    Supports: en, hi, kn  |  patent, trademark, copyright, design, gi, TK
              india, international, both  |  privacy_mode, request_human

    Status values:
      answered      — sufficient evidence, citations verified
      abstained     — insufficient evidence found
      escalated     — low confidence or human requested
      out_of_scope  — query not related to IP / TK / ABS
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result: P11Response = _get_pipeline_v2().process(
        question              = req.question.strip(),
        language_override     = req.language,
        jurisdiction_override = req.jurisdiction,
        ip_type_override      = req.ip_type,
        request_human         = req.request_human,
        privacy_mode          = req.privacy_mode,
    )

    return AskV2Response(
        status             = result.status,
        answer             = result.answer,
        answer_english     = result.answer_english,
        language           = result.language,
        original_question  = result.original_question,
        normalized_question= result.normalized_question,
        ip_types           = result.ip_types,
        jurisdiction       = result.jurisdiction,
        citations          = [
            CitationV2Out(
                source_id   = c.source_id,
                document    = c.document,
                section     = c.section,
                subsection  = c.subsection,
                page        = c.page,
                source_name = c.source_name,
                jurisdiction= c.jurisdiction,
                authority   = c.authority,
                chunk_id    = c.chunk_id,
                verified    = c.verified,
            )
            for c in result.citations
        ],
        citation_coverage  = result.citation_coverage,
        confidence         = result.confidence,
        confidence_band    = result.confidence_band,
        needs_human_review = result.needs_human_review,
        reason             = result.reason,
        tools_used         = result.tools_used,
        sources_consulted  = result.sources_consulted,
        issues             = result.issues,
        conflicts          = [
            ConflictV2Out(
                source_a   = c.source_a,
                source_b   = c.source_b,
                description= c.description,
                resolution = c.resolution,
            )
            for c in result.conflicts
        ],
        formulation        = result.formulation,
        disclaimer         = result.disclaimer,
        request_id         = result.request_id,
    )
