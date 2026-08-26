"""
Query Orchestrator  (Phase 5)
──────────────────────────────
The main entry point for the unified /ask endpoint.

Flow
────
User query
    │
    ├─ IP Router  → QueryRoute (one or more IP types)
    │
    ├─ [if formulation keywords detected]
    │    └─ Formulation Classifier  → FormulationClassification
    │
    ├─ Domain Retriever  → chunks (domain-filtered, multi-domain aware)
    │
    ├─ Evidence check  → sufficient?
    │    ├─ NO  → "Insufficient evidence" response
    │    └─ YES
    │         └─ LLM with domain-specific prompt  → answer + citations
    │
    └─ Structured response

Output schema
─────────────
{
  "query"      : str,
  "ip_types"   : list[str],
  "primary_ip" : str,
  "formulation": FormulationClassification | None,
  "answer"     : str,
  "confidence" : str,
  "sufficient" : bool,
  "citations"  : list[dict],
  "domains_used": list[str],
  "query_type" : str,
}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from routing.schemas import IPType, QueryRoute
from routing.ip_router import IPRouter
from formulation.schemas import FormulationClassification
from formulation.classifier import FormulationClassifier
from retrieval.domain_retriever import DomainRetriever, get_domain_prompt
from ingestion.schema import LegalChunk


# ─────────────────────────────────────────────────────────────────────────────
# Formulation detection heuristic
# ─────────────────────────────────────────────────────────────────────────────

_FORMULATION_TRIGGER = re.compile(
    r"\bformulat|\bherbal\b|\bayurved|\bingredient|\bextract\b|"
    r"\bcomposit|\bmedicin|\bplant\b|\bneem\b|\bturmeric\b|"
    r"\bashwagandha\b|\bsiddha\b|\bunani\b|\btkdl\b|"
    r"\btraditional\s+know|\btraditional\s+med",
    re.IGNORECASE,
)


def _needs_formulation_check(query: str, route: QueryRoute) -> bool:
    """True if the query likely involves a formulation that needs classification."""
    return (
        IPType.TRADITIONAL_KNOWLEDGE in route.ip_types
        or IPType.PATENT in route.ip_types
    ) and bool(_FORMULATION_TRIGGER.search(query))


# ─────────────────────────────────────────────────────────────────────────────
# Context block builder
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_context(chunk: LegalChunk) -> str:
    parts = [f"[{chunk.title}"]
    for attr, prefix in [
        ("chapter",    ""),
        ("section",    ""),
        ("subsection", "§ "),
        ("page",       "p. "),
    ]:
        val = getattr(chunk, attr, None)
        if val:
            parts.append(f"  {prefix}{val}")
    parts.append(f"  Source: {chunk.source}]")
    return "\n".join(parts) + f"\n\n{chunk.text}"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class QueryOrchestrator:
    """
    Unified query orchestrator for the /ask endpoint.

    Parameters
    ----------
    embeddings : shared HuggingFaceEmbeddings
    llm        : shared ChatGroq
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        llm: ChatGroq,
    ):
        self._llm        = llm
        self._router     = IPRouter(llm=llm)
        self._classifier = FormulationClassifier(llm=llm)
        self._retriever  = DomainRetriever(embeddings=embeddings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, query: str) -> dict:
        """
        Full pipeline: route → classify → retrieve → generate → respond.

        Returns a structured dict (serialisable to JSON).
        """
        # 1. IP type routing
        route = self._router.classify(query)

        # 2. Formulation classification (only when relevant)
        formulation: Optional[FormulationClassification] = None
        if _needs_formulation_check(query, route):
            formulation = self._classifier.classify(query)
            # If TK is detected, ensure it's included in route
            if formulation.is_tk_relevant():
                types = list(route.ip_types)
                if IPType.TRADITIONAL_KNOWLEDGE not in types:
                    types.append(IPType.TRADITIONAL_KNOWLEDGE)
                route = QueryRoute(
                    ip_types   = types,
                    primary    = route.primary,
                    confidence = route.confidence,
                    reason     = route.reason,
                    is_multi   = len(types) > 1,
                )

        # 3. Domain-aware retrieval
        retrieval = self._retriever.retrieve_by_route(query, route)
        chunks    = retrieval["chunks"]
        sufficient= retrieval["sufficient"]
        confidence= retrieval["confidence"]
        domains   = retrieval.get("domains_used", route.domain_filters)

        # 4. Evidence check
        if not sufficient or not chunks:
            return self._insufficient_response(query, route, formulation)

        # 5. Generate answer with domain-specific prompt
        context  = "\n\n---\n\n".join(_chunk_to_context(c) for c in chunks)
        sys_prompt = get_domain_prompt([d for d in domains if d])
        user_msg   = (
            f"Documents:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}\n\nAnswer:"
        )

        answer = self._llm.invoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg},
        ]).content.strip()

        # 6. Build citations
        citations = self._build_citations(chunks)

        return {
            "query"       : query,
            "ip_types"    : [t.value for t in route.ip_types],
            "primary_ip"  : route.primary.value,
            "router_reason": route.reason,
            "formulation" : formulation.model_dump() if formulation else None,
            "answer"      : answer,
            "confidence"  : confidence,
            "sufficient"  : True,
            "citations"   : citations,
            "domains_used": [d for d in domains if d],
            "query_type"  : retrieval.get("query_type", "semantic"),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _insufficient_response(
        query: str,
        route: QueryRoute,
        formulation: Optional[FormulationClassification],
    ) -> dict:
        return {
            "query"       : query,
            "ip_types"    : [t.value for t in route.ip_types],
            "primary_ip"  : route.primary.value,
            "router_reason": route.reason,
            "formulation" : formulation.model_dump() if formulation else None,
            "answer"      : (
                "I could not find sufficient information in the "
                "provided authoritative documents."
            ),
            "confidence"  : "low",
            "sufficient"  : False,
            "citations"   : [],
            "domains_used": [],
            "query_type"  : "unknown",
        }

    @staticmethod
    def _build_citations(chunks: list[LegalChunk]) -> list[dict]:
        seen: set[str] = set()
        citations = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            c = chunk.citation()
            citations.append({
                "document"  : c["document"],
                "chapter"   : c.get("chapter")    or None,
                "section"   : c.get("section")    or None,
                "subsection": c.get("subsection") or None,
                "page"      : c.get("page"),
                "source"    : c["source"],
                "source_url": c.get("source_url") or None,
                "domain"    : c.get("domain", ""),
                "chunk_id"  : c["chunk_id"],
            })
        return citations
