"""
Query Orchestrator  (Phase 5 + 6)
───────────────────────────────────
Unified pipeline for the /ask and /jurisdictional-query endpoints.

Phase 5 flow  (/ask)
────────────────────
  query → IP router → formulation classifier → domain retrieval → answer

Phase 6 flow  (/jurisdictional-query)
───────────────────────────────────────
  query → IP router → jurisdiction router
        → [india]  separate retrieval → separate answer
        → [intl]   separate retrieval → separate answer
        → [both]   two independent retrievals → two independent answers
                   → optional comparison paragraph

The two answer sets are NEVER merged. This satisfies the requirement:
"two answer-sets kept visibly separate."
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
from routing.jurisdiction import Jurisdiction, JurisdictionRoute, JurisdictionRouter
from formulation.schemas import FormulationClassification
from formulation.classifier import FormulationClassifier
from retrieval.domain_retriever import DomainRetriever, get_domain_prompt
from retrieval.jurisdiction_retriever import JurisdictionRetriever, JURISDICTION_PROMPTS
from ingestion.schema import LegalChunk


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_FORMULATION_TRIGGER = re.compile(
    r"\bformulat|\bherbal\b|\bayurved|\bingredient|\bextract\b|"
    r"\bcomposit|\bmedicin|\bplant\b|\bneem\b|\bturmeric\b|"
    r"\bashwagandha\b|\bsiddha\b|\bunani\b|\btkdl\b|"
    r"\btraditional\s+know|\btraditional\s+med",
    re.IGNORECASE,
)


def _needs_formulation_check(query: str, route: QueryRoute) -> bool:
    return (
        IPType.TRADITIONAL_KNOWLEDGE in route.ip_types
        or IPType.PATENT in route.ip_types
    ) and bool(_FORMULATION_TRIGGER.search(query))


def _chunk_to_context(chunk: LegalChunk) -> str:
    parts = [f"[{chunk.title}"]
    for attr, prefix in [("chapter", ""), ("section", ""), ("subsection", "§ "), ("page", "p. ")]:
        val = getattr(chunk, attr, None)
        if val:
            parts.append(f"  {prefix}{val}")
    jur = getattr(chunk, "jurisdiction", "india")
    parts.append(f"  Jurisdiction: {jur}  |  Source: {chunk.source}]")
    return "\n".join(parts) + f"\n\n{chunk.text}"


def _build_citations(chunks: list[LegalChunk]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        c = chunk.citation()
        out.append({
            "document"    : c["document"],
            "chapter"     : c.get("chapter")    or None,
            "section"     : c.get("section")    or None,
            "subsection"  : c.get("subsection") or None,
            "page"        : c.get("page"),
            "source"      : c["source"],
            "source_url"  : c.get("source_url") or None,
            "domain"      : c.get("domain", ""),
            "jurisdiction": getattr(chunk, "jurisdiction", "india"),
            "chunk_id"    : c["chunk_id"],
        })
    return out


def _generate_answer(
    llm: ChatGroq,
    query: str,
    chunks: list[LegalChunk],
    domain_prompt: str,
    jurisdiction_note: str = "",
) -> str:
    if not chunks:
        return "I could not find sufficient information in the provided authoritative documents."
    context  = "\n\n---\n\n".join(_chunk_to_context(c) for c in chunks)
    sys_prompt = domain_prompt
    if jurisdiction_note:
        sys_prompt = jurisdiction_note + "\n\n" + sys_prompt
    user_msg = f"Documents:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"
    return llm.invoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_msg},
    ]).content.strip()


_COMPARISON_SYSTEM = """\
You are an IP law comparison assistant.
You are given two separately generated answers:
  - INDIA ANSWER: based solely on Indian law
  - INTERNATIONAL ANSWER: based solely on international instruments

Write a brief COMPARISON SUMMARY (3–6 sentences) that:
1. Highlights the key differences between the two jurisdictions on this topic.
2. Notes where Indian law aligns with or diverges from the international standard.
3. Does NOT introduce any new legal claims — only reference what is in the two answers.
4. Ends with: "This comparison is for informational purposes only."

Do not repeat the full answers — only synthesise the differences."""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class QueryOrchestrator:
    """
    Unified orchestrator for /ask (Phase 5) and /jurisdictional-query (Phase 6).
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        llm: ChatGroq,
    ):
        self._llm          = llm
        self._ip_router    = IPRouter(llm=llm)
        self._jur_router   = JurisdictionRouter(llm=llm)
        self._classifier   = FormulationClassifier(llm=llm)
        self._domain_ret   = DomainRetriever(embeddings=embeddings)
        self._jur_ret      = JurisdictionRetriever(embeddings=embeddings)

    # ──────────────────────────────────────────────────────────────────────
    # Phase 5 — /ask  (domain-routed, single answer)
    # ──────────────────────────────────────────────────────────────────────

    def process(self, query: str) -> dict:
        """Phase 5: domain-aware single-jurisdiction answer."""

        # 1. IP type routing
        route = self._ip_router.classify(query)

        # 2. Formulation classification
        formulation: Optional[FormulationClassification] = None
        if _needs_formulation_check(query, route):
            formulation = self._classifier.classify(query)
            if formulation.is_tk_relevant():
                types = list(route.ip_types)
                if IPType.TRADITIONAL_KNOWLEDGE not in types:
                    types.append(IPType.TRADITIONAL_KNOWLEDGE)
                route = QueryRoute(
                    ip_types=types, primary=route.primary,
                    confidence=route.confidence, reason=route.reason,
                    is_multi=len(types) > 1,
                )

        # 3. Domain-aware retrieval
        retrieval  = self._domain_ret.retrieve_by_route(query, route)
        chunks     = retrieval["chunks"]
        sufficient = retrieval["sufficient"]
        confidence = retrieval["confidence"]
        domains    = retrieval.get("domains_used", route.domain_filters)

        if not sufficient or not chunks:
            return self._insufficient(query, route, formulation)

        # 4. Generate
        answer = _generate_answer(
            self._llm, query, chunks,
            get_domain_prompt([d for d in domains if d]),
        )

        return {
            "query"        : query,
            "ip_types"     : [t.value for t in route.ip_types],
            "primary_ip"   : route.primary.value,
            "router_reason": route.reason,
            "formulation"  : formulation.model_dump() if formulation else None,
            "answer"       : answer,
            "confidence"   : confidence,
            "sufficient"   : True,
            "citations"    : _build_citations(chunks),
            "domains_used" : [d for d in domains if d],
            "query_type"   : retrieval.get("query_type", "semantic"),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Phase 6 — /jurisdictional-query  (jurisdiction-split answers)
    # ──────────────────────────────────────────────────────────────────────

    def process_jurisdictional(
        self,
        query: str,
        jurisdiction_override: Optional[str] = None,
        ip_type_override: Optional[str] = None,
    ) -> dict:
        """
        Phase 6: jurisdiction-aware pipeline.

        Returns
        -------
        {
          "query"        : str,
          "ip_types"     : list[str],
          "jurisdiction" : "india" | "international" | "both",
          "formulation"  : dict | None,
          "india"        : { answer, citations, confidence, sufficient } | None,
          "international": { answer, citations, confidence, sufficient } | None,
          "comparison"   : str | None,   # only when jurisdiction == "both"
        }
        """
        # 1. IP type routing
        route = self._ip_router.classify(query)
        if ip_type_override:
            # Wrap override into route
            from routing.schemas import IPType
            try:
                override_type = IPType(ip_type_override.lower())
                route = QueryRoute(
                    ip_types=[override_type], primary=override_type,
                    confidence=1.0, reason="User-supplied IP type override.",
                    is_multi=False,
                )
            except ValueError:
                pass

        # 2. Jurisdiction routing
        jur_route = self._jur_router.classify(query, override=jurisdiction_override)
        jurisdiction = jur_route.jurisdiction

        # 3. Formulation classification
        formulation: Optional[FormulationClassification] = None
        if _needs_formulation_check(query, route):
            formulation = self._classifier.classify(query)

        # Primary domain for filtering
        domain = route.domain_filters[0] if route.domain_filters else None

        # 4. Jurisdiction-split retrieval + generation
        india_result = intl_result = comparison = None

        if jurisdiction in (Jurisdiction.INDIA, Jurisdiction.BOTH, Jurisdiction.UNKNOWN):
            india_ret = self._jur_ret.retrieve_india(query, domain)
            india_chunks = india_ret["chunks"]
            india_answer = _generate_answer(
                self._llm, query, india_chunks,
                get_domain_prompt([domain] if domain else []),
                JURISDICTION_PROMPTS.get("india", ""),
            )
            india_result = {
                "answer"    : india_answer,
                "citations" : _build_citations(india_chunks),
                "confidence": india_ret["confidence"],
                "sufficient": india_ret["sufficient"],
            }

        if jurisdiction in (Jurisdiction.INTERNATIONAL, Jurisdiction.BOTH):
            intl_ret = self._jur_ret.retrieve_international(query, domain)
            intl_chunks = intl_ret["chunks"]
            intl_answer = _generate_answer(
                self._llm, query, intl_chunks,
                get_domain_prompt([domain] if domain else []),
                JURISDICTION_PROMPTS.get("international", ""),
            )
            intl_result = {
                "answer"    : intl_answer,
                "citations" : _build_citations(intl_chunks),
                "confidence": intl_ret["confidence"],
                "sufficient": intl_ret["sufficient"],
            }

        # 5. Comparison paragraph for "both"
        if jurisdiction == Jurisdiction.BOTH and india_result and intl_result:
            comparison = self._generate_comparison(
                query,
                india_result["answer"],
                intl_result["answer"],
            )

        return {
            "query"        : query,
            "ip_types"     : [t.value for t in route.ip_types],
            "primary_ip"   : route.primary.value,
            "jurisdiction" : jurisdiction.value,
            "jur_confidence": jur_route.confidence,
            "jur_reason"   : jur_route.reason,
            "formulation"  : formulation.model_dump() if formulation else None,
            "india"        : india_result,
            "international": intl_result,
            "comparison"   : comparison,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _generate_comparison(
        self,
        query: str,
        india_answer: str,
        intl_answer: str,
    ) -> str:
        user_msg = (
            f"Original question: {query}\n\n"
            f"INDIA ANSWER:\n{india_answer}\n\n"
            f"INTERNATIONAL ANSWER:\n{intl_answer}\n\n"
            f"Write the comparison summary:"
        )
        try:
            return self._llm.invoke([
                {"role": "system", "content": _COMPARISON_SYSTEM},
                {"role": "user",   "content": user_msg},
            ]).content.strip()
        except Exception:
            return "Comparison could not be generated."

    @staticmethod
    def _insufficient(
        query: str,
        route: QueryRoute,
        formulation: Optional[FormulationClassification],
    ) -> dict:
        return {
            "query"        : query,
            "ip_types"     : [t.value for t in route.ip_types],
            "primary_ip"   : route.primary.value,
            "router_reason": route.reason,
            "formulation"  : formulation.model_dump() if formulation else None,
            "answer"       : "I could not find sufficient information in the provided authoritative documents.",
            "confidence"   : "low",
            "sufficient"   : False,
            "citations"    : [],
            "domains_used" : [],
            "query_type"   : "unknown",
        }
