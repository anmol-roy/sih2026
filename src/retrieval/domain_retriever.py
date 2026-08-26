"""
Domain Retriever  (Phase 5)
────────────────────────────
Wraps the Phase 2 HybridRetriever and adds domain-aware multi-retrieval.

For single-domain queries:
    retrieve_for_domain(query, "patent") → chunks from patent domain

For multi-domain queries:
    retrieve_multi_domain(query, ["patent", "trademark"]) → merged chunks

Also exposes domain-specific system prompts for the LLM generation step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from retrieval.retriever import HybridRetriever
from ingestion.schema import LegalChunk


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific system prompts
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_PROMPTS: dict[str, str] = {

    "patent": """\
You are an Indian patent law information assistant.
Focus on: patentability, novelty, inventive step, excluded subject matter
(Sections 3 and 4 of the Patents Act 1970), prior art, patent application
procedure, and patent term.
Answer ONLY from the provided excerpts. Cite Act, Section, and Subsection.
Do not invent legal provisions. If evidence is insufficient, say so.""",

    "trademark": """\
You are an Indian trademark law information assistant.
Focus on: registrability of marks, distinctiveness, well-known marks,
absolute and relative grounds for refusal, trademark infringement, passing off,
trademark registration procedure, and classification of goods/services.
Answer ONLY from the provided excerpts. Cite the Trade Marks Act 1999 sections.""",

    "copyright": """\
You are an Indian copyright law information assistant.
Focus on: copyrightable works, authorship and ownership, duration of copyright,
moral rights, economic rights, copyright infringement, fair dealing, and
licensing. Answer ONLY from the provided excerpts.
Cite the Copyright Act 1957 sections.""",

    "design": """\
You are an Indian design law information assistant.
Focus on: registrability of designs, novelty and originality requirements,
design registration procedure, duration, and infringement.
Answer ONLY from the provided excerpts.
Cite the Designs Act 2000 sections.""",

    "gi": """\
You are an Indian Geographical Indication law information assistant.
Focus on: GI registration eligibility, definition of geographical indications,
producer/applicant requirements, GI protection scope, and enforcement.
Answer ONLY from the provided excerpts.
Cite the Geographical Indications of Goods Act 1999 sections.""",

    "ayush": """\
You are an Indian traditional knowledge and AYUSH information assistant.
Focus on: TKDL, AYUSH systems (Ayurveda, Siddha, Unani, Yoga), Section 3(p)
of the Patents Act (TK exclusion), Access and Benefit Sharing (ABS),
traditional formulations, and prior art based on traditional knowledge.
Answer ONLY from the provided excerpts. Cite sources with page numbers.""",

    "traditional_knowledge": """\
You are an Indian traditional knowledge and AYUSH information assistant.
Focus on: TKDL, AYUSH systems (Ayurveda, Siddha, Unani, Yoga), Section 3(p)
of the Patents Act (TK exclusion), Access and Benefit Sharing (ABS),
traditional formulations, and prior art based on traditional knowledge.
Answer ONLY from the provided excerpts. Cite sources with page numbers.""",

    "general": """\
You are an Indian intellectual property law information assistant.
Answer the user's question using only the provided document excerpts.
Cite all relevant legal provisions (Act name, Section, Subsection).
If the evidence is insufficient, say:
"I could not find sufficient information in the provided authoritative documents."
Do not invent legal provisions.""",
}

_MULTI_DOMAIN_SUFFIX = """

NOTE: This query spans multiple IP domains. Clearly address each relevant
domain separately in your answer. Structure your response by domain.
Do not mix information across domains without clearly labelling it."""


def get_domain_prompt(domains: list[str]) -> str:
    """Return the appropriate system prompt for the given domain(s)."""
    if not domains:
        return _DOMAIN_PROMPTS["general"]
    if len(domains) == 1:
        return _DOMAIN_PROMPTS.get(domains[0], _DOMAIN_PROMPTS["general"])
    # Multi-domain: pick the first known prompt and append multi-domain note
    base = _DOMAIN_PROMPTS.get(domains[0], _DOMAIN_PROMPTS["general"])
    return base + _MULTI_DOMAIN_SUFFIX


# ─────────────────────────────────────────────────────────────────────────────
# Domain Retriever
# ─────────────────────────────────────────────────────────────────────────────

class DomainRetriever:
    """
    Domain-aware retrieval layer wrapping HybridRetriever.

    Parameters
    ----------
    embeddings : shared HuggingFaceEmbeddings instance
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self._retriever = HybridRetriever(
            embeddings   = embeddings,
            bm25_top_k   = 10,
            vector_top_k = 10,
            final_top_k  = 5,
        )

    # ------------------------------------------------------------------
    # Single-domain retrieval
    # ------------------------------------------------------------------

    def retrieve_for_domain(
        self,
        query: str,
        domain: Optional[str],
    ) -> dict:
        """
        Retrieve chunks filtered to *domain*.
        Returns the same dict as HybridRetriever.retrieve().
        """
        return self._retriever.retrieve(query, domain_filter=domain)

    # ------------------------------------------------------------------
    # Multi-domain retrieval
    # ------------------------------------------------------------------

    def retrieve_multi_domain(
        self,
        query: str,
        domains: list[str],
        chunks_per_domain: int = 5,
    ) -> dict:
        """
        Retrieve chunks from each domain independently, then merge.

        Returns
        -------
        {
          "chunks"     : list[LegalChunk]  — deduplicated, sorted by score proxy
          "confidence" : str
          "sufficient" : bool
          "query_type" : str
          "domains_used": list[str]
        }
        """
        if not domains:
            return self._retriever.retrieve(query)

        seen_ids: set[str] = set()
        all_chunks: list[LegalChunk] = []
        any_sufficient = False
        best_confidence = "low"

        _conf_rank = {"high": 2, "medium": 1, "low": 0}

        for domain in domains:
            result = self._retriever.retrieve(query, domain_filter=domain)
            if result["sufficient"]:
                any_sufficient = True
            if _conf_rank.get(result["confidence"], 0) > _conf_rank.get(best_confidence, 0):
                best_confidence = result["confidence"]
            for chunk in result["chunks"][:chunks_per_domain]:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_chunks.append(chunk)

        return {
            "chunks"      : all_chunks,
            "confidence"  : best_confidence,
            "sufficient"  : any_sufficient or len(all_chunks) > 0,
            "query_type"  : "multi_domain",
            "domains_used": domains,
        }

    # ------------------------------------------------------------------
    # Convenience: retrieve by route object
    # ------------------------------------------------------------------

    def retrieve_by_route(self, query: str, route) -> dict:
        """
        Accept a QueryRoute object and dispatch to the right retrieval method.
        """
        domains = route.domain_filters  # list of Qdrant domain values
        # Remove None entries
        domains = [d for d in domains if d]

        if not domains:
            return self._retriever.retrieve(query)
        if len(domains) == 1:
            return self.retrieve_for_domain(query, domains[0])
        return self.retrieve_multi_domain(query, domains)
