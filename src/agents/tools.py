"""
Agent Tools  (Phase 10)
────────────────────────
Six LangChain @tool wrappers over the existing pipeline modules.

Each tool:
  - Has a clear docstring (the LLM uses this to decide when to call it)
  - Returns a serialisable dict (not internal objects)
  - Records success/failure in its result
  - Never invents evidence

Tools:
  1. legal_search          — Indian legal corpus (acts, rules)
  2. patent_search         — Patent / prior-art corpus
  3. tk_search             — TK / AYUSH corpus
  4. formulation_classifier — Ingredient extraction + classification
  5. abs_check             — ABS relevance assessment
  6. international_search  — International treaties / WIPO material
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# LangChain tool decorator
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singletons (avoid loading heavy models on import)
# ─────────────────────────────────────────────────────────────────────────────

_retriever = None
_tk_matcher = None
_pat_searcher = None
_form_classifier = None
_form_extractor  = None
_abs_checker     = None
_embeddings      = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def _get_retriever():
    global _retriever
    if _retriever is None:
        from retrieval.retriever import HybridRetriever
        _retriever = HybridRetriever(embeddings=_get_embeddings())
    return _retriever


def _get_tk_matcher():
    global _tk_matcher
    if _tk_matcher is None:
        from tk.tk_matcher import TKMatcher
        _tk_matcher = TKMatcher(embeddings=_get_embeddings(), top_k=5)
    return _tk_matcher


def _get_pat_searcher():
    global _pat_searcher
    if _pat_searcher is None:
        from patents.patent_search import PatentSearcher
        _pat_searcher = PatentSearcher(
            embeddings=_get_embeddings(), top_k_retrieval=10, top_k_final=5
        )
    return _pat_searcher


def _get_form_classifier():
    global _form_classifier
    if _form_classifier is None:
        from formulation.classifier import FormulationClassifier
        _form_classifier = FormulationClassifier()
    return _form_classifier


def _get_form_extractor():
    global _form_extractor
    if _form_extractor is None:
        from formulation.extractor import FormulationExtractor
        _form_extractor = FormulationExtractor()
    return _form_extractor


def _get_abs_checker():
    global _abs_checker
    if _abs_checker is None:
        from abs.abs_checker import ABSChecker
        _abs_checker = ABSChecker()
    return _abs_checker


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Legal search (Indian)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def legal_search(query: str, domain: str = "patent") -> dict:
    """
    Search authoritative Indian IP laws and rules.

    Use for: Patents Act sections, Trade Marks Act, Copyright Act,
    Designs Act, GI Act, Biodiversity Act, AYUSH guidelines.
    Always use for any question about Indian law or legal provisions.

    Parameters
    ----------
    query  : natural-language search query
    domain : "patent" | "trademark" | "copyright" | "design" | "gi" | "ayush" | "general"
    """
    try:
        retriever = _get_retriever()
        result    = retriever.retrieve(query, domain_filter=domain or "patent")
        chunks    = result.get("chunks", [])

        from evidence.store import from_legal_chunk
        items = [from_legal_chunk(c, 0.7) for c in chunks]

        return {
            "status"   : "success",
            "tool"     : "legal_search",
            "count"    : len(items),
            "evidence" : [i.to_dict() for i in items[:5]],
            "confidence": result.get("confidence", "low"),
        }
    except Exception as e:
        return {"status": "failed", "tool": "legal_search", "error": str(e), "evidence": []}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Patent search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def patent_search(query: str) -> dict:
    """
    Search patent and prior-art records.

    Use for: finding similar patents, prior-art search, checking whether
    an invention already exists in the patent corpus.
    Always use when patentability or novelty is in question.

    Parameters
    ----------
    query : natural-language description of the invention or technology
    """
    try:
        from ingestion.schema import Invention

        inv = Invention(
            title=query[:80],
            components=[],
            keywords=query.lower().split()[:8],
            intended_use=query[:100],
        )
        results = _get_pat_searcher().search(inv, [query])

        from evidence.store import from_patent_chunk
        items = []
        for r in results[:5]:
            chunk = r.get("chunk")
            if chunk:
                item = from_patent_chunk(chunk, r.get("similarity_score", 0.0))
                items.append(item)

        return {
            "status"  : "success",
            "tool"    : "patent_search",
            "count"   : len(items),
            "evidence": [i.to_dict() for i in items],
        }
    except Exception as e:
        return {"status": "failed", "tool": "patent_search", "error": str(e), "evidence": []}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — TK search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def tk_search(query: str) -> dict:
    """
    Search traditional knowledge and AYUSH sources (TKDL, Ayurvedic Formulary, etc.).

    Use when: the query involves herbs, traditional medicine, Ayurveda, Siddha,
    Unani, traditional formulations, or TKDL prior art.

    Parameters
    ----------
    query : description of formulation or ingredients
    """
    try:
        from ingestion.schema import Invention

        inv = Invention(
            title=query[:80],
            components=query.lower().split()[:6],
            keywords=query.lower().split()[:8],
        )
        result = _get_tk_matcher().match(inv)
        matches = result.get("matches", [])

        from evidence.store import from_tk_match
        items = [from_tk_match(m) for m in matches[:5]]

        return {
            "status"   : "success",
            "tool"     : "tk_search",
            "count"    : len(items),
            "tk_match" : result.get("traditional_knowledge_match", False),
            "evidence" : [i.to_dict() for i in items],
        }
    except Exception as e:
        return {"status": "failed", "tool": "tk_search", "error": str(e), "evidence": []}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Formulation classifier
# ─────────────────────────────────────────────────────────────────────────────

@tool
def formulation_classifier(formulation: str) -> dict:
    """
    Classify a formulation description and identify ingredients.

    Returns: formulation type (ayurveda/siddha/unani/modern_pharmaceutical/…),
    ingredient list with scientific names, biological-resource flags,
    traditional-use indicators.

    Use when: the user describes a formulation, composition, or mixture.

    Parameters
    ----------
    formulation : free-text description of the formulation
    """
    try:
        classification = _get_form_classifier().classify(formulation)
        ingredients    = _get_form_extractor().extract(formulation)

        return {
            "status"             : "success",
            "tool"               : "formulation_classifier",
            "formulation_type"   : classification.formulation_type.value,
            "ingredients"        : classification.ingredients,
            "biological_resources": classification.biological_resources,
            "tk_indicators"      : classification.traditional_knowledge_indicators,
            "tk_systems"         : classification.tk_systems,
            "confidence"         : classification.confidence,
            "is_tk_relevant"     : classification.is_tk_relevant(),
            "enriched_ingredients": [
                {
                    "name"             : i.name,
                    "scientific_name"  : i.scientific_name,
                    "biological_resource": i.biological_resource,
                    "traditional_use"  : i.traditional_use_indicator,
                    "systems"          : i.traditional_systems,
                }
                for i in ingredients
            ],
        }
    except Exception as e:
        return {"status": "failed", "tool": "formulation_classifier", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — ABS check
# ─────────────────────────────────────────────────────────────────────────────

@tool
def abs_check(formulation: str) -> dict:
    """
    Check whether a formulation has potential ABS (Access and Benefit Sharing)
    relevance under the Biological Diversity Act, 2002 and Nagoya Protocol.

    Use when: the query involves biological resources, plant extracts, or
    traditional formulations. Always use alongside formulation_classifier.

    Returns: potentially_relevant flag, biological resources identified,
    TK detected flag, relevant legal provisions.

    Parameters
    ----------
    formulation : free-text description of the formulation
    """
    try:
        classification = _get_form_classifier().classify(formulation)
        ingredients    = _get_form_extractor().extract(formulation)
        classification.ingredient_objects = ingredients

        from abs.abs_checker import assess_abs
        result = assess_abs(classification)

        from evidence.store import EvidenceItem, EvidenceType, Authority
        items = []
        for provision in result.suggested_provisions[:3]:
            items.append(EvidenceItem(
                source_id  = provision[:30].replace(" ", "_").lower(),
                source_type= EvidenceType.ABS,
                jurisdiction= "india",
                domain     = "general",
                text       = provision,
                authority  = Authority.PRIMARY,
                title      = provision,
                source_name= "Biological Diversity Act / Nagoya Protocol",
            ))

        return {
            "status"                     : "success",
            "tool"                       : "abs_check",
            "potentially_relevant"       : result.potentially_relevant,
            "biological_resources"       : result.biological_resources,
            "traditional_knowledge_detected": result.traditional_knowledge_detected,
            "reasons"                    : result.reasons,
            "suggested_provisions"       : result.suggested_provisions,
            "requires_human_review"      : result.requires_human_review,
            "evidence"                   : [i.to_dict() for i in items],
        }
    except Exception as e:
        return {"status": "failed", "tool": "abs_check", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6 — International search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def international_search(query: str) -> dict:
    """
    Search international IP treaties and WIPO/PCT reference material.

    Use when: the query asks about PCT, TRIPS, Paris Convention, Berne
    Convention, Madrid System, WIPO, or any international IP framework.
    Use ONLY for international jurisdiction queries — not for Indian law.

    Parameters
    ----------
    query : natural-language query about international IP
    """
    try:
        retriever = _get_retriever()
        # Search without domain filter — international docs use jurisdiction="international"
        result = retriever.retrieve(query)
        chunks = [
            c for c in result.get("chunks", [])
            if getattr(c, "jurisdiction", "india") == "international"
        ]
        if not chunks:
            # Fall back: search general domain
            result = retriever.retrieve(query, domain_filter="general")
            chunks = result.get("chunks", [])

        from evidence.store import EvidenceItem, EvidenceType, Authority
        items = []
        for chunk in chunks[:5]:
            items.append(EvidenceItem(
                source_id   = getattr(chunk, "chunk_id", "intl_unknown"),
                source_type = EvidenceType.INTERNATIONAL,
                jurisdiction= "international",
                domain      = getattr(chunk, "domain", "general"),
                text        = getattr(chunk, "text", ""),
                section     = getattr(chunk, "section", None),
                page        = getattr(chunk, "page", None),
                authority   = getattr(chunk, "authority_level", Authority.SECONDARY),
                chunk_id    = getattr(chunk, "chunk_id", ""),
                title       = getattr(chunk, "title", ""),
                source_name = getattr(chunk, "source", "WIPO"),
            ))

        return {
            "status"  : "success",
            "tool"    : "international_search",
            "count"   : len(items),
            "evidence": [i.to_dict() for i in items],
        }
    except Exception as e:
        return {"status": "failed", "tool": "international_search",
                "error": str(e), "evidence": []}


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    legal_search,
    patent_search,
    tk_search,
    formulation_classifier,
    abs_check,
    international_search,
]

TOOL_NAMES = [t.name for t in ALL_TOOLS]
