"""
Formulation Analysis Pipeline  (Phase 7)
──────────────────────────────────────────
Single entry point that orchestrates:

  description (free text)
       ↓
  FormulationExtractor  → list[Ingredient]
       ↓
  FormulationClassifier → FormulationClassification (enriched)
       ↓
  ABSChecker            → ABSAssessment
       ↓
  ┌────────────────────────────────────────┐
  │  (parallel — only when relevant)       │
  │  TK Matcher    → TK evidence           │
  │  Patent Search → prior-art evidence    │
  │  Legal Retriever → ABS / biodiversity  │
  └────────────────────────────────────────┘
       ↓
  Evidence Fusion → FormulationAnalysisResult
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from formulation.schemas import (
    FormulationClassification, FormulationType, ABSAssessment, Ingredient
)
from formulation.extractor import FormulationExtractor
from formulation.classifier import FormulationClassifier
from abs.abs_checker import ABSChecker


# ─────────────────────────────────────────────────────────────────────────────
# LLM report prompt
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_SYSTEM = """\
You are IP-SAKTI Sahayak, an Indian IP and biodiversity compliance information assistant.

You are given a structured analysis of a formulation including:
  - Classification (Ayurveda / Siddha / Modern Pharmaceutical / etc.)
  - Biological resources identified
  - Traditional knowledge indicators
  - ABS relevance assessment
  - Evidence from TK/AYUSH corpus, prior-art patents, and legal provisions

Produce a PRELIMINARY GUIDANCE document structured as follows:

## Formulation Summary
## Classification
## Biological Resources
## Traditional Knowledge Indicators
## ABS Relevance Assessment
## TK / TKDL Pointer
## Prior Art Pointer
## Relevant Legal Provisions
## Preliminary Guidance
## Important Disclaimer

STRICT RULES:
1. Every factual or legal claim must cite a specific evidence item.
2. Never say "you need ABS approval" — say "potential ABS relevance detected, further assessment required."
3. Never say "this is not patentable" — say "the retrieved evidence suggests potential overlap with traditional knowledge."
4. If evidence is insufficient for any section, write "Insufficient evidence retrieved for this section."
5. Always include the disclaimer: "This is a preliminary information analysis only. It does not constitute legal advice or regulatory clearance."
6. Be concise and precise.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

class FormulationAnalysisResult:
    """Aggregated result of the full Phase 7 pipeline."""

    def __init__(
        self,
        classification   : FormulationClassification,
        abs_assessment   : ABSAssessment,
        tk_results       : dict,
        patent_results   : list[dict],
        legal_chunks     : list,
        report           : str,
        confidence       : str,
    ):
        self.classification  = classification
        self.abs_assessment  = abs_assessment
        self.tk_results      = tk_results
        self.patent_results  = patent_results
        self.legal_chunks    = legal_chunks
        self.report          = report
        self.confidence      = confidence

    def to_dict(self) -> dict:
        return {
            "classification" : self.classification.model_dump(),
            "abs_assessment" : self.abs_assessment.model_dump(),
            "tk_results"     : self.tk_results,
            "patent_results" : [
                {
                    "publication_number": r.get("publication_number", ""),
                    "title"             : r.get("title", ""),
                    "similarity_score"  : r.get("similarity_score", 0.0),
                    "matched_components": r.get("matched_components", []),
                    "source"            : r.get("source", ""),
                }
                for r in self.patent_results
            ],
            "legal_provisions": [
                {
                    "document"  : getattr(c, "title", ""),
                    "section"   : getattr(c, "section", ""),
                    "subsection": getattr(c, "subsection", ""),
                    "source"    : getattr(c, "source", ""),
                    "page"      : getattr(c, "page", None),
                    "chunk_id"  : getattr(c, "chunk_id", ""),
                }
                for c in self.legal_chunks
            ],
            "report"    : self.report,
            "confidence": self.confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FormulationAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class FormulationAnalyzer:
    """
    Orchestrates the full Phase 7 formulation analysis pipeline.

    Parameters
    ----------
    embeddings : shared HuggingFaceEmbeddings
    llm        : shared ChatMistralAI
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        llm       : ChatMistralAI,
    ):
        self._llm        = llm
        self._extractor  = FormulationExtractor(llm=llm)
        self._classifier = FormulationClassifier(llm=llm)
        self._abs        = ABSChecker()
        self._emb        = embeddings

        # Lazy imports to avoid circular deps
        self._tk_matcher    = None
        self._pat_searcher  = None
        self._retriever     = None

    def _get_tk_matcher(self):
        if self._tk_matcher is None:
            from tk.tk_matcher import TKMatcher
            self._tk_matcher = TKMatcher(embeddings=self._emb, top_k=5)
        return self._tk_matcher

    def _get_pat_searcher(self):
        if self._pat_searcher is None:
            from patents.patent_search import PatentSearcher
            self._pat_searcher = PatentSearcher(
                embeddings=self._emb, top_k_retrieval=10, top_k_final=5
            )
        return self._pat_searcher

    def _get_retriever(self):
        if self._retriever is None:
            from retrieval.retriever import HybridRetriever
            self._retriever = HybridRetriever(embeddings=self._emb)
        return self._retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, description: str) -> FormulationAnalysisResult:
        """
        Full Phase 7 pipeline.

        Returns a FormulationAnalysisResult with all evidence assembled.
        """
        # ── 1. Extract ingredients ────────────────────────────────────────
        ingredients = self._extractor.extract(description)

        # ── 2. Classify formulation (enriched) ───────────────────────────
        classification = self._classifier.classify(description)
        # Attach enriched ingredient objects
        classification.ingredient_objects = ingredients

        # Sync string lists from enriched objects
        if ingredients:
            bio_names = [i.name for i in ingredients if i.biological_resource]
            if bio_names:
                # Merge with any already extracted
                existing = set(classification.biological_resources)
                for n in bio_names:
                    if n not in existing:
                        classification.biological_resources.append(n)
                        existing.add(n)

        # ── 3. ABS assessment ─────────────────────────────────────────────
        abs_assessment = self._abs.check(classification)

        # ── 4. Build search representation ───────────────────────────────
        from ingestion.schema import Invention
        inv = Invention(
            title         = f"Formulation: {description[:60]}",
            technical_field= "Herbal / Pharmaceutical formulation",
            components    = classification.ingredients[:10],
            intended_use  = description[:100],
            keywords      = classification.ingredients[:6] + classification.tk_systems,
        )

        # ── 5. TK search (only if TK-relevant) ───────────────────────────
        tk_results: dict = {"traditional_knowledge_match": False, "matches": []}
        if classification.is_tk_relevant() or abs_assessment.traditional_knowledge_detected:
            tk_results = self._get_tk_matcher().match(inv)

        # ── 6. Patent / prior-art search ──────────────────────────────────
        from analysis.query_generator import generate_queries
        queries = generate_queries(inv)
        patent_results: list[dict] = []
        if classification.formulation_type != FormulationType.MODERN_PHARMACEUTICAL or \
           classification.is_tk_relevant():
            patent_results = self._get_pat_searcher().search(inv, queries)

        # ── 7. Legal retrieval ────────────────────────────────────────────
        legal_chunks = []
        retriever = self._get_retriever()
        legal_map: dict[str, object] = {}

        # Always retrieve patent-domain legal provisions
        for q in queries[:4]:
            for chunk in retriever.retrieve(q, domain_filter="patent")["chunks"]:
                if chunk.chunk_id not in legal_map:
                    legal_map[chunk.chunk_id] = chunk

        # If ABS is potentially relevant, also retrieve biodiversity domain
        if abs_assessment.potentially_relevant:
            for q in [
                "Biological Diversity Act biological resources access",
                "ABS Nagoya Protocol biodiversity India",
                "traditional knowledge biological resources patent India",
            ]:
                for chunk in retriever.retrieve(q, domain_filter="general")["chunks"]:
                    if chunk.chunk_id not in legal_map:
                        legal_map[chunk.chunk_id] = chunk

        legal_chunks = list(legal_map.values())

        # ── 8. Generate report ────────────────────────────────────────────
        report = self._generate_report(
            description, classification, abs_assessment,
            tk_results, patent_results, legal_chunks,
        )

        # ── 9. Overall confidence ─────────────────────────────────────────
        conf_scores = []
        if tk_results.get("matches"):
            conf_scores.extend([m["score"] for m in tk_results["matches"] if m.get("score")])
        if patent_results:
            conf_scores.extend([r.get("similarity_score", 0) for r in patent_results[:3]])
        if legal_chunks:
            conf_scores.append(0.6)

        if not conf_scores:
            overall_conf = "low"
        elif sum(conf_scores) / len(conf_scores) >= 0.65:
            overall_conf = "high"
        elif sum(conf_scores) / len(conf_scores) >= 0.40:
            overall_conf = "medium"
        else:
            overall_conf = "low"

        return FormulationAnalysisResult(
            classification = classification,
            abs_assessment = abs_assessment,
            tk_results     = {
                "match_found": tk_results.get("traditional_knowledge_match", False),
                "matches": [
                    {
                        "source"            : m["chunk"].source if hasattr(m.get("chunk"), "source") else "",
                        "title"             : m["chunk"].title  if hasattr(m.get("chunk"), "title")  else "",
                        "matched_components": m.get("matched_components", []),
                        "score"             : m.get("score", 0.0),
                        "page"              : m.get("page"),
                    }
                    for m in tk_results.get("matches", [])
                ],
            },
            patent_results = patent_results,
            legal_chunks   = legal_chunks,
            report         = report,
            confidence     = overall_conf,
        )

    # ------------------------------------------------------------------
    # Report generator
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        description    : str,
        classification : FormulationClassification,
        abs_assessment : ABSAssessment,
        tk_results     : dict,
        patent_results : list[dict],
        legal_chunks   : list,
    ) -> str:
        """Build the LLM prompt context and generate the guidance report."""

        # Build evidence block
        lines = [f"FORMULATION DESCRIPTION:\n{description}\n"]

        lines.append(f"CLASSIFICATION:\n  Type: {classification.formulation_type.value}")
        if classification.secondary_types:
            lines.append(f"  Secondary: {[t.value for t in classification.secondary_types]}")
        lines.append(f"  Confidence: {classification.confidence:.0%}")
        lines.append(f"  Ingredients: {', '.join(classification.ingredients) or 'none identified'}")
        lines.append(f"  Biological resources: {', '.join(classification.all_bio_resource_names()) or 'none'}")
        lines.append(f"  TK indicators: {', '.join(classification.traditional_knowledge_indicators) or 'none'}")

        if classification.ingredient_objects:
            lines.append("\nENRICHED INGREDIENTS:")
            for ing in classification.ingredient_objects:
                sci = f" ({ing.scientific_name})" if ing.scientific_name else ""
                flags = []
                if ing.biological_resource:      flags.append("biological resource")
                if ing.traditional_use_indicator: flags.append("TK indicator")
                lines.append(f"  • {ing.name}{sci} — {', '.join(flags) or 'no flags'}")

        lines.append(f"\nABS ASSESSMENT:")
        lines.append(f"  Potentially relevant: {abs_assessment.potentially_relevant}")
        lines.append(f"  TK detected: {abs_assessment.traditional_knowledge_detected}")
        for r in abs_assessment.reasons:
            lines.append(f"  Reason: {r}")
        for p in abs_assessment.suggested_provisions:
            lines.append(f"  Provision: {p}")

        if tk_results.get("matches"):
            lines.append("\nTK / TKDL EVIDENCE:")
            for m in tk_results["matches"][:5]:
                chunk = m.get("chunk") if hasattr(m.get("chunk"), "title") else None
                if chunk:
                    lines.append(
                        f"  • {chunk.title}"
                        f" | p.{chunk.page or '?'}"
                        f" | score={m.get('score', 0):.2f}"
                        f" | matched={m.get('matched_components', [])}"
                    )

        if patent_results:
            lines.append("\nPRIOR-ART EVIDENCE:")
            for r in patent_results[:5]:
                chunk = r.get("chunk")
                pub   = r.get("publication_number") or (chunk.publication_number if chunk else "?")
                title = r.get("title") or (chunk.title if chunk else "?")
                sim   = r.get("similarity_score", 0)
                matched = r.get("matched_components", [])
                lines.append(f"  • {pub} — {title[:50]} | sim={sim:.2f} | matched={matched}")

        if legal_chunks:
            lines.append("\nLEGAL EVIDENCE:")
            for chunk in legal_chunks[:6]:
                lines.append(
                    f"  • {chunk.title}"
                    + (f" | {chunk.section}" if chunk.section else "")
                    + (f" § {chunk.subsection}" if chunk.subsection else "")
                    + f" | {chunk.source}"
                )

        context = "\n".join(lines)

        messages = [
            {"role": "system", "content": _REPORT_SYSTEM},
            {"role": "user",   "content": f"{context}\n\n---\nWrite the Preliminary Guidance:"},
        ]
        try:
            return self._llm.invoke(messages).content.strip()
        except Exception as e:
            return f"Report generation failed: {e}\n\nEvidence collected:\n{context}"
