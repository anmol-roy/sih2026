"""
Report Generator
─────────────────
Takes the fused evidence package and produces a structured
Preliminary Assessment report via the LLM.

Output
------
{
  "invention_summary"  : str,
  "relevant_provisions": list[str],
  "similar_patents"    : list[dict],
  "tk_matches"         : list[dict],
  "issues"             : list[str],
  "assessment"         : str,
  "confidence"         : str,
  "citations"          : list[dict],
}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention
from analysis.evidence_fusion import EvidenceItem


# ---------------------------------------------------------------------------
# System prompt — strictly evidence-based reasoning
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are IP-SAKTI Sahayak, an Indian intellectual-property information assistant.

Your role is to produce a PRELIMINARY ASSESSMENT of a submitted invention
description based SOLELY on the retrieved evidence provided below.

STRICT RULES:
1. Never provide a definitive legal opinion or conclusion.
2. Use ONLY the retrieved evidence. Do not invent sources, sections, or cases.
3. Clearly separate: Facts | Legal Provisions | Prior Art | TK Evidence | Inference.
4. Every substantive claim you make must reference a specific evidence item.
5. If evidence is insufficient for a point, say "Insufficient evidence to assess."
6. Do NOT use language like "your patent will be rejected" or "you cannot patent this."
   Use language like "the retrieved evidence suggests potential overlap" or
   "Section 3(p) may be relevant based on the retrieved traditional-knowledge material."
7. Be concise, structured, and precise.

Return a single coherent assessment in plain text with the following sections:

## Invention Summary
## Relevant Legal Provisions
## Similar Prior Art
## Traditional Knowledge Evidence
## Potential Issues
## Preliminary Assessment
## Sources
"""


# ---------------------------------------------------------------------------
# Evidence block builder
# ---------------------------------------------------------------------------

def _format_evidence_block(items: list[EvidenceItem], heading: str) -> str:
    if not items:
        return f"### {heading}\nNo relevant evidence retrieved.\n"
    lines = [f"### {heading}"]
    for i, item in enumerate(items, 1):
        lines.append(f"\n[{i}] {item.label}  ({item.type})")
        if item.section:
            lines.append(f"    Section: {item.section}")
        if item.page:
            lines.append(f"    Page: {item.page}")
        lines.append(f"    Source: {item.source}")
        # Show first 400 chars of text as context
        snippet = item.text[:400].replace("\n", " ").strip()
        if len(item.text) > 400:
            snippet += "…"
        lines.append(f"    Excerpt: {snippet}")
    return "\n".join(lines) + "\n"


def _build_user_prompt(evidence_package: dict) -> str:
    invention: Invention        = evidence_package["invention"]
    legal   : list[EvidenceItem] = evidence_package["legal_evidence"]
    patents : list[EvidenceItem] = evidence_package["patent_evidence"]
    tk      : list[EvidenceItem] = evidence_package["tk_evidence"]
    issues  : list[str]          = evidence_package["issues"]

    inv_block = (
        f"Title           : {invention.title}\n"
        f"Technical Field : {invention.technical_field or 'Not specified'}\n"
        f"Problem         : {invention.problem or 'Not specified'}\n"
        f"Solution        : {invention.solution or 'Not specified'}\n"
        f"Components      : {', '.join(invention.components) or 'None identified'}\n"
        f"Intended Use    : {invention.intended_use or 'Not specified'}\n"
        f"Novelty Claim   : {invention.novelty_claim or 'Not specified'}\n"
    )

    issues_block = (
        "\n".join(f"- {i}" for i in issues)
        if issues else "None flagged automatically."
    )

    legal_block  = _format_evidence_block(legal[:5],   "Legal Evidence")
    patent_block = _format_evidence_block(patents[:5],  "Similar Patent Records")
    tk_block     = _format_evidence_block(tk[:5],       "Traditional Knowledge Evidence")

    return f"""
INVENTION
─────────
{inv_block}

AUTOMATICALLY FLAGGED ISSUES
─────────────────────────────
{issues_block}

RETRIEVED EVIDENCE
──────────────────
{legal_block}
{patent_block}
{tk_block}

---
Using only the evidence above, write the Preliminary Assessment.
"""


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    def __init__(self, llm: Optional[ChatGroq] = None):
        self._llm = llm or ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

    def generate(self, evidence_package: dict) -> dict:
        """
        Parameters
        ----------
        evidence_package : output of evidence_fusion.fuse_evidence()

        Returns
        -------
        Structured report dict
        """
        invention : Invention        = evidence_package["invention"]
        legal     : list[EvidenceItem] = evidence_package["legal_evidence"]
        patents   : list[EvidenceItem] = evidence_package["patent_evidence"]
        tk        : list[EvidenceItem] = evidence_package["tk_evidence"]
        issues    : list[str]          = evidence_package["issues"]
        confidence: str                = evidence_package["overall_confidence"]

        user_prompt = _build_user_prompt(evidence_package)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        response  = self._llm.invoke(messages)
        full_text = response.content.strip()

        # ── Build structured citations ─────────────────────────────────────
        citations = []
        seen: set[str] = set()

        for item in legal + patents + tk:
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            citations.append({
                "type"    : item.type,
                "label"   : item.label,
                "source"  : item.source,
                "section" : item.section,
                "page"    : item.page,
                "chunk_id": item.chunk_id,
            })

        # ── Build similar_patents summary ──────────────────────────────────
        similar_patents = []
        seen_pubs: set[str] = set()
        for item in patents:
            pub = item.chunk_id.split("_claim")[0].split("_abstract")[0]
            if pub in seen_pubs:
                continue
            seen_pubs.add(pub)
            similar_patents.append({
                "label"           : item.label,
                "similarity_score": item.score,
                "source"          : item.source,
                "chunk_id"        : item.chunk_id,
            })

        # ── TK match summary ───────────────────────────────────────────────
        tk_matches = [
            {
                "label"  : item.label,
                "source" : item.source,
                "page"   : item.page,
                "score"  : item.score,
            }
            for item in tk
        ]

        return {
            "invention_summary"  : (
                f"{invention.title}. "
                f"Components: {', '.join(invention.components) or 'N/A'}. "
                f"Use: {invention.intended_use or 'N/A'}."
            ),
            "relevant_provisions": [
                item.label for item in legal
                if item.type == "PRIMARY_LAW"
            ],
            "similar_patents"    : similar_patents,
            "tk_matches"         : tk_matches,
            "issues"             : issues,
            "assessment"         : full_text,
            "confidence"         : confidence,
            "citations"          : citations,
        }
