"""
Report Generator  (Phase 3 + 4)
─────────────────────────────────
Two entry points:

  generate(evidence_package)
      Phase 3 — invention + legal/patent/TK evidence → preliminary report

  generate_patentability(pat4_package)
      Phase 4 — feature matrix + novelty + inventive-step → patentability report
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention, InventionFeature, ClaimRepresentation
from analysis.evidence_fusion import EvidenceItem


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        snippet = item.text[:400].replace("\n", " ").strip()
        if len(item.text) > 400:
            snippet += "…"
        lines.append(f"    Excerpt: {snippet}")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Preliminary assessment
# ─────────────────────────────────────────────────────────────────────────────

_P3_SYSTEM = """\
You are IP-SAKTI Sahayak, an Indian intellectual-property information assistant.

Produce a PRELIMINARY ASSESSMENT based SOLELY on the retrieved evidence.

STRICT RULES:
1. Never provide a definitive legal opinion or conclusion.
2. Use ONLY retrieved evidence — never invent sources, sections, or cases.
3. Separate: Facts | Legal Provisions | Prior Art | TK Evidence | Inference.
4. Every substantive claim must reference a specific evidence item.
5. If evidence is insufficient say "Insufficient evidence to assess."
6. Do NOT say "your patent will be rejected." Say "the retrieved evidence
   suggests potential overlap" or "Section 3(p) may be relevant."
7. Be concise, structured, and precise.

Write the assessment with these sections:
## Invention Summary
## Relevant Legal Provisions
## Similar Prior Art
## Traditional Knowledge Evidence
## Potential Issues
## Preliminary Assessment
## Sources
"""


def _p3_user_prompt(evidence_package: dict) -> str:
    invention: Invention         = evidence_package["invention"]
    legal    : list[EvidenceItem] = evidence_package["legal_evidence"]
    patents  : list[EvidenceItem] = evidence_package["patent_evidence"]
    tk       : list[EvidenceItem] = evidence_package["tk_evidence"]
    issues   : list[str]          = evidence_package["issues"]

    inv_block = (
        f"Title           : {invention.title}\n"
        f"Technical Field : {invention.technical_field or 'Not specified'}\n"
        f"Problem         : {invention.problem or 'Not specified'}\n"
        f"Solution        : {invention.solution or 'Not specified'}\n"
        f"Components      : {', '.join(invention.components) or 'None identified'}\n"
        f"Intended Use    : {invention.intended_use or 'Not specified'}\n"
        f"Novelty Claim   : {invention.novelty_claim or 'Not specified'}\n"
    )
    issues_block = "\n".join(f"- {i}" for i in issues) if issues else "None flagged."
    return (
        f"\nINVENTION\n─────────\n{inv_block}\n"
        f"FLAGGED ISSUES\n──────────────\n{issues_block}\n\n"
        f"RETRIEVED EVIDENCE\n──────────────────\n"
        f"{_format_evidence_block(legal[:5], 'Legal Evidence')}\n"
        f"{_format_evidence_block(patents[:5], 'Similar Patent Records')}\n"
        f"{_format_evidence_block(tk[:5], 'Traditional Knowledge Evidence')}\n"
        f"---\nUsing only the evidence above, write the Preliminary Assessment.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Patentability assessment
# ─────────────────────────────────────────────────────────────────────────────

_P4_SYSTEM = """\
You are IP-SAKTI Sahayak, an Indian intellectual-property analysis assistant.

Produce a PRELIMINARY PATENTABILITY ASSESSMENT based SOLELY on the
feature-level prior-art evidence provided below.

STRICT RULES:
1. Never provide a definitive legal opinion.
2. Separate each analytical section clearly.
3. Reference every factual claim with a specific document or feature ID.
4. Use language like "the retrieved evidence suggests…" or "warrants examination."
5. Do NOT say "patent will be granted/rejected."
6. If a feature has no matching prior art, state that explicitly.
7. Be precise, structured, and concise.

Write the report with these sections:

## Invention Summary
## Feature Analysis
   (For each feature F1…Fn: what prior art was found, if any)
## Novelty Assessment
   (Single-reference analysis — is there a document covering all features?)
## Inventive Step Assessment
   (Multi-reference analysis — do combinations cover all features?)
## Traditional Knowledge Relevance
## Relevant Legal Provisions
## Potential Issues
## Preliminary Conclusion
## Sources
"""


def _p4_user_prompt(pat4_package: dict) -> str:
    invention : Invention               = pat4_package["invention"]
    features  : list[InventionFeature]  = pat4_package["features"]
    claim     : ClaimRepresentation     = pat4_package["claim"]
    novelty   : dict                    = pat4_package["novelty_result"]
    inv_step  : dict                    = pat4_package["inventive_step_result"]
    legal_ev  : list[EvidenceItem]      = pat4_package["legal_evidence"]
    tk_ev     : list[EvidenceItem]      = pat4_package["tk_evidence"]
    candidates: list[dict]              = pat4_package["prior_art_summaries"]

    # ── Invention block ───────────────────────────────────────────────────
    feat_lines = "\n".join(
        f"  {f.id} [{f.category}]: {f.feature}" for f in features
    )
    inv_block = (
        f"Title           : {invention.title}\n"
        f"Technical Field : {invention.technical_field or 'N/A'}\n"
        f"Claim type      : {claim.claim_type}\n"
        f"Suggested IPC   : {', '.join(claim.ipc_suggested) or 'N/A'}\n"
        f"Independent claim: {claim.independent_claim}\n\n"
        f"Extracted features:\n{feat_lines}\n"
    )

    # ── Feature-match matrix ──────────────────────────────────────────────
    matrix_lines = []
    if candidates:
        # Header row
        doc_ids = [c["publication_number"][:12] for c in candidates[:5]]
        matrix_lines.append("Feature      | " + " | ".join(f"{d:12}" for d in doc_ids))
        matrix_lines.append("-" * (14 + 15 * len(doc_ids)))
        for feat in features:
            row = f"{feat.id} {feat.feature[:20]:20} | "
            for cand in candidates[:5]:
                fd = cand.get("feature_detail", {}).get(feat.id, {})
                symbol = "✓" if fd.get("matched") else "✗"
                row += f"{symbol:12} | "
            matrix_lines.append(row)
    matrix_block = "\n".join(matrix_lines) if matrix_lines else "No candidates retrieved."

    # ── Top prior-art candidates ──────────────────────────────────────────
    pa_lines = []
    for i, c in enumerate(candidates[:5], 1):
        pa_lines.append(
            f"[{i}] {c['publication_number']} — {c['title'][:60]}\n"
            f"     Coverage: {c['coverage']*100:.0f}%  "
            f"Matched: {', '.join(c['matched_features']) or 'none'}\n"
            f"     Date: {c.get('filing_date') or c.get('publication_date') or 'unknown'}"
        )
    pa_block = "\n".join(pa_lines) or "No prior art retrieved."

    # ── Novelty ───────────────────────────────────────────────────────────
    nov_block = (
        f"Assessment     : {novelty['assessment']}\n"
        f"Single-ref found: {novelty['single_reference_found']}\n"
        f"Explanation    : {novelty['explanation']}\n"
    )

    # ── Inventive step ────────────────────────────────────────────────────
    is_block = (
        f"Assessment      : {inv_step['assessment']}\n"
        f"Coverage ratio  : {inv_step.get('coverage_ratio', 0)*100:.0f}%\n"
        f"Uncovered feats : {', '.join(inv_step['uncovered_features']) or 'none'}\n"
        f"Explanation     : {inv_step['explanation']}\n"
    )

    return (
        f"\nINVENTION\n─────────\n{inv_block}\n"
        f"FEATURE-MATCH MATRIX\n────────────────────\n{matrix_block}\n\n"
        f"TOP PRIOR-ART CANDIDATES\n────────────────────────\n{pa_block}\n\n"
        f"NOVELTY EVIDENCE\n────────────────\n{nov_block}\n"
        f"INVENTIVE-STEP EVIDENCE\n───────────────────────\n{is_block}\n"
        f"{_format_evidence_block(legal_ev[:5], 'Legal Provisions')}\n"
        f"{_format_evidence_block(tk_ev[:5], 'Traditional Knowledge')}\n"
        f"---\nUsing only the evidence above, write the Preliminary Patentability Assessment.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ReportGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    def __init__(self, llm: Optional[ChatGroq] = None):
        self._llm = llm or ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

    # ── Phase 3 ───────────────────────────────────────────────────────────

    def generate(self, evidence_package: dict) -> dict:
        """Phase 3: preliminary invention assessment."""
        invention  = evidence_package["invention"]
        legal      = evidence_package["legal_evidence"]
        patents    = evidence_package["patent_evidence"]
        tk         = evidence_package["tk_evidence"]
        issues     = evidence_package["issues"]
        confidence = evidence_package["overall_confidence"]

        user_prompt = _p3_user_prompt(evidence_package)
        messages = [
            {"role": "system", "content": _P3_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        full_text = self._llm.invoke(messages).content.strip()

        citations = []
        seen: set[str] = set()
        for item in legal + patents + tk:
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            citations.append({
                "type": item.type, "label": item.label,
                "source": item.source, "section": item.section,
                "page": item.page, "chunk_id": item.chunk_id,
            })

        seen_pubs: set[str] = set()
        similar_patents = []
        for item in patents:
            pub = item.chunk_id.split("_claim")[0].split("_abstract")[0]
            if pub in seen_pubs:
                continue
            seen_pubs.add(pub)
            similar_patents.append({
                "label": item.label, "similarity_score": item.score,
                "source": item.source, "chunk_id": item.chunk_id,
            })

        return {
            "invention_summary": (
                f"{invention.title}. "
                f"Components: {', '.join(invention.components) or 'N/A'}. "
                f"Use: {invention.intended_use or 'N/A'}."
            ),
            "relevant_provisions": [i.label for i in legal if i.type == "PRIMARY_LAW"],
            "similar_patents":  similar_patents,
            "tk_matches":       [{"label": i.label, "source": i.source,
                                  "page": i.page, "score": i.score} for i in tk],
            "issues":           issues,
            "assessment":       full_text,
            "confidence":       confidence,
            "citations":        citations,
        }

    # ── Phase 4 ───────────────────────────────────────────────────────────

    def generate_patentability(self, pat4_package: dict) -> dict:
        """
        Phase 4: feature-level patentability report.

        pat4_package keys:
          invention, features, claim, novelty_result, inventive_step_result,
          legal_evidence, tk_evidence, prior_art_summaries, overall_confidence
        """
        invention  : Invention              = pat4_package["invention"]
        features   : list[InventionFeature] = pat4_package["features"]
        claim      : ClaimRepresentation    = pat4_package["claim"]
        novelty    : dict                   = pat4_package["novelty_result"]
        inv_step   : dict                   = pat4_package["inventive_step_result"]
        legal_ev   : list[EvidenceItem]     = pat4_package["legal_evidence"]
        tk_ev      : list[EvidenceItem]     = pat4_package["tk_evidence"]
        candidates : list[dict]             = pat4_package["prior_art_summaries"]
        confidence : str                    = pat4_package.get("overall_confidence", "low")

        user_prompt = _p4_user_prompt(pat4_package)
        messages = [
            {"role": "system", "content": _P4_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        full_text = self._llm.invoke(messages).content.strip()

        # ── Citations ─────────────────────────────────────────────────────
        citations = []
        seen: set[str] = set()
        for item in legal_ev + tk_ev:
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            citations.append({
                "type": item.type, "label": item.label,
                "source": item.source, "section": item.section,
                "page": item.page, "chunk_id": item.chunk_id,
            })
        for cand in candidates[:10]:
            cid = cand.get("chunk_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                citations.append({
                    "type": "PATENT_RECORD",
                    "label": f"{cand['publication_number']} — {cand['title']}",
                    "source": cand.get("source", "IP India"),
                    "section": None, "page": None,
                    "chunk_id": cid,
                })

        # ── Overall status ────────────────────────────────────────────────
        if novelty["assessment"] == "potential_novelty_issue":
            status = "requires_detailed_examination"
            reason = "A retrieved prior-art reference may cover all identified features."
        elif inv_step["assessment"] == "potential_inventive_step_concern":
            status = "requires_detailed_examination"
            reason = "Multiple prior-art references in combination may cover all features."
        elif novelty["assessment"] == "partial_prior_art":
            status = "further_search_recommended"
            reason = "Partial prior-art overlap found. Broader search is recommended."
        else:
            status = "insufficient_prior_art_found"
            reason = "The retrieved corpus does not contain strong prior-art evidence."

        return {
            "invention": {
                "title"         : invention.title,
                "technical_field": invention.technical_field,
                "features"      : [{"id": f.id, "feature": f.feature,
                                    "category": f.category} for f in features],
                "claim_type"    : claim.claim_type,
                "ipc_suggested" : claim.ipc_suggested,
                "independent_claim": claim.independent_claim,
            },
            "legal_basis": [
                {
                    "label"  : item.label,
                    "source" : item.source,
                    "section": item.section,
                    "page"   : item.page,
                }
                for item in legal_ev
                if item.type == "PRIMARY_LAW"
            ],
            "prior_art": candidates,
            "traditional_knowledge": [
                {
                    "label"  : item.label,
                    "source" : item.source,
                    "page"   : item.page,
                    "score"  : item.score,
                    "section": item.section,
                }
                for item in tk_ev
            ],
            "novelty_analysis"       : novelty,
            "inventive_step_analysis": inv_step,
            "assessment": {
                "status": status,
                "reason": reason,
            },
            "report"    : full_text,
            "confidence": confidence,
            "citations" : citations,
        }
