"""
ABS Checker  (Phase 7)
──────────────────────
Access and Benefit Sharing (ABS) relevance assessor.

Based on the Biological Diversity Act, 2002 (India) and the
Nagoya Protocol on ABS under the Convention on Biological Diversity.

This module implements a CONSERVATIVE decision tree:

  Formulation
       ↓
  Biological resource present?
       ├── NO  → potentially_relevant = False
       └── YES
              ↓
         Traditional knowledge detected?
              ├── NO  → potentially_relevant = True  (moderate)
              └── YES → potentially_relevant = True  (higher relevance)

IMPORTANT: All findings marked requires_human_review = True must be
reviewed by a qualified legal professional. This module does NOT
provide legal clearance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from formulation.schemas import FormulationClassification, ABSAssessment


# ─────────────────────────────────────────────────────────────────────────────
# Suggested provisions (populated when ABS is potentially relevant)
# ─────────────────────────────────────────────────────────────────────────────

_ABS_PROVISIONS = [
    "Biological Diversity Act, 2002 (India) — Section 3 (access to biological resources)",
    "Biological Diversity Act, 2002 (India) — Section 6 (applications for IP involving biological resources)",
    "Biological Diversity Rules, 2004 — Rule 14 (National Biodiversity Authority approval)",
    "Nagoya Protocol on Access to Genetic Resources and Benefit Sharing (2010)",
    "Convention on Biological Diversity — Article 15 (Access to Genetic Resources)",
]

_TK_PROVISIONS = [
    "Patents Act, 1970 — Section 3(p) (traditional knowledge exclusion)",
    "Patents Act, 1970 — Section 25(1)(k) / 25(2)(k) (opposition on TK grounds)",
    "Traditional Knowledge Digital Library (TKDL) — prior art database",
    "Biological Diversity Act, 2002 — Section 36(5) (protection of TK)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main assessment function
# ─────────────────────────────────────────────────────────────────────────────

def assess_abs(formulation: FormulationClassification) -> ABSAssessment:
    """
    Assess ABS relevance for a classified formulation.

    Parameters
    ----------
    formulation : FormulationClassification from classifier + extractor

    Returns
    -------
    ABSAssessment — conservative, evidence-based, always flags for human review
    """
    # ── Collect biological resources ──────────────────────────────────────
    bio_resources: list[str] = formulation.all_bio_resource_names()

    # ── Detect TK indicators ──────────────────────────────────────────────
    tk_detected = (
        bool(formulation.traditional_knowledge_indicators)
        or any(i.traditional_use_indicator for i in formulation.ingredient_objects)
        or formulation.is_tk_relevant()
    )

    # ── Decision tree ─────────────────────────────────────────────────────
    potentially_relevant = len(bio_resources) > 0

    reasons: list[str] = []
    relevant_sources: list[str] = []
    suggested_provisions: list[str] = []

    if bio_resources:
        reasons.append(
            f"Biological resource(s) identified: {', '.join(bio_resources)}. "
            "Access to biological resources in India may be subject to the "
            "Biological Diversity Act, 2002."
        )
        relevant_sources.append("Biological Diversity Act, 2002")
        relevant_sources.append("Biological Diversity Rules, 2004")
        suggested_provisions.extend(_ABS_PROVISIONS)

    if tk_detected:
        reasons.append(
            "Traditional-knowledge indicators detected. "
            "If the formulation is based on or derived from Indian traditional "
            "knowledge, additional requirements under the Patents Act, 1970 "
            "and the Biological Diversity Act, 2002 may apply."
        )
        relevant_sources.append("Traditional Knowledge Digital Library (TKDL)")
        relevant_sources.append("Patents Act, 1970 — Section 3(p)")
        suggested_provisions.extend(_TK_PROVISIONS)

    if not potentially_relevant:
        reasons.append(
            "No biological resources were identified in this formulation. "
            "ABS provisions may not be directly applicable, but further "
            "professional review is recommended."
        )

    # Deduplicate
    relevant_sources    = list(dict.fromkeys(relevant_sources))
    suggested_provisions= list(dict.fromkeys(suggested_provisions))

    return ABSAssessment(
        potentially_relevant           = potentially_relevant,
        biological_resources           = bio_resources,
        traditional_knowledge_detected = tk_detected,
        reasons                        = reasons,
        relevant_sources               = relevant_sources,
        requires_human_review          = True,   # always conservative
        suggested_provisions           = suggested_provisions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Class wrapper (for use as a singleton in the API)
# ─────────────────────────────────────────────────────────────────────────────

class ABSChecker:
    """Stateless wrapper around assess_abs for API singleton pattern."""

    def check(self, formulation: FormulationClassification) -> ABSAssessment:
        return assess_abs(formulation)
