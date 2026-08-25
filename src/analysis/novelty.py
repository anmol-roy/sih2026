"""
Novelty Analyzer  (Phase 4)
────────────────────────────
Asks: "Is there a SINGLE prior-art reference that discloses all
relevant features of this invention?"

A single document that discloses every feature is potentially
novelty-destroying evidence. Multiple documents together that
cover all features is NOT the same — that is an inventive-step concern.

Output
------
{
  "single_reference_found"   : bool,
  "full_coverage_candidates" : list[dict],   # candidates covering all features
  "partial_coverage"         : list[dict],   # candidates covering ≥ 50 %
  "assessment"               : "potential_novelty_issue"
                               | "partial_prior_art"
                               | "no_prior_art_found",
  "explanation"              : str,
  "evidence"                 : list[dict],
}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import InventionFeature, PriorArtCandidate


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

FULL_COVERAGE_THRESHOLD    = 0.80   # ≥ 80 % of features matched → "full coverage"
PARTIAL_COVERAGE_THRESHOLD = 0.50   # ≥ 50 % → "partial"


# ---------------------------------------------------------------------------
# Novelty Analyzer
# ---------------------------------------------------------------------------

class NoveltyAnalyzer:
    """
    Determine whether any single candidate covers the invention fully.

    Parameters
    ----------
    min_features : minimum number of features that must be present for any
                   assessment to be meaningful (avoid false positives when
                   only 1-2 features were extracted)
    """

    def __init__(self, min_features: int = 2):
        self._min_features = min_features

    def analyze(
        self,
        features: list[InventionFeature],
        candidates: list[PriorArtCandidate],
    ) -> dict:
        if not features or not candidates:
            return self._empty_result()

        total = len(features)
        full_coverage: list[PriorArtCandidate]    = []
        partial_coverage: list[PriorArtCandidate] = []

        for cand in candidates:
            cov = cand.coverage()
            if cov >= FULL_COVERAGE_THRESHOLD:
                full_coverage.append(cand)
            elif cov >= PARTIAL_COVERAGE_THRESHOLD:
                partial_coverage.append(cand)

        # ── Assessment ────────────────────────────────────────────────────
        if full_coverage:
            assessment = "potential_novelty_issue"
            explanation = (
                f"{len(full_coverage)} retrieved prior-art reference(s) appear to "
                f"disclose {FULL_COVERAGE_THRESHOLD*100:.0f}% or more of the "
                f"identified features. This warrants detailed examination. "
                f"Note: this is a preliminary evidence-based finding, not a "
                f"legal determination of novelty."
            )
        elif partial_coverage:
            assessment = "partial_prior_art"
            explanation = (
                f"No single reference covers all features, but "
                f"{len(partial_coverage)} reference(s) cover 50% or more. "
                f"Further examination is recommended."
            )
        else:
            assessment = "no_prior_art_found"
            explanation = (
                "No retrieved prior-art document covers a significant portion "
                "of the identified features. This does not guarantee novelty — "
                "the prior-art corpus may be incomplete."
            )

        return {
            "single_reference_found"  : len(full_coverage) > 0,
            "full_coverage_candidates": [c.to_summary() for c in full_coverage[:3]],
            "partial_coverage"        : [c.to_summary() for c in partial_coverage[:5]],
            "assessment"              : assessment,
            "explanation"             : explanation,
            "evidence"                : self._build_evidence(full_coverage + partial_coverage),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence(candidates: list[PriorArtCandidate]) -> list[dict]:
        evidence = []
        for cand in candidates[:5]:
            for fid, fm in cand.feature_matches.items():
                if fm.matched:
                    evidence.append({
                        "document"      : cand.publication_number,
                        "title"         : cand.title,
                        "feature_id"    : fid,
                        "feature_text"  : fm.feature_text,
                        "match_type"    : fm.match_type,
                        "confidence"    : fm.confidence,
                        "evidence_text" : (fm.evidence_text or "")[:300],
                        "page"          : fm.evidence_page,
                    })
        return evidence

    @staticmethod
    def _empty_result() -> dict:
        return {
            "single_reference_found"  : False,
            "full_coverage_candidates": [],
            "partial_coverage"        : [],
            "assessment"              : "no_prior_art_found",
            "explanation"             : "Insufficient data for novelty analysis.",
            "evidence"                : [],
        }
