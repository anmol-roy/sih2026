"""
Inventive Step Analyzer  (Phase 4)
────────────────────────────────────
Asks: "If no single reference contains all features, do MULTIPLE
references together disclose the full combination?"

This is different from novelty:
  - Novelty   : one document discloses everything
  - Inventive step : the COMBINATION of multiple documents reveals everything

Important:
  The system does NOT conclude "lacks inventive step."
  It identifies whether the technical combination appears in the prior-art
  corpus and flags it for human examination.

Output
------
{
  "potential_overlap"   : bool,
  "feature_coverage_map": {feature_id: [document_ids that match it]},
  "uncovered_features"  : [feature_ids not found in any document],
  "reference_groups"    : list[dict],   # minimal sets of docs covering all features
  "assessment"          : "potential_inventive_step_concern"
                          | "partial_overlap"
                          | "insufficient_evidence",
  "explanation"         : str,
  "evidence"            : list[dict],
}
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import InventionFeature, PriorArtCandidate


# ---------------------------------------------------------------------------
# Inventive Step Analyzer
# ---------------------------------------------------------------------------

class InventiveStepAnalyzer:
    """
    Identify combinations of prior-art references that collectively
    disclose all invention features.

    Parameters
    ----------
    max_combination_size : maximum number of documents to combine
                           (2 or 3 is realistic; beyond that becomes speculative)
    """

    def __init__(self, max_combination_size: int = 3):
        self._max_combo = max_combination_size

    def analyze(
        self,
        features: list[InventionFeature],
        candidates: list[PriorArtCandidate],
        novelty_result: dict,
    ) -> dict:
        """
        Parameters
        ----------
        novelty_result : output of NoveltyAnalyzer.analyze()
                         If novelty already found a single-reference match,
                         inventive-step analysis is not the primary concern.
        """
        if not features or not candidates:
            return self._empty_result()

        feature_ids = [f.id for f in features]

        # ── Build feature coverage map ─────────────────────────────────────
        # feature_id → list of (publication_number, match_type, confidence)
        coverage_map: dict[str, list[dict]] = {fid: [] for fid in feature_ids}

        for cand in candidates:
            for fid, fm in cand.feature_matches.items():
                if fm.matched:
                    coverage_map[fid].append({
                        "document"   : cand.publication_number,
                        "title"      : cand.title,
                        "match_type" : fm.match_type,
                        "confidence" : fm.confidence,
                        "chunk_id"   : cand.chunk_id,
                    })

        uncovered = [fid for fid in feature_ids if not coverage_map[fid]]
        covered   = [fid for fid in feature_ids if coverage_map[fid]]
        coverage_ratio = len(covered) / len(feature_ids) if feature_ids else 0.0

        # ── Find minimal reference groups covering all features ────────────
        reference_groups = self._find_minimal_groups(
            feature_ids, candidates, max_size=self._max_combo
        )

        # ── Assessment ────────────────────────────────────────────────────
        if novelty_result.get("single_reference_found"):
            # Novelty is the bigger concern; inventive-step is secondary
            assessment  = "potential_inventive_step_concern"
            explanation = (
                "A single reference was already identified as potentially "
                "covering all features (see novelty analysis). "
                "Multi-reference inventive-step analysis is secondary in this case."
            )
        elif reference_groups and coverage_ratio >= 0.80:
            assessment  = "potential_inventive_step_concern"
            explanation = (
                f"The retrieved prior-art documents, when considered in "
                f"combination, appear to disclose "
                f"{coverage_ratio*100:.0f}% of the identified features. "
                f"The smallest identified combination has "
                f"{min(len(g['documents']) for g in reference_groups)} document(s). "
                f"Whether such a combination would have been obvious to a person "
                f"skilled in the art requires expert legal examination."
            )
        elif coverage_ratio >= 0.50:
            assessment  = "partial_overlap"
            explanation = (
                f"The retrieved evidence covers {coverage_ratio*100:.0f}% of "
                f"features across multiple documents. The uncovered features are: "
                f"{', '.join(uncovered) or 'none'}. Further prior-art search is recommended."
            )
        else:
            assessment  = "insufficient_evidence"
            explanation = (
                "The retrieved prior-art corpus does not sufficiently cover the "
                "identified features in combination. This does not guarantee a "
                "positive inventive-step finding — the corpus may be incomplete."
            )

        return {
            "potential_overlap"    : assessment == "potential_inventive_step_concern",
            "feature_coverage_map" : {
                fid: docs for fid, docs in coverage_map.items()
            },
            "uncovered_features"   : uncovered,
            "covered_features"     : covered,
            "coverage_ratio"       : round(coverage_ratio, 3),
            "reference_groups"     : reference_groups[:5],
            "assessment"           : assessment,
            "explanation"          : explanation,
            "evidence"             : self._build_evidence(coverage_map),
        }

    # ------------------------------------------------------------------

    def _find_minimal_groups(
        self,
        feature_ids: list[str],
        candidates: list[PriorArtCandidate],
        max_size: int,
    ) -> list[dict]:
        """
        Find combinations of ≤ max_size candidates that together cover
        all features. Returns a list of group dicts sorted by size.
        """
        groups = []

        # Build per-candidate feature sets
        cand_feature_sets: list[tuple[PriorArtCandidate, set[str]]] = [
            (c, set(c.matched_feature_ids()))
            for c in candidates
        ]

        target = set(feature_ids)

        for size in range(1, max_size + 1):
            for combo in combinations(cand_feature_sets, size):
                union = set()
                for _, fset in combo:
                    union |= fset
                if target.issubset(union):
                    groups.append({
                        "documents": [c.publication_number for c, _ in combo],
                        "titles"   : [c.title for c, _ in combo],
                        "coverage" : sorted(union & target),
                    })
            if groups:
                break   # stop at smallest combo size that works

        return groups[:10]

    @staticmethod
    def _build_evidence(
        coverage_map: dict[str, list[dict]]
    ) -> list[dict]:
        evidence = []
        for fid, docs in coverage_map.items():
            for doc in docs[:2]:
                evidence.append({
                    "feature_id" : fid,
                    "document"   : doc["document"],
                    "title"      : doc["title"],
                    "match_type" : doc["match_type"],
                    "confidence" : doc["confidence"],
                })
        return evidence

    @staticmethod
    def _empty_result() -> dict:
        return {
            "potential_overlap"    : False,
            "feature_coverage_map" : {},
            "uncovered_features"   : [],
            "covered_features"     : [],
            "coverage_ratio"       : 0.0,
            "reference_groups"     : [],
            "assessment"           : "insufficient_evidence",
            "explanation"          : "Insufficient data for inventive-step analysis.",
            "evidence"             : [],
        }
