"""
Confidence Engine  (Phase 11)
───────────────────────────────
Multi-signal confidence calculator.

Signals (and weights):
  retrieval_quality   0.35  — top similarity score from retriever
  source_authority    0.25  — best authority level in evidence pool
  evidence_coverage   0.20  — fraction of claims with verified citations
  citation_validity   0.15  — fraction of cited sources that are verified
  source_agreement    0.05  — 1.0 if no conflicts, lower if conflicts exist

Final score → band:
  high   : ≥ 0.80
  medium : 0.60 – 0.79
  low    : < 0.60
"""

from __future__ import annotations

from evidence.store import EvidenceItem, Authority


# ─────────────────────────────────────────────────────────────────────────────
# Signal weights
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "retrieval_quality": 0.35,
    "source_authority" : 0.25,
    "evidence_coverage": 0.20,
    "citation_validity": 0.15,
    "source_agreement" : 0.05,
}

# Band thresholds
_HIGH   = 0.80
_MEDIUM = 0.60


def confidence_band(score: float) -> str:
    if score >= _HIGH:
        return "high"
    if score >= _MEDIUM:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Individual signal calculators
# ─────────────────────────────────────────────────────────────────────────────

def _retrieval_quality(items: list[EvidenceItem], retriever_band: str) -> float:
    """Combine retriever confidence band with top similarity score."""
    band_map = {"high": 0.90, "medium": 0.65, "low": 0.35}
    band_score = band_map.get(retriever_band, 0.40)
    if not items:
        return band_score * 0.5
    top_sim = max((i.similarity_score for i in items), default=0.0)
    return (band_score + top_sim) / 2.0


def _source_authority(items: list[EvidenceItem]) -> float:
    """Best authority score across all items."""
    if not items:
        return 0.0
    return max(Authority.score(i.authority) for i in items)


def _evidence_coverage(citation_coverage: float) -> float:
    """Pass-through: fraction of claims that have verified evidence."""
    return max(0.0, min(1.0, citation_coverage))


def _citation_validity(verified_citations: list) -> float:
    """Fraction of citations that are marked verified=True."""
    if not verified_citations:
        return 1.0  # no citations → not penalised
    valid = sum(1 for c in verified_citations if getattr(c, "verified", True))
    return valid / len(verified_citations)


def _source_agreement(conflict_count: int) -> float:
    """Penalty for conflicting sources."""
    if conflict_count == 0:
        return 1.0
    if conflict_count == 1:
        return 0.75
    return max(0.40, 1.0 - conflict_count * 0.20)


# ─────────────────────────────────────────────────────────────────────────────
# Main confidence function
# ─────────────────────────────────────────────────────────────────────────────

def calculate_confidence(
    items             : list[EvidenceItem],
    retriever_band    : str   = "low",
    citation_coverage : float = 1.0,
    verified_citations: list  = None,
    conflict_count    : int   = 0,
) -> tuple[float, str, dict]:
    """
    Compute multi-signal confidence.

    Returns
    -------
    (score, band, signals)
    score   : float 0–1
    band    : "high" | "medium" | "low"
    signals : dict with individual signal values for transparency
    """
    verified_citations = verified_citations or []

    signals = {
        "retrieval_quality": _retrieval_quality(items, retriever_band),
        "source_authority" : _source_authority(items),
        "evidence_coverage": _evidence_coverage(citation_coverage),
        "citation_validity": _citation_validity(verified_citations),
        "source_agreement" : _source_agreement(conflict_count),
    }

    score = sum(
        signals[k] * _WEIGHTS[k]
        for k in signals
    )
    score = round(min(1.0, max(0.0, score)), 4)
    band  = confidence_band(score)

    return score, band, signals
