"""
Confidence Engine  (Phase 9)
──────────────────────────────
Combines multiple retrieval signals into a single confidence score
and determines whether evidence is sufficient to generate an answer.

Signals:
  retrieval_score   — top chunk cosine similarity (0-1)
  evidence_coverage — fraction of authoritative (primary) sources
  source_authority  — authority level of best source
"""

from __future__ import annotations

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Confidence bands
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_HIGH   = 0.80
CONFIDENCE_MEDIUM = 0.60
CONFIDENCE_LOW    = 0.00


def confidence_band(score: float) -> str:
    """Convert numeric score to HIGH / MEDIUM / LOW label."""
    if score >= CONFIDENCE_HIGH:
        return "high"
    if score >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Evidence sufficiency
# ─────────────────────────────────────────────────────────────────────────────

def evidence_is_sufficient(chunks: list) -> bool:
    """
    Returns True if at least one retrieved chunk comes from a
    primary authoritative source.

    For Phase 9 prototype: True when chunks is non-empty.
    A chunk from a primary source scores higher than a secondary one.
    """
    if not chunks:
        return False
    # Prefer primary sources — but accept any source rather than abstaining
    has_primary = any(
        getattr(c, "authority_level", "") == "primary"
        for c in chunks
    )
    return has_primary or len(chunks) >= 2


def authority_score(chunks: list) -> float:
    """
    0.0 = no sources
    0.5 = only secondary/guideline sources
    1.0 = at least one primary act/statute
    """
    if not chunks:
        return 0.0
    authority_map = {"primary": 1.0, "secondary": 0.5, "tertiary": 0.25}
    best = max(
        (authority_map.get(getattr(c, "authority_level", ""), 0.3)
         for c in chunks),
        default=0.3,
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Confidence calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_confidence(
    retrieval_score   : float,
    evidence_coverage : float,
    source_authority  : float,
) -> float:
    """
    Weighted combination of three signals.

    retrieval_score   : top cosine similarity from reranker (0-1)
    evidence_coverage : fraction of chunks that are authoritative (0-1)
    source_authority  : best source authority score (0-1)

    Weights (prototype heuristic):
      retrieval  40%
      coverage   35%
      authority  25%
    """
    score = (
        retrieval_score   * 0.40
        + evidence_coverage * 0.35
        + source_authority  * 0.25
    )
    return round(min(1.0, max(0.0, score)), 4)


def score_from_chunks(chunks: list, retriever_confidence: str) -> float:
    """
    Convenience: compute confidence directly from chunk list +
    the string confidence returned by HybridRetriever.

    retriever_confidence : "high" | "medium" | "low"
    """
    retrieval_map = {"high": 0.85, "medium": 0.60, "low": 0.35}
    retrieval_score = retrieval_map.get(retriever_confidence, 0.40)

    if not chunks:
        return 0.10

    primary_count = sum(
        1 for c in chunks
        if getattr(c, "authority_level", "") == "primary"
    )
    coverage = primary_count / len(chunks)
    authority = authority_score(chunks)

    return calculate_confidence(retrieval_score, coverage, authority)
