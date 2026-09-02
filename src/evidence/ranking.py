"""
Evidence Ranking  (Phase 10)
──────────────────────────────
Ranks and deduplicates evidence items by authority + similarity score.

Authority weights (from spec):
  primary    1.00   — Acts, statutes
  registry   0.95   — Patent office records
  government 0.90   — Official government publications
  secondary  0.70   — Guidelines, AYUSH publications
  official   0.70   — TKDL, AYUSH
  general    0.40   — Other
"""

from __future__ import annotations

from evidence.store import EvidenceItem, Authority


# ─────────────────────────────────────────────────────────────────────────────
# Combined score
# ─────────────────────────────────────────────────────────────────────────────

def combined_score(item: EvidenceItem) -> float:
    """
    authority_weight × 0.60  +  similarity_score × 0.40
    """
    return round(
        item.authority_score() * 0.60 + item.similarity_score * 0.40,
        4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """
    Remove duplicate items by chunk_id (keep highest-scoring version).
    Falls back to source_id when chunk_id is missing.
    """
    seen: dict[str, EvidenceItem] = {}
    for item in items:
        key = item.chunk_id or item.source_id
        if key not in seen or combined_score(item) > combined_score(seen[key]):
            seen[key] = item
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# Public ranking API
# ─────────────────────────────────────────────────────────────────────────────

def rank_evidence(
    items        : list[EvidenceItem],
    top_k        : int  = 10,
    min_score    : float = 0.0,
) -> list[EvidenceItem]:
    """
    Deduplicate, score, and return top-k evidence items.

    Parameters
    ----------
    items     : raw evidence from all tools
    top_k     : max items to return
    min_score : discard items whose combined score is below this
    """
    unique = deduplicate(items)
    scored = [(combined_score(item), item) for item in unique]
    scored = [(s, item) for s, item in scored if s >= min_score]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def rank_by_type(
    items: list[EvidenceItem],
) -> dict[str, list[EvidenceItem]]:
    """
    Return a dict keyed by source_type with each list ranked.
    """
    from collections import defaultdict
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in items:
        grouped[item.source_type].append(item)
    return {t: rank_evidence(g) for t, g in grouped.items()}
