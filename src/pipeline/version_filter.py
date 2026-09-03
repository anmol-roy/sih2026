"""
Version-Aware Retrieval Filter  (Phase 11 — 2nd half)
───────────────────────────────────────────────────────
Enforces "current / applicable version" priority in retrieved evidence.

For each document_id group, returns only the most current version
unless the query explicitly asks for historical material.

Metadata used:
  effective_date   — ISO date when this version came into force
  version          — version label (e.g. "as amended up to 2023")
  status           — "current" | "superseded" | "historical"
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from evidence.store import EvidenceItem


# ─────────────────────────────────────────────────────────────────────────────
# Historical query detection
# ─────────────────────────────────────────────────────────────────────────────

_HISTORICAL_RE = re.compile(
    r"\boriginal\b|\bhistorical?\b|\bold\s+version\b|"
    r"\bpre[-\s]\d{4}\b|\bbefore\s+\d{4}\b|"
    r"\b(1970|1999|2000|1957|2002)\s+version\b",
    re.IGNORECASE,
)


def is_historical_query(query: str) -> bool:
    """True if the user is asking about a historical/old version."""
    return bool(_HISTORICAL_RE.search(query))


# ─────────────────────────────────────────────────────────────────────────────
# Date parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str or not date_str.strip():
        return None
    s = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Version filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_version(
    items        : list[EvidenceItem],
    query        : str,
    cutoff_date  : Optional[str] = None,
) -> list[EvidenceItem]:
    """
    Filter evidence items to prefer the current/applicable version.

    Parameters
    ----------
    items        : all retrieved evidence items
    query        : user query (used to detect historical requests)
    cutoff_date  : ISO date — treat documents effective after this date
                   as "future" and exclude them (for prior-art dating)

    Returns
    -------
    Filtered + sorted list with current versions first.
    """
    if not items:
        return items

    historical = is_historical_query(query)
    cutoff     = _parse_date(cutoff_date)

    # ── Apply cutoff date (prior-art use case) ────────────────────────────
    if cutoff:
        items = [
            item for item in items
            if _parse_date(item.publication_date) is None
            or _parse_date(item.publication_date) <= cutoff
        ]

    # ── Group by document_id (base key) ──────────────────────────────────
    groups: dict[str, list[EvidenceItem]] = {}
    for item in items:
        # Use first two parts of source_id as doc key to group versions
        parts  = item.source_id.split("_")
        doc_key= "_".join(parts[:2]) if len(parts) >= 2 else item.source_id
        groups.setdefault(doc_key, []).append(item)

    result: list[EvidenceItem] = []

    for doc_key, doc_items in groups.items():
        if historical:
            # Historical query: return all versions sorted oldest first
            sorted_items = sorted(
                doc_items,
                key=lambda i: _parse_date(i.publication_date) or date.min,
            )
        else:
            # Current query: prefer items marked current or latest effective date
            def _sort_key(item: EvidenceItem):
                # Items with "current" in version string get top priority
                version = getattr(item, "version", "") or ""
                is_current = "current" in version.lower() or "latest" in version.lower()
                eff_date = _parse_date(item.publication_date) or date.min
                return (int(is_current), eff_date)

            sorted_items = sorted(doc_items, key=_sort_key, reverse=True)

        result.extend(sorted_items)

    return result


def prefer_current_versions(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """
    Simple sort: items with 'current' version label come first.
    Used after conflict detection to prefer the authoritative/current source.
    """
    def _key(item: EvidenceItem):
        version   = getattr(item, "version", "") or ""
        is_current= int("current" in version.lower())
        auth_score= item.authority_score()
        return (is_current, auth_score)

    return sorted(items, key=_key, reverse=True)
