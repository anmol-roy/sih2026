"""
Conflict Detector  (Phase 11)
───────────────────────────────
Detects contradictions in retrieved evidence.

For the prototype two detection strategies:
  1. Version conflict — two chunks from the same Act but different versions
  2. Statement conflict — two chunks with contradictory key phrases

When a conflict is found:
  - confidence is reduced
  - needs_human_review is set to True
  - the newer/higher-authority source is preferred
"""

from __future__ import annotations

import re
from pipeline.response_model import ConflictRecord
from evidence.store import EvidenceItem, Authority


# ─────────────────────────────────────────────────────────────────────────────
# Contradiction phrase pairs
# ─────────────────────────────────────────────────────────────────────────────

# (phrase_a, phrase_b) — if both appear in the same evidence pool,
# this *might* indicate a conflict.  Conservative: only use high-confidence pairs.
_CONTRADICTION_PAIRS: list[tuple[str, str]] = [
    ("is patentable",    "is not patentable"),
    ("shall be granted", "shall not be granted"),
    ("is registrable",   "is not registrable"),
    ("is excluded",      "is not excluded"),
    ("does not apply",   "does apply"),
]


def _text_contains(text: str, phrase: str) -> bool:
    return bool(re.search(re.escape(phrase), text, re.IGNORECASE))


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_conflicts(items: list[EvidenceItem]) -> list[ConflictRecord]:
    """
    Scan evidence items for potential conflicts.

    Returns a list of ConflictRecord objects (may be empty).
    """
    conflicts: list[ConflictRecord] = []

    # ── Strategy 1: Version conflict ─────────────────────────────────────
    # Group items by document_id; if same doc has different versions, flag it.
    doc_versions: dict[str, list[EvidenceItem]] = {}
    for item in items:
        # Use first part of source_id as document key
        doc_key = item.source_id.split("_")[0] if "_" in item.source_id else item.source_id
        doc_versions.setdefault(doc_key, []).append(item)

    for doc_key, doc_items in doc_versions.items():
        versions = list({
            getattr(item, "version", None) or "unknown"
            for item in doc_items
        })
        versions = [v for v in versions if v != "unknown"]
        if len(versions) > 1:
            conflicts.append(ConflictRecord(
                source_a   = doc_items[0].source_id,
                source_b   = doc_items[-1].source_id,
                description= (
                    f"Multiple versions of '{doc_key}' detected: {versions}. "
                    "Prefer the latest effective version."
                ),
                resolution = "prefer_latest_version",
            ))

    # ── Strategy 2: Statement conflict ───────────────────────────────────
    all_text = " ".join(item.text for item in items)
    for phrase_a, phrase_b in _CONTRADICTION_PAIRS:
        if _text_contains(all_text, phrase_a) and _text_contains(all_text, phrase_b):
            # Find which items contain which phrase
            items_a = [i for i in items if _text_contains(i.text, phrase_a)]
            items_b = [i for i in items if _text_contains(i.text, phrase_b)]
            if items_a and items_b and items_a[0].source_id != items_b[0].source_id:
                # Determine preferred: higher authority wins
                auth_a = Authority.score(items_a[0].authority)
                auth_b = Authority.score(items_b[0].authority)
                preferred = items_a[0].source_id if auth_a >= auth_b else items_b[0].source_id
                conflicts.append(ConflictRecord(
                    source_a   = items_a[0].source_id,
                    source_b   = items_b[0].source_id,
                    description= (
                        f"Potentially contradictory statements detected: "
                        f"'{phrase_a}' vs '{phrase_b}'."
                    ),
                    resolution = f"prefer_higher_authority:{preferred}",
                ))

    # Deduplicate by source pair
    seen: set[frozenset] = set()
    unique: list[ConflictRecord] = []
    for c in conflicts:
        key = frozenset([c.source_a, c.source_b])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def prefer_authoritative(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """
    When conflicts exist, sort items by authority descending so the
    most authoritative source leads generation.
    """
    return sorted(items, key=lambda i: Authority.score(i.authority), reverse=True)
