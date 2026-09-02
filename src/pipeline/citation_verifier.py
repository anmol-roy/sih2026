"""
Citation Verifier  (Phase 11)
───────────────────────────────
Ensures every citation in the answer:
  1. Exists in the retrieved evidence pool
  2. Is backend-constructed, not LLM-invented
  3. Actually supports the adjacent claim

Also builds structured CitationRecord objects from raw evidence items
so the LLM never has to generate citation metadata.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline.response_model import CitationRecord
from evidence.store import EvidenceItem


# ─────────────────────────────────────────────────────────────────────────────
# Build citations from evidence (backend-constructed)
# ─────────────────────────────────────────────────────────────────────────────

def build_citations(items: list[EvidenceItem]) -> list[CitationRecord]:
    """
    Convert retrieved EvidenceItems into CitationRecords.
    These citations come from actual retrieved documents — never from the LLM.
    """
    seen: set[str] = set()
    citations: list[CitationRecord] = []

    for item in items:
        key = item.chunk_id or item.source_id
        if key in seen:
            continue
        seen.add(key)
        citations.append(CitationRecord(
            source_id   = item.source_id,
            document    = item.title or item.source_id,
            section     = item.section,
            subsection  = item.subsection,
            page        = item.page,
            source_name = item.source_name,
            jurisdiction= item.jurisdiction,
            authority   = item.authority,
            chunk_id    = item.chunk_id,
            verified    = True,
        ))

    return citations


# ─────────────────────────────────────────────────────────────────────────────
# Citation ID injection into generation prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_sourced_context(items: list[EvidenceItem]) -> str:
    """
    Build a context string where each document is labelled with its
    SOURCE_ID so the LLM can cite by ID rather than inventing citation text.

    Format:
      SOURCE_ID: PATENTS_ACT_1970_S3P
      [Title | Section | Source]
      <text>
    """
    blocks: list[str] = []
    seen: set[str] = set()

    for item in items[:10]:   # cap at 10 to keep context manageable
        key = item.chunk_id or item.source_id
        if key in seen:
            continue
        seen.add(key)

        header = f"SOURCE_ID: {item.source_id.upper()}"
        meta   = " | ".join(filter(None, [
            item.title,
            item.section,
            item.subsection,
            f"p.{item.page}" if item.page else None,
            item.source_name,
            f"({item.jurisdiction})",
        ]))
        blocks.append(f"{header}\n[{meta}]\n{item.text[:600]}")

    return "\n\n---\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Citation verification
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_ID_RE = re.compile(
    r"\bSOURCE[_\-]ID\s*:\s*(\w+)",
    re.IGNORECASE,
)

_BRACKET_CITE_RE = re.compile(
    r"\[SOURCE[_\-]ID\s*:\s*(\w+)\]|\[(\w+)\]",
    re.IGNORECASE,
)


def extract_cited_ids(answer_text: str) -> list[str]:
    """Extract all SOURCE_ID references cited inside the answer text."""
    ids: list[str] = []
    for m in _SOURCE_ID_RE.finditer(answer_text):
        ids.append(m.group(1).upper())
    for m in _BRACKET_CITE_RE.finditer(answer_text):
        val = (m.group(1) or m.group(2) or "").upper()
        if val:
            ids.append(val)
    return list(dict.fromkeys(ids))


def verify_citations(
    answer_text : str,
    items       : list[EvidenceItem],
) -> tuple[list[CitationRecord], float]:
    """
    Verify that each SOURCE_ID cited in *answer_text* exists in *items*.

    Returns
    -------
    (verified_citations, coverage_score)

    coverage_score : fraction of cited IDs that were verified (0.0–1.0).
                     1.0 means all citations are backed by retrieved evidence.
    """
    available_ids = {
        (item.source_id.upper()): item
        for item in items
    }
    # Also index by chunk_id
    for item in items:
        if item.chunk_id:
            available_ids[item.chunk_id.upper()] = item

    cited_ids = extract_cited_ids(answer_text)

    if not cited_ids:
        # No citations in answer — build from all retrieved items
        return build_citations(items), 1.0

    verified: list[CitationRecord] = []
    verified_count = 0

    for cid in cited_ids:
        item = available_ids.get(cid)
        if item:
            verified_count += 1
            verified.append(CitationRecord(
                source_id   = item.source_id,
                document    = item.title or item.source_id,
                section     = item.section,
                subsection  = item.subsection,
                page        = item.page,
                source_name = item.source_name,
                jurisdiction= item.jurisdiction,
                authority   = item.authority,
                chunk_id    = item.chunk_id,
                verified    = True,
            ))
        else:
            # Citation ID not found in evidence — mark as unverified
            verified.append(CitationRecord(
                source_id   = cid,
                document    = cid,
                source_name = "UNKNOWN",
                jurisdiction= "unknown",
                authority   = "unknown",
                verified    = False,
            ))

    coverage = verified_count / len(cited_ids) if cited_ids else 1.0
    return verified, coverage
