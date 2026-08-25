"""
Evidence Fusion
────────────────
Merges legal, patent, and TK evidence into a single structured package
that is then passed to the report generator.

Output shape
------------
{
  "invention"                : Invention,
  "legal_evidence"           : list[EvidenceItem],
  "patent_evidence"          : list[EvidenceItem],
  "tk_evidence"              : list[EvidenceItem],
  "issues"                   : list[str],
  "overall_confidence"       : "high" | "medium" | "low",
}

EvidenceItem
------------
{
  "type"    : "PRIMARY_LAW" | "PATENT_RECORD" | "TRADITIONAL_KNOWLEDGE"
              | "AYUSH_REFERENCE" | "MODEL_INFERENCE",
  "label"   : str,          # short human-readable label
  "text"    : str,          # relevant excerpt
  "source"  : str,
  "section" : str | None,
  "page"    : int | None,
  "chunk_id": str,
  "score"   : float | None, # similarity score if available
}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention, LegalChunk, PatentChunk


# ---------------------------------------------------------------------------
# Evidence item
# ---------------------------------------------------------------------------

class EvidenceItem:
    __slots__ = ("type", "label", "text", "source", "section",
                 "page", "chunk_id", "score")

    def __init__(
        self,
        evidence_type: str,
        label: str,
        text: str,
        source: str,
        section: Optional[str] = None,
        page: Optional[int] = None,
        chunk_id: str = "",
        score: Optional[float] = None,
    ):
        self.type     = evidence_type
        self.label    = label
        self.text     = text
        self.source   = source
        self.section  = section
        self.page     = page
        self.chunk_id = chunk_id
        self.score    = score

    def to_dict(self) -> dict:
        return {
            "type":     self.type,
            "label":    self.label,
            "text":     self.text[:500],   # truncate for API response
            "source":   self.source,
            "section":  self.section,
            "page":     self.page,
            "chunk_id": self.chunk_id,
            "score":    round(self.score, 4) if self.score is not None else None,
        }


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def _legal_chunk_to_evidence(chunk: LegalChunk, score: Optional[float] = None) -> EvidenceItem:
    label_parts = [chunk.title]
    if chunk.section:
        label_parts.append(chunk.section)
    if chunk.subsection:
        label_parts.append(f"§ {chunk.subsection}")

    return EvidenceItem(
        evidence_type="PRIMARY_LAW",
        label=", ".join(label_parts),
        text=chunk.text,
        source=chunk.source,
        section=chunk.subsection or chunk.section,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
        score=score,
    )


def _patent_chunk_to_evidence(
    chunk: PatentChunk,
    score: Optional[float] = None,
    matched_components: Optional[list[str]] = None,
) -> EvidenceItem:
    label = f"{chunk.publication_number} — {chunk.title}"
    if chunk.section_type == "claim" and chunk.claim_number:
        label += f" (Claim {chunk.claim_number})"

    text = chunk.text
    if matched_components:
        text = f"[Matched: {', '.join(matched_components)}]\n\n{text}"

    return EvidenceItem(
        evidence_type="PATENT_RECORD",
        label=label,
        text=text,
        source=chunk.source,
        section=chunk.section_type,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
        score=score,
    )


def _tk_chunk_to_evidence(chunk: LegalChunk, score: Optional[float] = None) -> EvidenceItem:
    label_parts = [chunk.title]
    if chunk.section:
        label_parts.append(chunk.section)

    return EvidenceItem(
        evidence_type="TRADITIONAL_KNOWLEDGE",
        label=", ".join(label_parts),
        text=chunk.text,
        source=chunk.source,
        section=chunk.section,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
        score=score,
    )


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------

_ISSUE_TRIGGERS: list[tuple[str, str]] = [
    # (keyword_in_legal_chunk_subsection_or_section, issue_message)
    ("3(p)",  "Section 3(p) of the Patents Act may exclude inventions that are in effect traditional knowledge."),
    ("3(d)",  "Section 3(d) of the Patents Act may exclude new forms of known substances without enhanced efficacy."),
    ("3(e)",  "Section 3(e) may be relevant if the invention involves mixing known components without synergy."),
    ("3(i)",  "Section 3(i) excludes methods of treatment of human beings or animals."),
    ("3(j)",  "Section 3(j) excludes plants, animals, and essentially biological processes."),
    ("3(b)",  "Section 3(b) may be relevant if the invention is contrary to public order or morality."),
]


def detect_issues(
    legal_evidence: list[EvidenceItem],
    tk_evidence: list[EvidenceItem],
) -> list[str]:
    issues: list[str] = []
    seen: set[str] = set()

    for item in legal_evidence:
        sec = (item.section or "").lower()
        for trigger, message in _ISSUE_TRIGGERS:
            if trigger.lower() in sec and message not in seen:
                issues.append(message)
                seen.add(message)

    if tk_evidence:
        msg = (
            "Retrieved traditional-knowledge or AYUSH material contains "
            "potentially similar formulations or uses. This may be relevant "
            "to the Section 3(p) assessment."
        )
        if msg not in seen:
            issues.append(msg)

    return issues


# ---------------------------------------------------------------------------
# Overall confidence
# ---------------------------------------------------------------------------

def _overall_confidence(
    legal: list[EvidenceItem],
    patents: list[EvidenceItem],
    tk: list[EvidenceItem],
) -> str:
    scores = [
        item.score for item in (legal + patents + tk)
        if item.score is not None
    ]
    if not scores:
        return "low"
    avg = sum(scores) / len(scores)
    if avg >= 0.65:
        return "high"
    if avg >= 0.45:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------

def fuse_evidence(
    invention: Invention,
    legal_chunks: list[tuple[LegalChunk, float]],
    patent_results: list[dict],
    tk_chunks: list[tuple[LegalChunk, float]],
) -> dict:
    """
    Parameters
    ----------
    invention       : structured invention object
    legal_chunks    : list of (LegalChunk, score) from legal retriever
    patent_results  : list of patent similarity dicts from patent_search
    tk_chunks       : list of (LegalChunk, score) from TK retriever

    Returns
    -------
    Fused evidence package dict
    """
    # ── Legal evidence ────────────────────────────────────────────────────
    legal_evidence = [
        _legal_chunk_to_evidence(chunk, score)
        for chunk, score in legal_chunks
    ]

    # ── Patent evidence ───────────────────────────────────────────────────
    patent_evidence = []
    for pr in patent_results:
        chunk = pr.get("chunk")
        score = pr.get("similarity_score", 0.0)
        matched = pr.get("matched_components", [])
        if isinstance(chunk, PatentChunk):
            patent_evidence.append(_patent_chunk_to_evidence(chunk, score, matched))
        elif isinstance(chunk, LegalChunk):
            # Patent was stored as LegalChunk (e.g. from legal corpus patent section)
            patent_evidence.append(_legal_chunk_to_evidence(chunk, score))

    # ── TK evidence ───────────────────────────────────────────────────────
    tk_evidence = [
        _tk_chunk_to_evidence(chunk, score)
        for chunk, score in tk_chunks
    ]

    issues = detect_issues(legal_evidence, tk_evidence)
    confidence = _overall_confidence(legal_evidence, patent_evidence, tk_evidence)

    return {
        "invention":          invention,
        "legal_evidence":     legal_evidence,
        "patent_evidence":    patent_evidence,
        "tk_evidence":        tk_evidence,
        "issues":             issues,
        "overall_confidence": confidence,
    }
