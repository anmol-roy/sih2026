"""
Evidence Store  (Phase 10)
───────────────────────────
Normalised evidence item schema used across all tools.

Every piece of evidence — whether from legal search, patent search,
TK search, or ABS check — is converted into an EvidenceItem so the
evidence fusion and ranking layers work with a uniform format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Evidence type constants
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceType:
    LEGAL               = "legal"
    PATENT              = "patent"
    TRADITIONAL_KNOWLEDGE = "traditional_knowledge"
    ABS                 = "abs"
    INTERNATIONAL       = "international"
    FORMULATION         = "formulation"


class Authority:
    PRIMARY     = "primary"       # Acts, statutes, official text
    REGISTRY    = "registry"      # Patent office records, GI registry
    GOVERNMENT  = "government"    # Official government publications
    SECONDARY   = "secondary"     # Guidelines, AYUSH publications
    OFFICIAL    = "official"      # Same as secondary but marks TKDL/AYUSH
    GENERAL     = "general"       # Other sources

    SCORES: dict[str, float] = {
        "primary"  : 1.00,
        "registry" : 0.95,
        "government": 0.90,
        "secondary": 0.70,
        "official" : 0.70,
        "general"  : 0.40,
    }

    @classmethod
    def score(cls, authority: str) -> float:
        return cls.SCORES.get(authority.lower(), 0.40)


# ─────────────────────────────────────────────────────────────────────────────
# Normalised evidence item
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """
    One normalised piece of evidence from any source.

    All tools return list[EvidenceItem] so fusion and ranking work
    without caring about the source type.
    """

    source_id          : str
    source_type        : str                      # EvidenceType constant
    jurisdiction       : str  = "india"
    domain             : str  = "patent"
    text               : str  = ""
    section            : Optional[str]  = None
    subsection         : Optional[str]  = None
    page               : Optional[int]  = None
    authority          : str  = Authority.SECONDARY
    publication_date   : Optional[str]  = None
    chunk_id           : str  = ""
    title              : str  = ""
    source_name        : str  = ""
    source_url         : Optional[str]  = None
    similarity_score   : float = 0.0

    # Graph population helpers
    entities           : list[str] = field(default_factory=list)  # extracted entity names

    def authority_score(self) -> float:
        return Authority.score(self.authority)

    def to_dict(self) -> dict:
        return {
            "source_id"      : self.source_id,
            "source_type"    : self.source_type,
            "jurisdiction"   : self.jurisdiction,
            "domain"         : self.domain,
            "text"           : self.text[:400] + ("…" if len(self.text) > 400 else ""),
            "section"        : self.section,
            "subsection"     : self.subsection,
            "page"           : self.page,
            "authority"      : self.authority,
            "publication_date": self.publication_date,
            "chunk_id"       : self.chunk_id,
            "title"          : self.title,
            "source_name"    : self.source_name,
            "source_url"     : self.source_url,
            "similarity_score": round(self.similarity_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Converters from existing chunk types
# ─────────────────────────────────────────────────────────────────────────────

def from_legal_chunk(chunk, score: float = 0.5) -> EvidenceItem:
    return EvidenceItem(
        source_id   = getattr(chunk, "chunk_id", "unknown"),
        source_type = EvidenceType.LEGAL,
        jurisdiction= getattr(chunk, "jurisdiction", "india"),
        domain      = getattr(chunk, "domain", "general"),
        text        = getattr(chunk, "text", ""),
        section     = getattr(chunk, "section", None),
        subsection  = getattr(chunk, "subsection", None),
        page        = getattr(chunk, "page", None),
        authority   = getattr(chunk, "authority_level", Authority.SECONDARY),
        chunk_id    = getattr(chunk, "chunk_id", ""),
        title       = getattr(chunk, "title", ""),
        source_name = getattr(chunk, "source", ""),
        source_url  = getattr(chunk, "source_url", None),
        similarity_score = score,
    )


def from_patent_chunk(chunk, score: float = 0.5) -> EvidenceItem:
    return EvidenceItem(
        source_id   = getattr(chunk, "publication_number", getattr(chunk, "chunk_id", "unknown")),
        source_type = EvidenceType.PATENT,
        jurisdiction= "india",
        domain      = "patent",
        text        = getattr(chunk, "text", ""),
        page        = getattr(chunk, "page", None),
        authority   = Authority.REGISTRY,
        publication_date = getattr(chunk, "publication_date", None),
        chunk_id    = getattr(chunk, "chunk_id", ""),
        title       = getattr(chunk, "title", ""),
        source_name = getattr(chunk, "source", "IP India"),
        similarity_score = score,
    )


def from_tk_match(match: dict) -> EvidenceItem:
    chunk = match.get("chunk")
    if chunk is None:
        return EvidenceItem(
            source_id="unknown_tk", source_type=EvidenceType.TRADITIONAL_KNOWLEDGE
        )
    return EvidenceItem(
        source_id   = getattr(chunk, "chunk_id", "unknown_tk"),
        source_type = EvidenceType.TRADITIONAL_KNOWLEDGE,
        jurisdiction= "india",
        domain      = "ayush",
        text        = getattr(chunk, "text", ""),
        section     = getattr(chunk, "section", None),
        page        = match.get("page") or getattr(chunk, "page", None),
        authority   = Authority.OFFICIAL,
        chunk_id    = getattr(chunk, "chunk_id", ""),
        title       = getattr(chunk, "title", ""),
        source_name = getattr(chunk, "source", "AYUSH/TKDL"),
        similarity_score = float(match.get("score", 0.0)),
        entities    = match.get("matched_components", []),
    )
