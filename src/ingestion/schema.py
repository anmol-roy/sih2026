"""
Document chunk schemas for IP-SAKTI.

LegalChunk           — one structural unit of a legal act / rule / guideline
PatentChunk          — one structural section of a patent document
Invention            — structured representation of a user-submitted invention
InventionFeature     — one extracted technical feature of an invention
ClaimRepresentation  — claim-like technical representation of the invention
FeatureMatch         — result of matching one feature against one document
PriorArtCandidate    — a candidate prior-art document with feature-match matrix
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Legal document chunk
# ═══════════════════════════════════════════════════════════════════════════

class LegalChunk(BaseModel):
    chunk_id: str = Field(description="Unique ID")
    document_id: str = Field(description="Stable machine identifier")
    title: str = Field(description="Full document title")
    source: str = Field(description="Publishing body")
    source_url: Optional[str] = None
    document_type: str = Field(description="act | rule | regulation | notification | guideline | form")
    domain: str = Field(description="patent | trademark | copyright | design | gi | ayush | general")
    authority_level: str = Field(description="primary | secondary | tertiary")
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    page: Optional[int] = None
    language: str = "english"
    version: Optional[str] = None
    effective_date: Optional[str] = None
    last_verified: str = "2026-08-25"
    jurisdiction: str = "india"            # "india" | "international"
    text: str

    def to_metadata(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def citation(self) -> dict:
        return {
            "document": self.title,
            "section": self.section or "",
            "subsection": self.subsection or "",
            "chapter": self.chapter or "",
            "page": self.page,
            "source": self.source,
            "source_url": self.source_url or "",
            "domain": self.domain,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
        }

    def citation_label(self) -> str:
        parts = [self.title]
        if self.chapter:
            parts.append(self.chapter)
        if self.section:
            parts.append(self.section)
        if self.subsection:
            parts.append(f"§ {self.subsection}")
        if self.page is not None:
            parts.append(f"p. {self.page}")
        parts.append(f"({self.source})")
        return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Patent document chunk
# ═══════════════════════════════════════════════════════════════════════════

class PatentChunk(BaseModel):
    """One structural section of a patent document."""

    chunk_id: str
    document_id: str
    publication_number: str
    application_number: Optional[str] = None
    title: str
    applicant: Optional[str] = None
    inventor: Optional[str] = None
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    classification: Optional[str] = None
    ipc_codes: List[str] = Field(default_factory=list)
    status: Optional[str] = None
    source: str = "IP India"
    source_url: Optional[str] = None
    section_type: str = Field(
        description="abstract | background | summary | description | claim | classification"
    )
    claim_number: Optional[int] = None
    page: Optional[int] = None
    text: str

    def to_metadata(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def citation(self) -> dict:
        return {
            "document": self.title,
            "publication_number": self.publication_number,
            "section_type": self.section_type,
            "claim_number": self.claim_number,
            "classification": self.classification or "",
            "source": self.source,
            "source_url": self.source_url or "",
            "filing_date": self.filing_date or "",
            "chunk_id": self.chunk_id,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Invention
# ═══════════════════════════════════════════════════════════════════════════

class Invention(BaseModel):
    """Structured representation of a user-submitted invention description."""

    title: str = Field(description="Short, descriptive title")
    technical_field: Optional[str] = None
    problem: Optional[str] = None
    solution: Optional[str] = None
    components: List[str] = Field(default_factory=list)
    intended_use: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    novelty_claim: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Feature-level analysis schemas
# ═══════════════════════════════════════════════════════════════════════════

class InventionFeature(BaseModel):
    """
    One discrete technical feature extracted from an invention.

    id      : "F1", "F2", … — stable reference for the feature matrix
    feature : plain-text description of the feature
    category: component | method | use | composition | ratio | other
    """
    id: str = Field(description="Stable ID: F1, F2, F3 …")
    feature: str = Field(description="Plain-text feature description")
    category: str = Field(
        default="component",
        description="component | method | use | composition | ratio | structural | other"
    )


class ClaimRepresentation(BaseModel):
    """
    Claim-like technical representation of the invention.
    Not a legally drafted patent claim — a structured technical summary
    used for prior-art comparison.
    """
    claim_type: str = Field(
        description="composition | method | device | system | process | use"
    )
    elements: List[str] = Field(
        description="Technical elements that define the invention"
    )
    independent_claim: str = Field(
        description="Human-readable independent claim draft"
    )
    ipc_suggested: List[str] = Field(
        default_factory=list,
        description="Suggested IPC classification codes"
    )


class FeatureMatch(BaseModel):
    """
    Result of matching one InventionFeature against one prior-art document.
    """
    feature_id: str
    feature_text: str
    matched: bool
    match_type: Literal["exact", "semantic", "no_match"] = "no_match"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: Optional[str] = None       # supporting excerpt from document
    evidence_page: Optional[int] = None
    notes: Optional[str] = None               # e.g. "Azadirachta indica = neem"


class PriorArtCandidate(BaseModel):
    """
    A candidate prior-art document with its per-feature match matrix.
    """
    document_id: str
    publication_number: str
    title: str
    source: str
    filing_date: Optional[str] = None
    publication_date: Optional[str] = None
    classification: Optional[str] = None
    ipc_codes: List[str] = Field(default_factory=list)
    chunk_id: str

    # Feature matrix: feature_id → FeatureMatch
    feature_matches: Dict[str, FeatureMatch] = Field(default_factory=dict)

    # Aggregate scores
    matched_count: int = 0            # how many features matched
    total_features: int = 0           # total features in invention
    overall_similarity: float = 0.0   # weighted score

    def coverage(self) -> float:
        """Fraction of invention features this document covers."""
        if self.total_features == 0:
            return 0.0
        return self.matched_count / self.total_features

    def matched_feature_ids(self) -> List[str]:
        return [fid for fid, fm in self.feature_matches.items() if fm.matched]

    def to_summary(self) -> dict:
        return {
            "document_id"        : self.document_id,
            "publication_number" : self.publication_number,
            "title"              : self.title,
            "source"             : self.source,
            "filing_date"        : self.filing_date,
            "publication_date"   : self.publication_date,
            "classification"     : self.classification,
            "ipc_codes"          : self.ipc_codes,
            "matched_features"   : self.matched_feature_ids(),
            "coverage"           : round(self.coverage(), 3),
            "overall_similarity" : round(self.overall_similarity, 4),
            "feature_detail"     : {
                fid: {
                    "matched"     : fm.matched,
                    "match_type"  : fm.match_type,
                    "confidence"  : fm.confidence,
                    "evidence"    : (fm.evidence_text or "")[:300],
                }
                for fid, fm in self.feature_matches.items()
            },
        }
