"""
Legal document chunk schema.
Every piece of text stored in Qdrant must conform to this model.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class LegalChunk(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id: str = Field(
        description="Unique ID, e.g. 'patents_act_1970_ch2_sec3_p_page8'"
    )
    document_id: str = Field(
        description="Stable machine identifier, e.g. 'patents_act_1970'"
    )

    # ── Human labels ──────────────────────────────────────────────────────────
    title: str = Field(description="Full document title")
    source: str = Field(description="Publishing body, e.g. 'India Code'")
    source_url: Optional[str] = None

    # ── Classification ────────────────────────────────────────────────────────
    document_type: str = Field(
        description="act | rule | regulation | notification | guideline | form"
    )
    domain: str = Field(
        description="patent | trademark | copyright | design | gi | ayush | general"
    )
    authority_level: str = Field(
        description="primary | secondary | tertiary"
    )

    # ── Position inside document ──────────────────────────────────────────────
    chapter: Optional[str] = None          # e.g. "Chapter II"
    section: Optional[str] = None          # e.g. "Section 3"
    subsection: Optional[str] = None       # e.g. "3(p)"
    page: Optional[int] = None

    # ── Versioning ────────────────────────────────────────────────────────────
    language: str = "english"
    version: Optional[str] = None          # e.g. "as amended up to 2023"
    effective_date: Optional[str] = None   # ISO-8601 date string
    last_verified: str = "2026-08-25"

    # ── Content ───────────────────────────────────────────────────────────────
    text: str

    # ── Helpers ───────────────────────────────────────────────────────────────
    def to_metadata(self) -> dict:
        """Return a flat dict suitable for Qdrant payload (no None values)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def citation_label(self) -> str:
        """Short human-readable citation string."""
        parts = [self.title]
        if self.section:
            parts.append(self.section)
        if self.subsection:
            parts.append(f"§ {self.subsection}")
        if self.page is not None:
            parts.append(f"p. {self.page}")
        parts.append(f"({self.source})")
        return ", ".join(parts)
