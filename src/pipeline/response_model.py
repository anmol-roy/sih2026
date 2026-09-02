"""
Pipeline Response Model  (Phase 11)
──────────────────────────────────────
Single unified response schema returned by process_query().

Every endpoint in the API wraps its output into this format so the
frontend always receives the same contract regardless of which
internal path was taken.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field

class CitationRecord(BaseModel):    
    """A verified, backend-constructed citation. Never LLM-invented."""
    source_id   : str
    document    : str
    section     : Optional[str] = None
    subsection  : Optional[str] = None
    page        : Optional[int] = None
    source_name : str
    jurisdiction: str  = "india"
    authority   : str  = "primary"
    chunk_id    : str  = ""
    verified    : bool = True   # False if citation could not be verified


class ConflictRecord(BaseModel):
    """Evidence conflict detected between two sources."""
    source_a    : str
    source_b    : str
    description : str
    resolution  : str  = "prefer_latest_version"


class PipelineResponse(BaseModel):
    """
    Universal response from process_query().

    status values:
      answered      — sufficient evidence, answer generated
      abstained     — insufficient evidence
      escalated     — low confidence or user requested human
      out_of_scope  — query not related to IP/TK/ABS
    """

    # Core result
    status              : str               # answered | abstained | escalated | out_of_scope
    answer              : Optional[str]     = None
    answer_english      : Optional[str]     = None  # original English before translation

    # Routing metadata
    language            : str               = "en"
    original_question   : str               = ""
    normalized_question : str               = ""    # English version
    ip_types            : List[str]         = Field(default_factory=list)
    jurisdiction        : str               = "india"

    # Citations (backend-constructed, never LLM-invented)
    citations           : List[CitationRecord] = Field(default_factory=list)
    citation_coverage   : float             = 0.0   # fraction of claims with evidence

    # Confidence
    confidence          : float             = 0.0
    confidence_band     : str               = "low"  # high | medium | low
    needs_human_review  : bool              = False
    reason              : Optional[str]     = None

    # Evidence metadata
    tools_used          : List[str]         = Field(default_factory=list)
    sources_consulted   : int               = 0
    issues              : List[str]         = Field(default_factory=list)
    conflicts           : List[ConflictRecord] = Field(default_factory=list)

    # Formulation (if detected)
    formulation         : Optional[dict]    = None

    # Disclaimer (always present — never LLM-generated)
    disclaimer          : str               = ""

    # Request tracking
    request_id          : Optional[str]     = None
