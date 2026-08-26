"""
IP Routing Schemas  (Phase 5)
──────────────────────────────
Defines the IP type taxonomy and the route object returned by the
IP router. Supports multi-domain routing for queries that span
more than one IP type.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class IPType(str, Enum):
    PATENT               = "patent"
    TRADEMARK            = "trademark"
    COPYRIGHT            = "copyright"
    DESIGN               = "design"
    GI                   = "gi"
    TRADITIONAL_KNOWLEDGE = "traditional_knowledge"
    UNKNOWN              = "unknown"


# Human-readable labels and Qdrant domain filter values
IP_TYPE_META: dict[IPType, dict] = {
    IPType.PATENT: {
        "label"       : "Patent",
        "domain_filter": "patent",
        "description" : "Inventions, technical solutions, patentability, novelty, prior art",
    },
    IPType.TRADEMARK: {
        "label"       : "Trademark",
        "domain_filter": "trademark",
        "description" : "Brand names, logos, marks, slogans, registration, infringement",
    },
    IPType.COPYRIGHT: {
        "label"       : "Copyright",
        "domain_filter": "copyright",
        "description" : "Books, software, music, films, artistic works, ownership",
    },
    IPType.DESIGN: {
        "label"       : "Design",
        "domain_filter": "design",
        "description" : "Visual appearance, shape, pattern, ornamental features",
    },
    IPType.GI: {
        "label"       : "Geographical Indication",
        "domain_filter": "gi",
        "description" : "Products tied to geographical regions, GI registration",
    },
    IPType.TRADITIONAL_KNOWLEDGE: {
        "label"       : "Traditional Knowledge / AYUSH",
        "domain_filter": "ayush",
        "description" : "AYUSH, Ayurveda, Siddha, Unani, traditional formulations, TKDL",
    },
    IPType.UNKNOWN: {
        "label"       : "Unknown",
        "domain_filter": None,
        "description" : "Could not determine IP type",
    },
}


class QueryRoute(BaseModel):
    """Result of classifying a user query into IP domain(s)."""

    ip_types   : List[IPType] = Field(
        description="One or more IP types relevant to this query (multi-domain supported)"
    )
    primary    : IPType = Field(
        description="The single most relevant IP type"
    )
    confidence : float = Field(ge=0.0, le=1.0)
    reason     : str   = Field(description="Short explanation of the classification")
    is_multi   : bool  = Field(
        default=False,
        description="True when the query spans more than one IP domain"
    )

    @property
    def domain_filters(self) -> List[Optional[str]]:
        """Qdrant domain filter values for all matched IP types."""
        return [
            IP_TYPE_META[t]["domain_filter"]
            for t in self.ip_types
            if IP_TYPE_META[t]["domain_filter"] is not None
        ]
