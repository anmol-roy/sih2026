"""
Formulation Classification Schemas  (Phase 5)
───────────────────────────────────────────────
Defines the formulation type taxonomy and the output of the
FormulationClassifier.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FormulationType(str, Enum):
    AYURVEDA              = "ayurveda"
    SIDDHA                = "siddha"
    UNANI                 = "unani"
    YOGA                  = "yoga"
    MODERN_PHARMACEUTICAL = "modern_pharmaceutical"
    TRADITIONAL_KNOWLEDGE = "traditional_knowledge"   # generic TK, not system-specific
    BIOLOGICAL_RESOURCE   = "biological_resource"     # plant/animal/microbial source
    MIXED                 = "mixed"                   # spans multiple systems
    UNKNOWN               = "unknown"


# Short description for each type
FORMULATION_TYPE_DESC: dict[FormulationType, str] = {
    FormulationType.AYURVEDA             : "Classical Ayurvedic formulation based on documented texts (Charaka, Sushruta, etc.)",
    FormulationType.SIDDHA               : "Siddha system of medicine, primarily from Tamil Nadu",
    FormulationType.UNANI                : "Unani (Greco-Arabic) system of medicine",
    FormulationType.YOGA                 : "Yoga or naturopathy-related practice or preparation",
    FormulationType.MODERN_PHARMACEUTICAL: "Synthetic or semi-synthetic pharmaceutical compound or formulation",
    FormulationType.TRADITIONAL_KNOWLEDGE: "Traditional knowledge not specific to one AYUSH system",
    FormulationType.BIOLOGICAL_RESOURCE  : "Formulation primarily based on plant, animal, or microbial resources",
    FormulationType.MIXED                : "Formulation spanning multiple systems or types",
    FormulationType.UNKNOWN              : "Cannot determine formulation type from the description",
}


class FormulationClassification(BaseModel):
    """
    Complete classification of a formulation description.

    formulation_type         : primary type
    secondary_types          : any additional applicable types
    ingredients              : all identified ingredients
    biological_resources     : subset of ingredients that are biological (plant/animal/microbial)
    traditional_knowledge_indicators : text cues suggesting TK origin
    tk_systems               : specific AYUSH systems identified (ayurveda, siddha, unani, yoga)
    confidence               : 0.0 – 1.0
    notes                    : any additional classifier notes
    """

    formulation_type                : FormulationType
    secondary_types                 : List[FormulationType]  = Field(default_factory=list)
    ingredients                     : List[str]              = Field(default_factory=list)
    biological_resources            : List[str]              = Field(default_factory=list)
    traditional_knowledge_indicators: List[str]              = Field(default_factory=list)
    tk_systems                      : List[str]              = Field(default_factory=list)
    confidence                      : float                  = Field(ge=0.0, le=1.0, default=0.0)
    notes                           : Optional[str]          = None

    def is_tk_relevant(self) -> bool:
        """True if this formulation has any traditional-knowledge relevance."""
        tk_types = {
            FormulationType.AYURVEDA,
            FormulationType.SIDDHA,
            FormulationType.UNANI,
            FormulationType.YOGA,
            FormulationType.TRADITIONAL_KNOWLEDGE,
        }
        return (
            self.formulation_type in tk_types
            or any(t in tk_types for t in self.secondary_types)
            or bool(self.traditional_knowledge_indicators)
        )

    def ip_domain_hints(self) -> List[str]:
        """
        Suggest which IP domains are likely relevant for this formulation.
        Consumed by the orchestrator to widen the search scope.
        """
        hints = ["patent"]   # always check patent for any formulation
        if self.is_tk_relevant():
            hints.append("traditional_knowledge")
            hints.append("ayush")
        if self.biological_resources:
            hints.append("biological_resource")
        return hints
