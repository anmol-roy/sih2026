"""
Formulation Schemas  (Phase 5 + 7)
────────────────────────────────────
FormulationType           — taxonomy enum
Ingredient                — one ingredient with scientific name + flags
FormulationClassification — full classification result (Phase 5 + 7)
ABSAssessment             — Access and Benefit Sharing relevance result
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# FormulationType
# ─────────────────────────────────────────────────────────────────────────────

class FormulationType(str, Enum):
    AYURVEDA              = "ayurveda"
    SIDDHA                = "siddha"
    UNANI                 = "unani"
    YOGA                  = "yoga"
    TRADITIONAL_KNOWLEDGE = "traditional_knowledge"
    MODERN_PHARMACEUTICAL = "modern_pharmaceutical"
    BIOLOGICAL_RESOURCE   = "biological_resource"
    MIXED                 = "mixed"
    UNKNOWN               = "unknown"


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


# ─────────────────────────────────────────────────────────────────────────────
# Ingredient  (Phase 7 — enriched)
# ─────────────────────────────────────────────────────────────────────────────

class Ingredient(BaseModel):
    """One ingredient identified in the formulation description."""

    name                    : str
    scientific_name         : Optional[str]  = None
    biological_resource     : bool           = False   # plant / animal / microbial
    traditional_use_indicator: bool          = False   # appears in TK texts
    traditional_systems     : List[str]      = Field(
        default_factory=list,
        description="AYUSH systems where this ingredient is documented: ayurveda, siddha, unani, yoga"
    )
    source                  : str            = "extracted"  # "extracted" | "db_enriched"


# ─────────────────────────────────────────────────────────────────────────────
# FormulationClassification  (extended for Phase 7)
# ─────────────────────────────────────────────────────────────────────────────

class FormulationClassification(BaseModel):
    """Complete classification of a formulation description."""

    formulation_type                : FormulationType
    secondary_types                 : List[FormulationType] = Field(default_factory=list)

    # Phase 5 — string lists (kept for backward compat with orchestrator)
    ingredients                     : List[str]             = Field(default_factory=list)
    biological_resources            : List[str]             = Field(default_factory=list)
    traditional_knowledge_indicators: List[str]             = Field(default_factory=list)
    tk_systems                      : List[str]             = Field(default_factory=list)

    # Phase 7 — enriched ingredient objects
    ingredient_objects              : List[Ingredient]      = Field(default_factory=list)

    confidence                      : float                 = Field(ge=0.0, le=1.0, default=0.0)
    notes                           : Optional[str]         = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def is_tk_relevant(self) -> bool:
        tk_types = {
            FormulationType.AYURVEDA, FormulationType.SIDDHA,
            FormulationType.UNANI,    FormulationType.YOGA,
            FormulationType.TRADITIONAL_KNOWLEDGE,
        }
        return (
            self.formulation_type in tk_types
            or any(t in tk_types for t in self.secondary_types)
            or bool(self.traditional_knowledge_indicators)
            or any(i.traditional_use_indicator for i in self.ingredient_objects)
        )

    def has_biological_resources(self) -> bool:
        return bool(self.biological_resources) or any(
            i.biological_resource for i in self.ingredient_objects
        )

    def ip_domain_hints(self) -> List[str]:
        hints = ["patent"]
        if self.is_tk_relevant():
            hints.extend(["traditional_knowledge", "ayush"])
        if self.has_biological_resources():
            hints.append("biological_resource")
        return hints

    def all_bio_resource_names(self) -> List[str]:
        """Return deduplicated list of biological-resource ingredient names."""
        names: list[str] = list(self.biological_resources)
        for ing in self.ingredient_objects:
            if ing.biological_resource and ing.name not in names:
                names.append(ing.name)
        return names


# ─────────────────────────────────────────────────────────────────────────────
# ABSAssessment  (Phase 7)
# ─────────────────────────────────────────────────────────────────────────────

class ABSAssessment(BaseModel):
    """
    Access and Benefit Sharing relevance assessment.

    This is a preliminary, evidence-based finding — NOT legal clearance.
    All results marked requires_human_review = True must be reviewed by
    a qualified professional.
    """

    potentially_relevant            : bool
    biological_resources            : List[str]  = Field(default_factory=list)
    traditional_knowledge_detected  : bool       = False
    reasons                         : List[str]  = Field(default_factory=list)
    relevant_sources                : List[str]  = Field(default_factory=list)
    requires_human_review           : bool       = True

    # Suggested legal provisions to check
    suggested_provisions            : List[str]  = Field(
        default_factory=list,
        description="Indian legal provisions potentially relevant to ABS"
    )
