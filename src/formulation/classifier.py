"""
Formulation Classifier  (Phase 5)
───────────────────────────────────
Classifies a formulation description into a FormulationType and extracts:
  - ingredients
  - biological resources
  - traditional knowledge indicators
  - specific AYUSH systems

Uses an LLM with structured JSON output.
Falls back to keyword heuristics if the LLM returns invalid JSON.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from formulation.schemas import FormulationClassification, FormulationType


# ─────────────────────────────────────────────────────────────────────────────
# Keyword heuristics (fallback)
# ─────────────────────────────────────────────────────────────────────────────

_AYURVEDA_KW  = re.compile(
    r"\bayurved|\bcharaka\b|\bsushruta\b|\bvagbhata\b|chyawanprash|triphala|"
    r"\bneem\b|\bturmeric\b|\bashwagandha\b|\bamla\b|\bbrahmi\b|"
    r"\bgiloy\b|\btulsi\b|\bharitaki\b|\bneem\s+extract\b", re.I
)
_SIDDHA_KW    = re.compile(r"\bsiddha\b|\btamil\s+medicine\b|\bkaya\s+kalpa\b", re.I)
_UNANI_KW     = re.compile(r"\bunani\b|\bgreek\s+medicine\b|\barabi\b|\bhakeem\b", re.I)
_YOGA_KW      = re.compile(r"\byoga\b|\bnaturopath\b|\bpranayama\b", re.I)
_MODERN_KW    = re.compile(
    r"\bsynthetic\b|\bchemical\s+compound\b|\bpharmaceutical\b|"
    r"\bdrug\b|\bAPI\b|\bcrystalline\b|\bpolymorph\b|\bsalt\s+form\b|"
    r"\bibuprofen\b|\bmetformin\b|\bgefitinib\b|\bpenicillin\b", re.I
)
_BIO_KW       = re.compile(
    r"\bplant\s+extract\b|\bherbal\b|\bbotanical\b|\bmicroorganism\b|"
    r"\bbacterium\b|\bfungus\b|\bbiological\b|\bstrain\b", re.I
)
_TK_KW        = re.compile(
    r"\btraditional\s+knowledge\b|\btraditional\s+medicine\b|"
    r"\bfolk\s+medicine\b|\bindigenous\b|\btkdl\b|\btraditional\s+use\b", re.I
)

# Common Indian medicinal plant names → biological resources
_PLANT_NAMES  = {
    "neem", "turmeric", "ashwagandha", "amla", "brahmi", "giloy", "tulsi",
    "haritaki", "bibhitaki", "shankhpushpi", "triphala", "chyawanprash",
    "aloe vera", "ginger", "garlic", "curcumin", "andrographolide",
    "boswellic acid", "withanolide", "neem extract", "turmeric extract",
    "bitter melon", "momordica charantia", "tinospora cordifolia",
    "withania somnifera", "azadirachta indica", "ocimum sanctum",
    "bacopa monnieri", "phyllanthus emblica", "curcuma longa",
    "andrographis paniculata", "boswellia serrata",
}


def _keyword_classify(text: str) -> FormulationClassification:
    """Heuristic fallback when LLM returns invalid JSON."""
    tl = text.lower()

    ftype = FormulationType.UNKNOWN
    secondary = []
    systems   = []

    if _MODERN_KW.search(text):
        ftype = FormulationType.MODERN_PHARMACEUTICAL
    if _AYURVEDA_KW.search(text):
        if ftype == FormulationType.UNKNOWN:
            ftype = FormulationType.AYURVEDA
        else:
            secondary.append(FormulationType.AYURVEDA)
        systems.append("ayurveda")
    if _SIDDHA_KW.search(text):
        if ftype == FormulationType.UNKNOWN:
            ftype = FormulationType.SIDDHA
        else:
            secondary.append(FormulationType.SIDDHA)
        systems.append("siddha")
    if _UNANI_KW.search(text):
        if ftype == FormulationType.UNKNOWN:
            ftype = FormulationType.UNANI
        else:
            secondary.append(FormulationType.UNANI)
        systems.append("unani")
    if _YOGA_KW.search(text):
        if ftype == FormulationType.UNKNOWN:
            ftype = FormulationType.YOGA
        else:
            secondary.append(FormulationType.YOGA)
        systems.append("yoga")
    if _TK_KW.search(text) and ftype == FormulationType.UNKNOWN:
        ftype = FormulationType.TRADITIONAL_KNOWLEDGE
    if _BIO_KW.search(text) and ftype == FormulationType.UNKNOWN:
        ftype = FormulationType.BIOLOGICAL_RESOURCE

    if len(secondary) >= 1 and ftype != FormulationType.UNKNOWN:
        ftype = FormulationType.MIXED

    # Extract ingredient names
    ingredients = [p for p in _PLANT_NAMES if p in tl]
    bio         = ingredients[:]   # all plants are bio resources
    tk_inds     = []
    if _TK_KW.search(text):
        tk_inds.append("traditional use mentioned")
    if systems:
        tk_inds.extend([f"{s} system identified" for s in systems])

    return FormulationClassification(
        formulation_type                = ftype,
        secondary_types                 = list(dict.fromkeys(secondary)),
        ingredients                     = list(dict.fromkeys(ingredients)),
        biological_resources            = list(dict.fromkeys(bio)),
        traditional_knowledge_indicators= tk_inds,
        tk_systems                      = list(dict.fromkeys(systems)),
        confidence                      = 0.55,
        notes                           = "Keyword-based fallback classification",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert in Indian traditional medicine systems and pharmaceutical classification.

Given a formulation description, classify it and extract structured information.
Return ONLY a valid JSON object with EXACTLY these keys:

{
  "formulation_type": one of:
      ayurveda | siddha | unani | yoga |
      modern_pharmaceutical | traditional_knowledge |
      biological_resource | mixed | unknown,

  "secondary_types": [string]  — other applicable types (may be empty list),

  "ingredients": [string]      — ALL ingredients/components mentioned,

  "biological_resources": [string] — subset of ingredients that are plant,
                                     animal, or microbial biological resources,

  "traditional_knowledge_indicators": [string] — phrases or cues suggesting
                                                  traditional knowledge origin,

  "tk_systems": [string]       — specific AYUSH systems: ayurveda, siddha,
                                  unani, yoga (may be empty),

  "confidence": float          — 0.0 to 1.0,

  "notes": string | null       — any important observations
}

Definitions:
  ayurveda              — Uses herbs/plants documented in classical Ayurvedic texts.
  siddha                — Tamil medicine system.
  unani                 — Greco-Arabic medicine system.
  yoga                  — Yoga/naturopathy practice or preparation.
  modern_pharmaceutical — Synthetic, semi-synthetic, or chemically defined compound.
  traditional_knowledge — Generic TK not specific to one AYUSH system.
  biological_resource   — Primarily plant, animal, or microbial resources.
  mixed                 — Spans multiple systems or types.
  unknown               — Cannot determine from description.

Rules:
- Output ONLY valid JSON. No markdown. No explanation outside JSON.
- ingredients must include ALL components mentioned (even chemical ones).
- biological_resources are only plant/animal/microbial sources.
- If the formulation is both Ayurvedic AND modern, use "mixed" as primary.
"""

_USER = "Formulation description:\n\n{text}\n\nJSON:"


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class FormulationClassifier:
    """
    Classify a formulation description into a FormulationClassification.

    Parameters
    ----------
    llm : optional shared ChatGroq instance
    """

    def __init__(self, llm: Optional[ChatGroq] = None):
        self._llm = llm or ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

    def classify(self, description: str) -> FormulationClassification:
        """
        Classify *description* into a FormulationClassification.
        Falls back to keyword heuristics on any LLM/JSON error.
        """
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER.format(text=description.strip())},
        ]

        try:
            raw = self._llm.invoke(messages).content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            m = re.search(r"\{.*\}", (raw if 'raw' in dir() else ""), re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return _keyword_classify(description)
            else:
                return _keyword_classify(description)

        # ── Coerce / validate ──────────────────────────────────────────────
        def _coerce_type(v: str) -> FormulationType:
            v = str(v).strip().lower()
            try:
                return FormulationType(v)
            except ValueError:
                return FormulationType.UNKNOWN

        def _safe_list(v) -> list[str]:
            if isinstance(v, list):
                return [str(x).strip() for x in v if x]
            if isinstance(v, str) and v:
                return [v]
            return []

        ftype       = _coerce_type(data.get("formulation_type", "unknown"))
        secondary   = [_coerce_type(t) for t in _safe_list(data.get("secondary_types", []))]
        secondary   = [t for t in secondary if t != FormulationType.UNKNOWN and t != ftype]
        confidence  = max(0.0, min(1.0, float(data.get("confidence", 0.7))))

        return FormulationClassification(
            formulation_type                = ftype,
            secondary_types                 = list(dict.fromkeys(secondary)),
            ingredients                     = _safe_list(data.get("ingredients", [])),
            biological_resources            = _safe_list(data.get("biological_resources", [])),
            traditional_knowledge_indicators= _safe_list(data.get("traditional_knowledge_indicators", [])),
            tk_systems                      = _safe_list(data.get("tk_systems", [])),
            confidence                      = confidence,
            notes                           = data.get("notes") or None,
        )
