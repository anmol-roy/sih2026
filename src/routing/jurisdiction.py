"""
Jurisdiction Router  =
────────────────────────────────
Classifies a query into one of three jurisdiction modes:
  india         — Indian law, IP India, India Code, AYUSH, TKDL
  international — WIPO, PCT, TRIPS, Paris Convention, Madrid System
  both          — explicit comparison between India and international
  unknown       — cannot determine

Two-layer approach:
  1. Keyword classification (instant, no LLM call)
  2. LLM classifier (accurate, used when keywords are ambiguous)

User-supplied jurisdiction always overrides classification.
"""

from __future__ import annotations

import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class Jurisdiction(str, Enum):
    INDIA         = "india"
    INTERNATIONAL = "international"
    BOTH          = "both"
    UNKNOWN       = "unknown"


class JurisdictionRoute(BaseModel):
    jurisdiction: Jurisdiction
    confidence  : float = Field(ge=0.0, le=1.0)
    reason      : str


# ─────────────────────────────────────────────────────────────────────────────
# Keyword patterns
# ─────────────────────────────────────────────────────────────────────────────

# Strong India indicators
_INDIA_RE = re.compile(
    r"""
    \bIndia(?:n)?\b              |  # Indian / India
    \bIP\s*India\b               |  # IP India
    India\s*Code                 |  # India Code
    Patents?\s*Act\s*1970        |  # Patents Act 1970
    Trade\s*Marks?\s*Act\s*1999  |  # Trade Marks Act
    Copyright\s*Act\s*1957       |  # Copyright Act 1957
    Designs?\s*Act\s*2000        |  # Designs Act 2000
    Section\s*\d+                |  # Section X (any section = Indian law context)
    \bAYUSH\b                    |  # AYUSH
    \bTKDL\b                     |  # TKDL
    \bIPO\b                      |  # Indian Patent Office
    India\s*Code                 |  # India Code
    \bDGFT\b                     |  # DGFT
    Geographical\s*Indication.*India
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Strong international indicators
_INTL_RE = re.compile(
    r"""
    \bWIPO\b                     |  # WIPO
    \bPCT\b                      |  # PCT
    \bTRIPS\b                    |  # TRIPS
    Paris\s*Convention           |  # Paris Convention
    Madrid\s*(?:Protocol|System) |  # Madrid System
    Berne\s*Convention           |  # Berne Convention
    Hague\s*(?:Agreement|System) |  # Hague System
    international\s*(?:patent|trademark|copyright|treaty|agreement|IP|filing|geographical) |
    \bUNCTAD\b                   |
    \bEPO\b                      |  # EPO
    \bUSPTO\b                    |  # USPTO
    multilateral\s*treaty        |
    \bPCT\s*application          |
    Patent\s*Cooperation\s*Treaty
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Explicit comparison / "both" indicators
_BOTH_RE = re.compile(
    r"""
    compare|comparison|versus|vs\.?|differ|difference|
    India.*international|international.*India|
    India.*PCT|PCT.*India|
    India.*TRIPS|TRIPS.*India|
    India.*WIPO|WIPO.*India|
    national.*international|international.*national
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _keyword_classify(query: str) -> JurisdictionRoute:
    """
    Fast keyword-based jurisdiction classification.
    Returns a JurisdictionRoute with confidence ≤ 0.85.
    """
    has_india = bool(_INDIA_RE.search(query))
    has_intl  = bool(_INTL_RE.search(query))
    has_both  = bool(_BOTH_RE.search(query))

    # "both" only fires when the comparison is genuinely between India AND international
    if has_both and has_india and has_intl:
        return JurisdictionRoute(
            jurisdiction=Jurisdiction.BOTH,
            confidence=0.80,
            reason="Query contains explicit comparison keywords with both India and international references.",
        )
    if has_india and has_intl:
        return JurisdictionRoute(
            jurisdiction=Jurisdiction.BOTH,
            confidence=0.72,
            reason="Query references both Indian law and international systems.",
        )
    if has_india:
        return JurisdictionRoute(
            jurisdiction=Jurisdiction.INDIA,
            confidence=0.85,
            reason="Query contains Indian law / IP India / AYUSH / TKDL keywords.",
        )
    if has_intl:
        return JurisdictionRoute(
            jurisdiction=Jurisdiction.INTERNATIONAL,
            confidence=0.85,
            reason="Query contains WIPO / PCT / TRIPS / Paris Convention keywords.",
        )

    # No strong signal — default to India
    return JurisdictionRoute(
        jurisdiction=Jurisdiction.INDIA,
        confidence=0.50,
        reason="No strong jurisdiction signal detected; defaulting to India.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an IP jurisdiction classification expert.

Classify this intellectual-property query into exactly one jurisdiction:

  india         — The question specifically concerns Indian law, Indian Patent
                  Office, India Code, IP India, AYUSH, TKDL, Indian trademarks,
                  Indian GI, Indian Copyright Act, Indian traditional knowledge,
                  or Indian ABS requirements.

  international — The question specifically concerns WIPO, PCT, TRIPS Agreement,
                  Paris Convention, Madrid System, Berne Convention, Hague System,
                  international IP treaties, or international filing systems.

  both          — The user explicitly asks to compare India with international,
                  or asks a question that requires citing both Indian law and
                  an international instrument.

  unknown       — Cannot determine from the query.

Return ONLY a JSON object with EXACTLY these keys:
{
  "jurisdiction": "india" | "international" | "both" | "unknown",
  "confidence"  : float between 0 and 1,
  "reason"      : string — one sentence explanation
}

Rules:
- Output ONLY valid JSON. No markdown. No explanation outside JSON.
- When in doubt between india and unknown, prefer india.
- Use "both" ONLY when the user explicitly requests a comparison OR
  when the query cannot be answered without citing both jurisdictions.
"""

_USER = "Query: {query}\n\nJSON:"


# ─────────────────────────────────────────────────────────────────────────────
# Jurisdiction Router
# ─────────────────────────────────────────────────────────────────────────────

class JurisdictionRouter:
    """
    Classifies a query's jurisdiction using keyword heuristics + optional LLM.

    Parameters
    ----------
    llm              : optional shared ChatGroq instance
    use_llm_threshold: keyword confidence below this value triggers LLM fallback
    """

    def __init__(
        self,
        llm: Optional[ChatGroq] = None,
        use_llm_threshold: float = 0.60,
    ):
        self._llm       = llm
        self._threshold = use_llm_threshold

    def classify(
        self,
        query: str,
        override: Optional[str] = None,
    ) -> JurisdictionRoute:
        """
        Classify jurisdiction of *query*.

        Parameters
        ----------
        query    : user query text
        override : explicit jurisdiction string from API request
                   ("india", "international", "both") — overrides all classification

        Returns a JurisdictionRoute.
        """
        # ── 1. Explicit user override ─────────────────────────────────────
        if override:
            jur = self._coerce(override)
            if jur != Jurisdiction.UNKNOWN:
                return JurisdictionRoute(
                    jurisdiction=jur,
                    confidence=1.0,
                    reason="User-supplied explicit jurisdiction override.",
                )

        # ── 2. Keyword classification ─────────────────────────────────────
        kw_result = _keyword_classify(query)

        # If confidence is high enough, use it directly
        if kw_result.confidence >= self._threshold:
            return kw_result

        # ── 3. LLM fallback ───────────────────────────────────────────────
        if self._llm is None:
            return kw_result   # no LLM available, return keyword result

        try:
            raw = self._llm.invoke([
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER.format(query=query.strip())},
            ]).content.strip()

            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            jur        = self._coerce(data.get("jurisdiction", "unknown"))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
            reason     = str(data.get("reason", "LLM classification"))

            return JurisdictionRoute(
                jurisdiction=jur,
                confidence=confidence,
                reason=reason,
            )
        except Exception:
            return kw_result   # LLM failed, fall back to keywords

    @staticmethod
    def _coerce(val: str) -> Jurisdiction:
        val = val.strip().lower()
        try:
            return Jurisdiction(val)
        except ValueError:
            return Jurisdiction.UNKNOWN
