"""
Scope Classifier  (Phase 9)
────────────────────────────
Determines whether a user query is in scope for IP-SAKTI.

Two-layer approach (same pattern as Phase 5-6 routers):
  1. Keyword check  — fast, deterministic, no LLM cost
  2. LLM classifier — accurate for ambiguous queries

In-scope domains:
  patents, trademarks, copyright, designs, GI,
  traditional knowledge, AYUSH, ABS, IP treaties,
  IP registration, IP compliance, TKDL, prior art
"""

from __future__ import annotations

import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class Scope(str, Enum):
    IN_SCOPE     = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    UNCERTAIN    = "uncertain"


class ScopeResult(BaseModel):
    scope     : Scope
    confidence: float = Field(ge=0.0, le=1.0)
    reason    : str


# ─────────────────────────────────────────────────────────────────────────────
# Keyword lists
# ─────────────────────────────────────────────────────────────────────────────

_IN_SCOPE_KW = re.compile(
    r"""
    \bpatent\b           | \btrademark\b      | \bcopyright\b     |
    \bdesign\b           | \bgeographical\b   | \bGI\b            |
    \bprior\s*art\b      | \btraditional\s*knowledge\b             |
    \bTKDL\b             | \bAYUSH\b          | \bABS\b           |
    \bintellectual\s*property\b               | \bIP\b            |
    \bpatentab\b         | \bnovelit\b        | \binventi\b       |
    \btrade\s*mark\b     | \bregistr\b        | \blicense\b       |
    \bnagoya\b           | \bbiodiversity\b   | \bPCT\b           |
    \bTRIPS\b            | \bParis\s*Convention\b                  |
    \bWIPO\b             | \bsection\s*3\b    | \bformulation\b   |
    \bherbal\b           | \bayurved\b        | \bsiddha\b        |
    \bunani\b            | \btraditional\s*med\b                   |
    \bIP\s*India\b       | \bIndia\s*Code\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_OUT_OF_SCOPE_STRONG = re.compile(
    r"""
    \bcricket\b          | \bfootball\b       | \bsoccer\b        |
    \bweather\b          | \brecipe\b         | \bcooking\b       |
    \bstock\s*price\b    | \bshare\s*price\b  | \bcryptocurrenc\b |
    \bcelebrit\b         | \bgossip\b         | \bmovie\b         |
    \bsong\b             | \blyric\b          | \blove\b          |
    \brelationship\b     | \bhoroscope\b      | \bastrology\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _keyword_scope(query: str) -> Optional[ScopeResult]:
    """
    Fast keyword-based scope check.
    Returns ScopeResult if confident, None if ambiguous.
    """
    if _OUT_OF_SCOPE_STRONG.search(query):
        return ScopeResult(
            scope=Scope.OUT_OF_SCOPE,
            confidence=0.92,
            reason="Query contains strong out-of-scope keywords.",
        )
    if _IN_SCOPE_KW.search(query):
        return ScopeResult(
            scope=Scope.IN_SCOPE,
            confidence=0.88,
            reason="Query contains IP / TK / AYUSH / ABS keywords.",
        )
    return None   # ambiguous — use LLM


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an IP-SAKTI scope classifier.

Classify whether this user query is within scope for an
Indian Intellectual Property and Traditional Knowledge assistant.

IN_SCOPE: questions about patents, trademarks, copyright, designs,
  geographical indications, traditional knowledge, AYUSH, ABS, IP treaties,
  IP registration, IP compliance, formulations, prior art, Section 3 exclusions,
  TKDL, WIPO, PCT, TRIPS, Paris Convention, biodiversity, Nagoya Protocol.

OUT_OF_SCOPE: questions unrelated to intellectual property, traditional
  knowledge, ABS, or supported legal domains (e.g. sports results, cooking,
  weather, general news, celebrity gossip, financial markets).

UNCERTAIN: borderline or ambiguous queries.

Return ONLY a JSON object:
{
  "scope"     : "in_scope" | "out_of_scope" | "uncertain",
  "confidence": float between 0 and 1,
  "reason"    : one-sentence explanation
}
No markdown. No explanation outside the JSON.
"""

_USER = "Query: {query}\n\nJSON:"


# ─────────────────────────────────────────────────────────────────────────────
# ScopeChecker
# ─────────────────────────────────────────────────────────────────────────────

class ScopeChecker:
    """
    Two-layer scope classification.

    Parameters
    ----------
    llm              : optional shared ChatGroq
    use_llm_threshold: keyword confidence below this triggers LLM
    """

    def __init__(self, llm=None, use_llm_threshold: float = 0.60):
        self._llm       = llm
        self._threshold = use_llm_threshold

    def check(self, query: str) -> ScopeResult:
        # 1. Keyword check
        kw = _keyword_scope(query)
        if kw and kw.confidence >= self._threshold:
            return kw

        # 2. LLM
        if self._llm:
            try:
                raw = self._llm.invoke([
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": _USER.format(query=query.strip())},
                ]).content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                scope = Scope(data.get("scope", "uncertain"))
                conf  = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                return ScopeResult(
                    scope=scope, confidence=conf,
                    reason=str(data.get("reason", "LLM classification")),
                )
            except Exception:
                pass

        # 3. Fallback
        if kw:
            return kw
        return ScopeResult(
            scope=Scope.UNCERTAIN, confidence=0.45,
            reason="Could not determine scope — proceeding as uncertain.",
        )
