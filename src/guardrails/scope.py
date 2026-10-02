"""
Scope & Guardrail Classifier  (02 Scope Agent)
───────────────────────────────────────────────
Four categories per 02.txt:
  in_scope            – process via downstream RAG / investigation
  needs_clarification – ask a clarifying question, do not run RAG
  out_of_scope        – politely refuse, do not run RAG
  unsafe_or_disallowed– refuse the unsafe part, optionally redirect to safe IP question

Deterministic rule-based + keyword (no LLM cost) with safe fallback
to in_scope on ambiguity for borderline but clearly legal-domain queries.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ScopeStatus(str, Enum):
    IN_SCOPE            = "in_scope"
    NEEDS_CLARIFICATION = "needs_clarification"
    OUT_OF_SCOPE        = "out_of_scope"
    UNSAFE_OR_DISALLOWED = "unsafe_or_disallowed"


class GuardrailResult(BaseModel):
    status     : ScopeStatus
    confidence : float = Field(ge=0.0, le=1.0)
    reason     : str = ""
    # If status is needs_clarification, ask this question (EN – translated later)
    clarification_question: Optional[str] = None
    # If status is out_of_scope / unsafe, a brief user-facing response (EN)
    user_response: Optional[str] = None


# Backwards-compatible aliases (used by guardrails.pipeline, older callers)
Scope        = ScopeStatus
ScopeResult  = GuardrailResult


# ─────────────────────────────────────────────────────────────────────────────
# Prompt injection / unsafe patterns
# ─────────────────────────────────────────────────────────────────────────────

_INJECTION_RE = re.compile(
    r"ignore\s+(previous|all|prior|earlier)\s+instructions?|"
    r"forget\s+(everything|all|prior|previous)\s+(content|instructions|rules)|"
    r"reveal\s+(system\s*prompt|api\s*key|credentials?|password|internal\s*prompt)|"
    r"disregard\s+(your\s+)?instructions?|"
    r"(act|pretend)\s+as\s+if\s+you\s+(are|were)|"
    r"you\s+are\s+now\s+(a\s+)?(lawyer|judge|solicitor|barrister)|"
    r"definitely\s+(will|is|be)\s+(granted|rejected|accepted|approved|patentable)|"
    r"invent\s+(three|several|some|new)\s+(patent|documents?|sources?|citations?)|"
    r"do\s+not\s+use\s+any\s+(sources?|retrieval|evidence|authorit)",
    re.IGNORECASE,
)

# Unsupported-legal-advice pressure: definitive WILL/WON'T outcomes
_DEF_INJURY_RE = re.compile(
    r"tell\s+me\s+that\s+my\s+(formulation|invention|patent|product|application)\s+"
    r"(definitely\s+)?(is|will\s+be|is\s+definitely)\s+(patentable|granted|approved|accepted)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# In-scope keyword list (English + a couple of well-known Indic forms)
# ─────────────────────────────────────────────────────────────────────────────

_IP_KW = re.compile(
    r"""
    \bpatent\b              | \bpatentab\w*\b        | \bprior\s*art\b |
    \btrademark\b           | \btrade\s*mark\b       | \btm\b          |
    \bcopyright\b           | \bcopy\s*right\b       | \bdesign\s*right\b |
    \bdesign\b              | \bgeographical\s*indication\b | \bgi\b(?!\w) |
    \bplant\s*variet\w*\b   | \btrade\s*secret\b     | \btrade\s*secret\b |
    \bregistr\w*\b          | \bprotect(ion)?\b      | \bip\s*india\b |
    \bnovelty\b             | \binventive\s*step\b   | \bpriority\s*date\b |
    \bsection\s*3\b         | \bsection\s+3\w*       | \bpatents?\s*act\b |
    \btrademarks?\s*act\b   | \bcopyright\s*act\b    | \bdesigns?\s*act\b |
    \bpct\b                 | \btrips\b              | \bwipo\b |
    \bparis\s*convention\b  | \bbiodiversit\w*\b     | \babs\b(?!\w) |
    \bnagoya\b              | \bbenefit\s*shar\w*\b  | \baccess\s+and\s+benefit\s+shar\w*\b |
    \bgenetic\s*resource\w* | \bbiologica?l\s*resource\w* |
    पेटेंट                  | ट्रेडमार्क              | कॉपीराइट       |
    अधिनियम                | धारा                     | बौद्धिक\s*संपदा |
    पेटेंट                  | ट्रेडमार्क              | सुरक्षा\b      |
    ಪೇಟೆಂಟ್                 | ಟ್ರೇಡ್‌ಮಾರ್ಕ್            | ಕಾಯ್ದೆ         |
    ಸೆಕ್ಷನ್                 | ಆಯುರ್ವೇದಿಕ್             | ರಕ್ಷಣೆ\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_AYUSH_KW = re.compile(
    r"""
    \bayurved\w*\b          | \bherbal\b             | \bformulation\b |
    \bsiddha\b              | \bunani\b              | \bayush\b |
    \bpharmacopoei\w*\b     | \bneem\b               | \bturmeric\b |
    \bhaldi\b               | \bkuppa\b              | \bbevu\b |
    \btraditional\s*knowled\w*\b | \btkdl\b           | \btk\b(?!\w) |
    \btraditional\s*medic\w*\b   | \btraditiona?l\s+formulation\w*\b |
    \btraditional\s+practic\w*\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

IN_SCOPE_KW = [_IP_KW, _AYUSH_KW]


# ─────────────────────────────────────────────────────────────────────────────
# Strong out-of-scope
# ─────────────────────────────────────────────────────────────────────────────

_OUT_STRONG = re.compile(
    r"""
    \bweather\b             | \bforecast\b           | \bcricket\b |
    \bfootball\b            | \bsoccer\b             | \bmatch\s*winner\b |
    \brecipe\b              | \bcooking\b            | \bbiryani\b |
    \bstock\s*price\b       | \bshare\s*price\b      | \bcryptocurrenc\w*\b |
    \bcelebrit\w*\b         | \bgossip\b             | \bmovie\b |
    \bsong\b                | \blyric\w*\b           | \bhoroscope\b |
    \bastrology\b           | \bpython\s+program\b   | \bfactorial\b |
    \bsort\s+an\s+array\b   | \blove\s+letter\b      | \brelationship\s+advic\w*\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Ambiguity detection for NEEDS_CLARIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_PRONOUN_IT = re.compile(r"\b(can|may|should|is|will|do)\s+(i|we)\s+("
                         r"patent|protect|trademark|copyright|register|"
                         r"sell|manufacture|launch|file)\s+(it|this|that|the\s+product|the\s+invention|the\s+formulation)?\b",
                         re.IGNORECASE)

_GENERIC_PROTECT = re.compile(
    r"\b(how\s+can|can|how\s+to|how\s+do|should|want\s+to)\s+"
    r"(i|we|my|our)?\s*("
    r"protect|register|get\s+protection|secure)\b",
    re.IGNORECASE,
)


def _is_ambiguous(query: str) -> Optional[str]:
    """
    If query is about IP/Ayurvedic domain but lacks enough context, return
    the clarification question to ask (in English). Returns None otherwise.
    """
    q = query.strip().rstrip("?").rstrip(".")
    words = re.findall(r"[A-Za-z\u0900-\u097F\u0C80-\u0CFF]+", q)

    has_ambiguous_pronoun = bool(re.search(r"\b(it|this|that)\b", q, re.IGNORECASE))
    has_generic_my = bool(re.search(
        r"\bmy\s+(product|invention|formulation|thing|idea|stuff)\b", q, re.IGNORECASE))
    has_protect_pattern = bool(_PRONOUN_IT.search(q)) or has_ambiguous_pronoun or has_generic_my

    # Check for substantive specific nouns (not just the IP action word like "patent"
    # by itself — we want the actual subject of the question).
    substantive_noun_re = re.compile(
        r"(formulation|invention|trademark|copyright|design|"
        r"brand\s+name|logo|process|recipe|traditional|ayurved|neem|turmeric|"
        r"compound|medicine|drug|skincare|product\s+in|product\s+for|"
        r"biological|resource|abs\b|geographical|tkdl|wipo|pct|trips|"
        r"plant\s*variety|trade\s*secret|novelty|prior\s*art)",
        re.IGNORECASE,
    )
    has_substantive_noun = bool(substantive_noun_re.search(q))

    # Very short queries: require a substantive noun. If the only domain word
    # is "patent" + "it" / "this" / "my product", it's still ambiguous.
    if len(words) <= 10:
        if _IP_KW.search(q) or _AYUSH_KW.search(q) or _GENERIC_PROTECT.search(q):
            if has_protect_pattern and not has_substantive_noun:
                return ("Could you provide a bit more detail — what is the "
                        "specific invention or formulation, and what aspect "
                        "are you trying to protect (e.g. the composition, "
                        "brand name, logo, design, or process)?")

    # "How can I protect ...?" pattern without any substantive subject noun
    if _GENERIC_PROTECT.search(q):
        if not has_substantive_noun:
            return ("What aspect are you trying to protect — the "
                    "formulation/invention, brand name, logo, "
                    "packaging/design, or traditional knowledge?")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Utility: out-of-scope & unsafe user-facing responses
# ─────────────────────────────────────────────────────────────────────────────

_OUT_OF_SCOPE_RESP = (
    "This query is outside the supported scope of Anvesh AI. "
    "Anvesh AI is a research assistant for Intellectual Property, "
    "Ayurveda, Traditional Knowledge, and Access-and-Benefit "
    "Sharing (ABS) questions. Please rephrase your question in "
    "those domains."
)

_UNSAFE_RESP = (
    "I'm designed to provide research-based information using "
    "source evidence, not to make definitive legal outcomes or "
    "ignore sources. I'll treat your underlying IP or Ayurvedic "
    "question as a normal query — please describe the formulation, "
    "jurisdiction, and protection type of interest so the evidence "
    "can be retrieved safely."
)


# ─────────────────────────────────────────────────────────────────────────────
# ScopeGuard class
# ─────────────────────────────────────────────────────────────────────────────

class ScopeGuard:
    """
    Deterministic Scope & Guardrail checker. No LLM calls.

    Ordering:
      1. Prompt injection / unsafe → UNSAFE_OR_DISALLOWED (keep underlying Q)
      2. Strong out-of-scope keyword → OUT_OF_SCOPE
      3. Domain keyword present AND ambiguous → NEEDS_CLARIFICATION
      4. Domain keyword present → IN_SCOPE
      5. No strong signal & short generic → NEEDS_CLARIFICATION
      6. No strong signal → OUT_OF_SCOPE (best-effort conservative)
    """

    def check(
        self,
        normalized_query: str,
        original_query: Optional[str] = None,
    ) -> GuardrailResult:
        query = normalized_query.strip()
        if not query:
            return GuardrailResult(
                status=ScopeStatus.NEEDS_CLARIFICATION,
                confidence=0.9,
                reason="Empty query.",
                clarification_question="Please enter a question about "
                                       "IP, Ayurveda, traditional knowledge, "
                                       "or biological resources.",
            )

        # ── 1. Prompt injection / unsafe ───────────────────────────────────
        if _INJECTION_RE.search(query) or _DEF_INJURY_RE.search(query):
            # Underlying question may still be in scope, but we refuse the
            # unsafe instruction portion. Response is the safe fallback text.
            return GuardrailResult(
                status=ScopeStatus.UNSAFE_OR_DISALLOWED,
                confidence=0.92,
                reason="Prompt-injection or unsupported-definite-outcome "
                       "instruction detected in user query.",
                user_response=_UNSAFE_RESP,
            )

        # ── 2. Strong out-of-scope (early short-circuit) ──────────────────
        if _OUT_STRONG.search(query):
            return GuardrailResult(
                status=ScopeStatus.OUT_OF_SCOPE,
                confidence=0.95,
                reason="Query contains strong out-of-scope keywords unrelated "
                       "to IP, Ayurveda, TK, or ABS.",
                user_response=_OUT_OF_SCOPE_RESP,
            )

        # ── 3/4. Domain keywords ──────────────────────────────────────────
        in_kw = any(p.search(query) for p in IN_SCOPE_KW)

        if in_kw:
            amb = _is_ambiguous(query)
            if amb:
                return GuardrailResult(
                    status=ScopeStatus.NEEDS_CLARIFICATION,
                    confidence=0.88,
                    reason="Domain-related query but missing specific "
                           "context needed for a meaningful answer.",
                    clarification_question=amb,
                )
            return GuardrailResult(
                status=ScopeStatus.IN_SCOPE,
                confidence=0.90,
                reason="Query contains IP / Ayurveda / TK / ABS keywords.",
            )

        # ── 5. No domain keyword — could be very vague or truly unrelated ─
        words = re.findall(r"[A-Za-z\u0900-\u097F\u0C80-\u0CFF]+", query)
        word_count = len(words)

        if word_count <= 4:
            # Short, no-domain match — safest is needs_clarification
            return GuardrailResult(
                status=ScopeStatus.NEEDS_CLARIFICATION,
                confidence=0.70,
                reason="Short query with no domain keywords — ambiguous intent.",
                clarification_question=(
                    "Please clarify your question. Anvesh AI focuses on "
                    "Intellectual Property, Ayurveda, Traditional Knowledge, "
                    "and biological resource / ABS questions."
                ),
            )

        # Longer but still no domain keyword → treat as out_of_scope
        return GuardrailResult(
            status=ScopeStatus.OUT_OF_SCOPE,
            confidence=0.80,
            reason="Longer query but no IP / Ayurveda / TK / ABS keywords — "
                   "treated as out of scope.",
            user_response=_OUT_OF_SCOPE_RESP,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible ScopeChecker  (used by guardrails.pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class ScopeChecker(ScopeGuard):
    """Legacy wrapper. Use ScopeGuard in new code."""
    pass
