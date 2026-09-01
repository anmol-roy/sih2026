"""
Multilingual Schemas  (Phase 8)
────────────────────────────────
Language taxonomy, detection result, and the GroundedAnswer schema
that keeps citations separate from the translated answer text.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Supported languages
# ─────────────────────────────────────────────────────────────────────────────

class Language(str, Enum):
    ENGLISH  = "en"
    HINDI    = "hi"
    KANNADA  = "kn"
    # Extension points — add Tamil, Telugu, Marathi, etc. later
    # TAMIL    = "ta"
    # TELUGU   = "te"
    # MARATHI  = "mr"
    # BENGALI  = "bn"


LANGUAGE_NAMES: dict[Language, str] = {
    Language.ENGLISH : "English",
    Language.HINDI   : "Hindi",
    Language.KANNADA : "Kannada",
}

# langdetect codes → Language enum
LANGDETECT_MAP: dict[str, Language] = {
    "en": Language.ENGLISH,
    "hi": Language.HINDI,
    "kn": Language.KANNADA,
}

# Keyword patterns that indicate each language (for short-query fallback)
LANGUAGE_KEYWORDS: dict[Language, list[str]] = {
    Language.HINDI   : ["क्या", "है", "में", "से", "का", "की", "के", "पर", "यह", "कैसे",
                        "धारा", "पेटेंट", "अधिनियम", "ज्ञान"],
    Language.KANNADA : ["ಏನು", "ಹೇಳು", "ಕಾಯ್ದೆ", "ಸೆಕ್ಷನ್", "ಪೇಟೆಂಟ್", "ಮಾಹಿತಿ", "ಇದು"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Detection result
# ─────────────────────────────────────────────────────────────────────────────

class LanguageResult(BaseModel):
    language  : Language
    confidence: float = Field(ge=0.0, le=1.0)
    method    : str   = "auto"   # "auto" | "override" | "keyword_fallback"


# ─────────────────────────────────────────────────────────────────────────────
# Grounded answer — keeps citations as structured data, separate from answer text
# ─────────────────────────────────────────────────────────────────────────────

class MultilingualCitation(BaseModel):
    """Citation that must never be translated or modified."""
    document    : str
    section     : Optional[str] = None
    subsection  : Optional[str] = None
    page        : Optional[int] = None
    source      : str
    jurisdiction: Optional[str] = None
    chunk_id    : str = ""


class GroundedAnswer(BaseModel):
    """
    Container that keeps the translatable answer text separate from
    structured citations. Translation only touches `answer_text`.
    All other fields pass through unchanged.
    """
    answer_text         : str                        # translatable
    translated_text     : Optional[str]   = None     # filled after translation
    citations           : List[MultilingualCitation] = Field(default_factory=list)
    confidence          : str                         # "high" | "medium" | "low"
    sufficient          : bool            = True
    language            : Language        = Language.ENGLISH
    original_language   : Language        = Language.ENGLISH
    original_question   : str             = ""
    normalized_question : str             = ""       # English version of the query
    ip_types            : List[str]       = Field(default_factory=list)
    jurisdiction        : Optional[str]   = None
    disclaimer          : str             = (
        "This information is preliminary and is not legal advice. "
        "Always consult a qualified IP professional."
    )
