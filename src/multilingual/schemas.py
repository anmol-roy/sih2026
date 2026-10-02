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
    UNKNOWN  = "unknown"
    # Extension points — add Tamil, Telugu, Marathi, etc. later
    # TAMIL    = "ta"
    # TELUGU   = "te"
    # MARATHI  = "mr"
    # BENGALI  = "bn"


LANGUAGE_NAMES: dict[Language, str] = {
    Language.ENGLISH : "English",
    Language.HINDI   : "Hindi",
    Language.KANNADA : "Kannada",
    Language.UNKNOWN : "Unknown",
}

# langdetect codes → Language enum
LANGDETECT_MAP: dict[str, Language] = {
    "en": Language.ENGLISH,
    "hi": Language.HINDI,
    "kn": Language.KANNADA,
}

# Devanagari Unicode range for script-based detection
DEVANAGARI_RANGE = (0x0900, 0x097F)
KANNADA_RANGE    = (0x0C80, 0x0CFF)

# Keyword patterns that indicate each language (for short-query fallback)
# Hindi: Devanagari + Romanized Hindi (Hinglish) common words
LANGUAGE_KEYWORDS: dict[Language, list[str]] = {
    Language.HINDI   : [
        # Devanagari keywords
        "क्या", "है", "में", "से", "का", "की", "के", "पर", "यह", "कैसे",
        "धारा", "पेटेंट", "अधिनियम", "ज्ञान", "नहीं", "कर", "सकता",
        "सकती", "सकते", "मुझे", "तुम", "वह", "हम", "आप", "कौन",
        "कब", "कहाँ", "क्यूं", "और", "या", "लेकिन", "अगर", "तो",
        "भारत", "दवा", "आयुर्वेदिक", "फॉर्मूलेशन", "नीम", "हल्दी",
        # Romanized Hindi / Hinglish keywords (lowercase)
        "kya", "hai", "mein", "se", "ka", "ki", "ke", "par", "yeh", "kaise",
        "dhara", "patent", "adhiniyam", "gyan", "nahi", "kar", "sakta",
        "sakti", "sakte", "mujhe", "tum", "vah", "hum", "aap", "kaun",
        "kab", "kahan", "kyun", "aur", "ya", "lekin", "agar", "to",
        "bharat", "dawa", "ayurvedic", "formulation", "neem", "haldi",
        "hain", "hoon", "ho", "karna", "karne", "karte", "karta", "karti",
        "apna", "apni", "apne", "mera", "meri", "mere", "tera", "teri",
    ],
    Language.KANNADA : [
        # Kannada keywords
        "ಏನು", "ಹೇಳು", "ಕಾಯ್ದೆ", "ಸೆಕ್ಷನ್", "ಪೇಟೆಂಟ್", "ಮಾಹಿತಿ", "ಇದು",
        "ನನ್ನ", "ಮತ್ತು", "ಅರಿಶಿನ", "ಬೇವು", "ಭಾರತ", "ಆಯುರ್ವೇದಿಕ್",
        "ಫಾರ್ಮುಲೇಶನ್", "ಪಡೆಯಬಹುದೇ", "ಕೊಡುತ್ತದೆ", "ಎಂದರೆ", "ಹೇಗೆ",
        "ಯಾವುದೇ", "ಇಲ್ಲ", "ಹೌದು", "ನೀವು", "ನಾವು", "ಅವರು",
    ],
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
