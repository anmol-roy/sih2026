"""
Language Detector  (Phase 8 / Language Agent)
───────────────────────────────────────────────
Four-layer detection:
  1. User override  — always wins
  2. Script-based   — Devanagari → Hindi, Kannada script → Kannada
  3. langdetect     — standard library detector
  4. Keyword fallback — for short queries (Devanagari + Romanized Hindi)
  5. UNKNOWN / English default — based on confidence

Returns a LanguageResult (language + confidence + method).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from multilingual.schemas import (
    Language, LanguageResult, LANGDETECT_MAP, LANGUAGE_KEYWORDS,
    DEVANAGARI_RANGE, KANNADA_RANGE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Script-based detection (most reliable for Devanagari / Kannada text)
# ─────────────────────────────────────────────────────────────────────────────

def _script_counts(text: str) -> dict[str, int]:
    """Count characters belonging to Devanagari / Kannada / Latin scripts."""
    deva = kann = latin = 0
    for ch in text:
        cp = ord(ch)
        if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
            deva += 1
        elif KANNADA_RANGE[0] <= cp <= KANNADA_RANGE[1]:
            kann += 1
        elif ch.isalpha():
            latin += 1
    return {"deva": deva, "kann": kann, "latin": latin}


def _script_detect(text: str) -> Optional[Language]:
    """
    Detect language based on script (Unicode block) presence.

    - Any Devanagari chars    → Hindi
    - Any Kannada chars       → Kannada
    - Only Latin / numbers    → None (need keyword / langdetect)
    """
    counts = _script_counts(text)
    total_alpha = counts["deva"] + counts["kann"] + counts["latin"]
    if total_alpha == 0:
        return None

    # Threshold: at least 10% of alpha chars must be in a script
    if counts["deva"] >= max(1, total_alpha * 0.10):
        return Language.HINDI
    if counts["kann"] >= max(1, total_alpha * 0.10):
        return Language.KANNADA
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Keyword fallback (for short / romanized queries)
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_detect(text: str) -> Optional[Language]:
    """
    Scan *text* for known language-specific keywords.
    Checks both native scripts and Romanized forms.

    Returns Language enum if confident, None otherwise.
    """
    text_lower = text.lower()
    best_lang: Optional[Language] = None
    best_hits = 0

    for lang, keywords in LANGUAGE_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            # Whole-word-ish match for Latin keywords; substring for Indic scripts
            if any(ord(c) > 127 for c in kw):
                # Indic script keyword — substring match
                if kw in text:
                    hits += 1
            else:
                # Romanized keyword — word boundary match
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    hits += 1
        if hits > best_hits:
            best_hits = hits
            best_lang = lang

    # Threshold: need at least 2 distinct keyword hits for confidence
    if best_hits >= 2:
        return best_lang
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main detector
# ─────────────────────────────────────────────────────────────────────────────

class LanguageDetector:
    """
    Detect the language of a text string.

    Priority:
      1. User-supplied override (`override` parameter)
      2. Script-based detection (Devanagari → hi, Kannada → kn)
      3. langdetect library (if available and confident)
      4. Keyword-based fallback (Devanagari + Romanized)
      5. UNKNOWN if truly ambiguous / too short
      6. English as final safety default
    """

    def detect(
        self,
        text: str,
        override: Optional[str] = None,
    ) -> LanguageResult:
        """
        Parameters
        ----------
        text     : the user's query or any text to detect
        override : explicit language code ("en", "hi", "kn") — beats detection

        Returns a LanguageResult.
        """
        # ── 1. User override ────────────────────────────────────────────────
        if override:
            lang = self._coerce(override)
            if lang is not None and lang != Language.UNKNOWN:
                return LanguageResult(
                    language=lang, confidence=1.0, method="override"
                )

        text_stripped = text.strip()
        if not text_stripped:
            return LanguageResult(
                language=Language.UNKNOWN, confidence=0.0, method="empty"
            )

        word_count = len(text_stripped.split())
        counts = _script_counts(text_stripped)
        total_alpha = counts["deva"] + counts["kann"] + counts["latin"]

        # ── 2. Script-based detection (high reliability) ────────────────────
        script_lang = _script_detect(text_stripped)
        if script_lang is not None:
            # Higher confidence if lots of script chars
            script_chars = (
                counts["deva"] if script_lang == Language.HINDI else counts["kann"]
            )
            if total_alpha > 0:
                script_ratio = script_chars / total_alpha
            else:
                script_ratio = 0.0
            conf = 0.95 if script_ratio > 0.50 else 0.85
            return LanguageResult(
                language=script_lang, confidence=conf, method="script"
            )

        # ── 3. langdetect library (works well for longer Latin text) ────────
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0   # deterministic results
            code = detect(text_stripped)
            lang = LANGDETECT_MAP.get(code)
            if lang is not None:
                # Confidence depends on query length
                if word_count >= 6:
                    conf = 0.90
                elif word_count >= 4:
                    conf = 0.75
                else:
                    conf = 0.55
                return LanguageResult(
                    language=lang, confidence=conf, method="auto"
                )
        except Exception:
            # langdetect can raise LangDetectException on weird text
            pass

        # ── 4. Keyword fallback (catches Hinglish, short queries) ───────────
        kw_lang = _keyword_detect(text_stripped)
        if kw_lang is not None:
            return LanguageResult(
                language=kw_lang, confidence=0.75, method="keyword_fallback"
            )

        # ── 5. Truly ambiguous → UNKNOWN for short text ─────────────────────
        # If it's all Latin letters but very short / no keywords, mark unknown
        if total_alpha < 5 or word_count <= 2:
            return LanguageResult(
                language=Language.UNKNOWN, confidence=0.30, method="ambiguous_short"
            )

        # ── 6. Final fallback: assume English (mostly-Latin text, long enough)
        return LanguageResult(
            language=Language.ENGLISH, confidence=0.50, method="default_english"
        )

    @staticmethod
    def _coerce(code: str) -> Optional[Language]:
        code = code.strip().lower()
        try:
            return Language(code)
        except ValueError:
            # Accept full names
            name_map = {
                "english": Language.ENGLISH,
                "hindi": Language.HINDI,
                "kannada": Language.KANNADA,
                "unknown": Language.UNKNOWN,
            }
            return name_map.get(code)
