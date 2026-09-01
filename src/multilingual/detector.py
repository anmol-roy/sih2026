"""
Language Detector  (Phase 8)
──────────────────────────────
Two-layer detection:
  1. User override  — always wins
  2. langdetect     — standard library detector
  3. Keyword fallback — for short queries where langdetect is unreliable

Returns a LanguageResult (language + confidence + method).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from multilingual.schemas import (
    Language, LanguageResult, LANGDETECT_MAP, LANGUAGE_KEYWORDS
)

# ─────────────────────────────────────────────────────────────────────────────
# Keyword fallback
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_detect(text: str) -> Optional[Language]:
    """
    Scan *text* for known language-specific keywords.
    Returns Language enum if confident, None otherwise.
    """
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits >= 2:
            return lang
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

class LanguageDetector:
    """
    Detect the language of a text string.

    Priority:
      1. User-supplied override (`override` parameter)
      2. langdetect library
      3. Keyword-based fallback
      4. Default to English
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
        # 1. User override
        if override:
            lang = self._coerce(override)
            if lang is not None:
                return LanguageResult(
                    language=lang, confidence=1.0, method="override"
                )

        # 2. langdetect
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0   # deterministic results
            code = detect(text)
            lang = LANGDETECT_MAP.get(code)
            if lang is not None:
                conf = 0.90 if len(text.split()) >= 4 else 0.70
                return LanguageResult(
                    language=lang, confidence=conf, method="auto"
                )
        except Exception:
            pass

        # 3. Keyword fallback
        lang = _keyword_detect(text)
        if lang is not None:
            return LanguageResult(
                language=lang, confidence=0.75, method="keyword_fallback"
            )

        # 4. Default: English
        return LanguageResult(
            language=Language.ENGLISH, confidence=0.50, method="default"
        )

    @staticmethod
    def _coerce(code: str) -> Optional[Language]:
        code = code.strip().lower()
        try:
            return Language(code)
        except ValueError:
            # Accept full names
            name_map = {"english": Language.ENGLISH, "hindi": Language.HINDI,
                        "kannada": Language.KANNADA}
            return name_map.get(code)
