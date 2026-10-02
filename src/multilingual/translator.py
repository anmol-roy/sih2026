"""
Multilingual Translator  (Phase 8 / Language Agent)
─────────────────────────────────────────────────────
Main entry point for all translation operations.

Pipeline:
  1. Detect source language (or accept user override)
  2. If non-English / non-Unknown: translate query → English  (before retrieval)
  3. Run existing IP-SAKTI pipeline (English only)
  4. Translate English answer → target language (after generation)
  5. Attach original structured citations unchanged

Key invariants:
  - Citations are NEVER translated or modified
  - Legal identifiers (Section 3(p), PCT, TRIPS, …) are protected via placeholders
  - Confidence / jurisdiction / IP type pass through unchanged
  - The disclaimer is re-appended in the target language (hard-coded)
  - UNKNOWN language → English passthrough (no translation attempted)
  - Any translation exception → fallback to original text (never empty)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from multilingual.schemas import (
    Language, LanguageResult, GroundedAnswer, MultilingualCitation
)
from multilingual.detector import LanguageDetector
from multilingual.provider import TranslationProvider, get_provider

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded safe responses (LLM cannot weaken these)
# ─────────────────────────────────────────────────────────────────────────────

_DISCLAIMERS: dict[Language, str] = {
    Language.ENGLISH: (
        "This information is preliminary and is not legal advice. "
        "Always consult a qualified IP professional."
    ),
    Language.HINDI: (
        "यह जानकारी प्रारंभिक है और कानूनी सलाह नहीं है। "
        "हमेशा एक योग्य IP पेशेवर से परामर्श करें।"
    ),
    Language.KANNADA: (
        "ಈ ಮಾಹಿತಿಯು ಪ್ರಾಥಮಿಕವಾಗಿದ್ದು ಕಾನೂನು ಸಲಹೆಯಲ್ಲ. "
        "ಯಾವಾಗಲೂ ಅರ್ಹ IP ವೃತ್ತಿಪರರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ),
    Language.UNKNOWN: (
        "This information is preliminary and is not legal advice. "
        "Always consult a qualified IP professional."
    ),
}

_INSUFFICIENT_RESPONSES: dict[Language, str] = {
    Language.ENGLISH: (
        "I could not find sufficient information in the provided "
        "authoritative documents."
    ),
    Language.HINDI: (
        "मुझे प्रदान किए गए अधिकृत दस्तावेज़ों में पर्याप्त जानकारी नहीं मिली।"
    ),
    Language.KANNADA: (
        "ಒದಗಿಸಲಾದ ಅಧಿಕೃತ ದಾಖಲೆಗಳಲ್ಲಿ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ."
    ),
    Language.UNKNOWN: (
        "I could not find sufficient information in the provided "
        "authoritative documents."
    ),
}


def _safe_disclaimer(lang: Language) -> str:
    """Return a disclaimer — always succeeds, falls back to English."""
    return _DISCLAIMERS.get(lang, _DISCLAIMERS[Language.ENGLISH])


def _safe_insufficient(lang: Language) -> str:
    """Return an 'insufficient info' response — always succeeds."""
    return _INSUFFICIENT_RESPONSES.get(lang, _INSUFFICIENT_RESPONSES[Language.ENGLISH])


# ─────────────────────────────────────────────────────────────────────────────
# MultilingualTranslator
# ─────────────────────────────────────────────────────────────────────────────

class MultilingualTranslator:
    """
    Orchestrates detect → translate-query → pipeline → translate-answer.

    All external calls (LLM, provider) are wrapped in try/except so the
    caller never crashes on translation errors.

    Parameters
    ----------
    provider        : TranslationProvider to use
    prefer_bhashini : if True and no explicit provider, try Bhashini first
    llm             : optional shared LLM (passed to LLMTranslationProvider)
    """

    def __init__(
        self,
        provider: Optional[TranslationProvider] = None,
        prefer_bhashini: bool = False,
        llm=None,
    ):
        try:
            self._provider = provider or get_provider(
                prefer_bhashini=prefer_bhashini, llm=llm
            )
        except Exception as e:
            logger.warning("Failed to init translation provider, using noop: %s", e)
            self._provider = None
        self._detector = LanguageDetector()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, text: str, override: Optional[str] = None) -> LanguageResult:
        """Detect or accept override language for *text*. Always succeeds."""
        try:
            return self._detector.detect(text, override=override)
        except Exception as e:
            logger.warning("Language detection failed, defaulting EN: %s", e)
            return LanguageResult(
                language=Language.ENGLISH, confidence=0.0, method="fallback_en"
            )

    def normalize_query(
        self,
        query: str,
        lang_result: LanguageResult,
    ) -> str:
        """
        Translate *query* to English if it is not already English / Unknown.

        FALLBACK: on any error, return the original query unmodified so the
        downstream pipeline still gets something to work with.
        """
        lang = lang_result.language
        # English or Unknown → no translation needed
        if lang in (Language.ENGLISH, Language.UNKNOWN):
            return query
        # No provider → return original (best effort)
        if self._provider is None:
            return query
        try:
            translated = self._provider.translate_to_english(query, lang)
            if not translated or not translated.strip():
                # Provider returned empty → keep original
                logger.warning("normalize_query returned empty, keeping original")
                return query
            return translated
        except Exception as e:
            logger.warning("Query normalization failed for lang=%s: %s", lang, e)
            # Fallback: original query
            return query

    def translate_answer(
        self,
        answer: GroundedAnswer,
        target_language: Language,
    ) -> GroundedAnswer:
        """
        Translate the `answer_text` field of *answer* into *target_language*.

        Guarantees:
          - NEVER raises
          - Citations are never modified
          - Confidence / ip_types / jurisdiction pass through unchanged
          - Unknown / English → passthrough (no LLM call)
          - On any provider error → original English answer is kept
        """
        # ── English or Unknown: passthrough ─────────────────────────────────
        if target_language in (Language.ENGLISH, Language.UNKNOWN):
            answer.translated_text = answer.answer_text
            answer.language        = target_language if target_language == Language.ENGLISH else Language.ENGLISH
            answer.disclaimer      = _safe_disclaimer(target_language)
            return answer

        # ── Insufficient-answer path: no LLM call, use hard-coded string ────
        if not answer.sufficient:
            translated_body = _safe_insufficient(target_language)
            answer.translated_text = translated_body
            answer.language        = target_language
            answer.disclaimer      = _safe_disclaimer(target_language)
            return answer

        # ── Sufficient answer: attempt provider translation ─────────────────
        if self._provider is None:
            # No provider → keep English answer
            answer.translated_text = answer.answer_text
            answer.language        = target_language
            answer.disclaimer      = _safe_disclaimer(target_language)
            return answer

        try:
            translated_body = self._provider.translate_from_english(
                answer.answer_text, target_language
            )
            if not translated_body or not translated_body.strip():
                logger.warning("translate_from_english returned empty, using original")
                translated_body = answer.answer_text
        except Exception as e:
            logger.warning(
                "Answer translation to %s failed: %s → using English",
                target_language, e,
            )
            translated_body = answer.answer_text

        answer.translated_text = translated_body
        answer.language        = target_language
        answer.disclaimer      = _safe_disclaimer(target_language)
        return answer

    def full_pipeline(
        self,
        query: str,
        language_override: Optional[str],
        pipeline_fn,          # callable(english_query: str) → dict
        citation_builder,     # callable(pipeline_result: dict) → list[MultilingualCitation]
    ) -> GroundedAnswer:
        """
        Convenience method that runs the complete Language Agent pipeline:

          1. Detect language (or accept override)
          2. Normalize query to English (fail-safe: keep original on error)
          3. Call pipeline_fn(english_query) to get the RAG result
          4. Build GroundedAnswer from result
          5. Translate answer to detected/requested language (fail-safe: keep EN)

        This method NEVER raises. On any inner error it returns a well-formed
        GroundedAnswer with the original query and "low" confidence.
        """
        # 1. Detect (always succeeds)
        lang_result = self.detect(query, override=language_override)

        # 2. Normalize to English (always succeeds — fallback to original query)
        english_query = self.normalize_query(query, lang_result)

        # 3. Run the downstream pipeline
        try:
            result = pipeline_fn(english_query)
        except Exception as e:
            logger.error("Downstream pipeline failed: %s", e)
            result = {
                "answer": (
                    "An error occurred while processing your query. "
                    "Please try again or rephrase."
                ),
                "confidence": "low",
                "sufficient": False,
                "ip_types": [],
                "jurisdiction": "india",
            }

        # 4. Build GroundedAnswer
        try:
            citations = citation_builder(result)
        except Exception as e:
            logger.warning("Citation builder failed: %s", e)
            citations = []

        answer = GroundedAnswer(
            answer_text         = str(result.get("answer", "")),
            citations           = citations,
            confidence          = str(result.get("confidence", "low")),
            sufficient          = bool(result.get("sufficient", False)),
            language            = Language.ENGLISH,   # updated in step 5
            original_language   = lang_result.language,
            original_question   = query,
            normalized_question = english_query,
            ip_types            = list(result.get("ip_types", [])),
            jurisdiction        = result.get("jurisdiction"),
        )

        # 5. Translate answer (always succeeds — fallbacks to English)
        return self.translate_answer(answer, lang_result.language)
