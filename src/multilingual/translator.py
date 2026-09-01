"""
Multilingual Translator  (Phase 8)
────────────────────────────────────
Main entry point for all translation operations.

Pipeline:
  1. Detect source language (or accept user override)
  2. If non-English: translate query → English  (before retrieval)
  3. Run existing IP-SAKTI pipeline (English only)
  4. Translate English answer → target language (after generation)
  5. Attach original structured citations unchanged

Key invariants:
  - Citations are NEVER translated or modified
  - Legal identifiers (Section 3(p), PCT, TRIPS, …) are protected
  - Confidence / jurisdiction / IP type pass through unchanged
  - The disclaimer is re-appended in the target language
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from multilingual.schemas import (
    Language, LanguageResult, GroundedAnswer, MultilingualCitation
)
from multilingual.detector import LanguageDetector
from multilingual.provider import TranslationProvider, get_provider

# Translated disclaimers (hard-coded so the LLM never weakens them)
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
}

# "I don't know" responses by language
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
}


class MultilingualTranslator:
    """
    Orchestrates detect → translate-query → pipeline → translate-answer.

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
        self._provider = provider or get_provider(
            prefer_bhashini=prefer_bhashini, llm=llm
        )
        self._detector = LanguageDetector()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, text: str, override: Optional[str] = None) -> LanguageResult:
        """Detect or accept override language for *text*."""
        return self._detector.detect(text, override=override)

    def normalize_query(
        self,
        query: str,
        lang_result: LanguageResult,
    ) -> str:
        """
        Translate *query* to English if it is not already English.
        Returns the English query ready for retrieval.
        """
        if lang_result.language == Language.ENGLISH:
            return query
        return self._provider.translate_to_english(query, lang_result.language)

    def translate_answer(
        self,
        answer: GroundedAnswer,
        target_language: Language,
    ) -> GroundedAnswer:
        """
        Translate the `answer_text` field of *answer* into *target_language*.

        - Citations are never modified
        - Confidence/IP-type/jurisdiction pass through unchanged
        - Disclaimer is replaced with the hard-coded target-language version
        - Returns a new GroundedAnswer with `translated_text` filled in
        """
        if target_language == Language.ENGLISH:
            answer.translated_text = answer.answer_text
            answer.language        = Language.ENGLISH
            answer.disclaimer      = _DISCLAIMERS[Language.ENGLISH]
            return answer

        if not answer.sufficient:
            translated_body = _INSUFFICIENT_RESPONSES.get(
                target_language, answer.answer_text
            )
        else:
            translated_body = self._provider.translate_from_english(
                answer.answer_text, target_language
            )

        answer.translated_text = translated_body
        answer.language        = target_language
        answer.disclaimer      = _DISCLAIMERS.get(
            target_language, _DISCLAIMERS[Language.ENGLISH]
        )
        return answer

    def full_pipeline(
        self,
        query: str,
        language_override: Optional[str],
        pipeline_fn,          # callable(english_query: str) → dict
        citation_builder,     # callable(pipeline_result: dict) → list[MultilingualCitation]
    ) -> GroundedAnswer:
        """
        Convenience method that runs the complete Phase 8 pipeline:

          1. Detect language
          2. Normalize query to English
          3. Call pipeline_fn(english_query) to get the RAG result
          4. Build GroundedAnswer from result
          5. Translate answer to detected/requested language

        Parameters
        ----------
        query            : user query in any language
        language_override: optional explicit language code
        pipeline_fn      : callable that accepts an English query and returns a
                           dict with keys: answer, confidence, sufficient,
                           ip_types, jurisdiction
        citation_builder : callable that converts the pipeline result dict to
                           list[MultilingualCitation]
        """
        # 1. Detect
        lang_result = self.detect(query, override=language_override)

        # 2. Normalize to English
        english_query = self.normalize_query(query, lang_result)

        # 3. Run pipeline
        result = pipeline_fn(english_query)

        # 4. Build GroundedAnswer
        answer = GroundedAnswer(
            answer_text       = result.get("answer", ""),
            citations         = citation_builder(result),
            confidence        = result.get("confidence", "low"),
            sufficient        = result.get("sufficient", False),
            language          = Language.ENGLISH,   # will be updated in step 5
            original_language = lang_result.language,
            original_question = query,
            normalized_question = english_query,
            ip_types          = result.get("ip_types", []),
            jurisdiction      = result.get("jurisdiction"),
        )

        # 5. Translate answer
        return self.translate_answer(answer, lang_result.language)
