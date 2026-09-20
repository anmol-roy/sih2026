"""
Translation Provider Abstraction  (Phase 8)
─────────────────────────────────────────────
TranslationProvider  — abstract base class
LLMTranslationProvider  — uses the existing ChatGroq LLM (always available)
BhashiniProvider    — in bhashini.py (real Indian-language API)

Choosing a provider:
    provider = get_provider(prefer_bhashini=True)
    # Returns BhashiniProvider if configured, else LLMTranslationProvider
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from multilingual.schemas import Language, LANGUAGE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# Protected-term handling (shared by all providers)
# ─────────────────────────────────────────────────────────────────────────────

_TERMS_PATH = Path(__file__).parent.parent.parent / "data" / "multilingual" / "protected_terms.json"

def _load_protected_terms() -> list[str]:
    if _TERMS_PATH.exists():
        with open(_TERMS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("terms", [])
    return []

_PROTECTED_TERMS: list[str] = _load_protected_terms()


def protect_terms(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace protected legal terms with stable placeholders.

    Returns (modified_text, mapping) where mapping is {placeholder: original}.
    """
    mapping: dict[str, str] = {}
    for i, term in enumerate(_PROTECTED_TERMS):
        # Case-sensitive, whole-word-ish match
        if term in text:
            placeholder = f"__TERM_{i}__"
            text = text.replace(term, placeholder)
            mapping[placeholder] = term
    return text, mapping


def restore_terms(text: str, mapping: dict[str, str]) -> str:
    """Restore all placeholders to their original protected terms."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class TranslationProvider:
    """
    Abstract translation provider interface.
    All providers must implement both methods.
    """

    def translate_to_english(self, text: str, source_language: Language) -> str:
        raise NotImplementedError

    def translate_from_english(self, text: str, target_language: Language) -> str:
        raise NotImplementedError

    # Shared utility: no-op for English
    @staticmethod
    def _is_english(lang: Language) -> bool:
        return lang == Language.ENGLISH


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based provider (default — uses existing ChatMistralAI)
# ─────────────────────────────────────────────────────────────────────────────

_TO_ENGLISH_PROMPT = """\
Translate the following intellectual-property query into English.

Rules:
1. Preserve ALL legal identifiers exactly: Section numbers, Rule numbers,
   Act names (e.g. Patents Act, 1970), IPC codes, PCT, TRIPS, TKDL, CPC.
2. Preserve scientific names of ingredients.
3. Preserve patent/application numbers (e.g. IN202011012345).
4. Do NOT add information.
5. Do NOT remove information.
6. Translate only the meaning; keep legal terms in English.

Source language: {language_name}
Text:
{text}

English translation:"""

_FROM_ENGLISH_PROMPT = """\
Translate the following intellectual-property information into {language_name}.

Rules:
1. Preserve ALL legal identifiers exactly as they appear: Section numbers,
   Rule numbers, Act names, IPC codes, PCT, TRIPS, TKDL, CPC, ABS.
2. Preserve scientific names of biological ingredients.
3. Preserve patent/application numbers.
4. Preserve ALL numerical values (section numbers, page numbers, years).
5. Do NOT change legal meaning.
6. Do NOT add legal conclusions.
7. Do NOT remove uncertainty or disclaimers.
8. Do NOT add information not present in the original.
9. The disclaimer "This information is not legal advice" must appear
   in translated form at the end.

English text:
{text}

{language_name} translation:"""


class LLMTranslationProvider(TranslationProvider):
    """
    Translation via the existing ChatMistralAI LLM.
    Always available — used as the default / fallback.
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            import os
            from langchain_mistralai import ChatMistralAI
            self._llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0,
                api_key=os.getenv("MISTRAL_API_KEY")
            )
        return self._llm

    def translate_to_english(self, text: str, source_language: Language) -> str:
        if self._is_english(source_language):
            return text

        # Protect terms before translation
        protected, mapping = protect_terms(text)

        prompt = _TO_ENGLISH_PROMPT.format(
            language_name=LANGUAGE_NAMES.get(source_language, source_language.value),
            text=protected,
        )
        result = self._get_llm().invoke(prompt).content.strip()

        # Restore any protected terms the LLM may have left as placeholders
        return restore_terms(result, mapping)

    def translate_from_english(self, text: str, target_language: Language) -> str:
        if self._is_english(target_language):
            return text

        # Protect terms — legal identifiers must not be translated
        protected, mapping = protect_terms(text)

        prompt = _FROM_ENGLISH_PROMPT.format(
            language_name=LANGUAGE_NAMES.get(target_language, target_language.value),
            text=protected,
        )
        result = self._get_llm().invoke(prompt).content.strip()

        return restore_terms(result, mapping)


# ─────────────────────────────────────────────────────────────────────────────
# Provider factory
# ─────────────────────────────────────────────────────────────────────────────

def get_provider(
    prefer_bhashini: bool = False,
    llm=None,
) -> TranslationProvider:
    """
    Return the best available translation provider.

    prefer_bhashini=True  → try BhashiniProvider first; fall back to LLM
    prefer_bhashini=False → always use LLMTranslationProvider
    """
    if prefer_bhashini:
        try:
            from multilingual.bhashini import BhashiniProvider
            bp = BhashiniProvider()
            if bp.is_available():
                return bp
        except Exception:
            pass

    return LLMTranslationProvider(llm=llm)
