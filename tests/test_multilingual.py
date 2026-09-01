"""
Phase 8 — Multilingual tests  (no LLM / no network)

Run from project root:
    python tests/test_multilingual.py
    python -m pytest tests/test_multilingual.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multilingual.schemas import Language, LanguageResult, GroundedAnswer, MultilingualCitation
from multilingual.detector import LanguageDetector, _keyword_detect
from multilingual.provider import protect_terms, restore_terms, _PROTECTED_TERMS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema tests
# ─────────────────────────────────────────────────────────────────────────────

def test_language_enum_values():
    assert Language.ENGLISH.value == "en"
    assert Language.HINDI.value   == "hi"
    assert Language.KANNADA.value == "kn"

def test_grounded_answer_citations_untouched():
    """Citations must survive even if answer_text changes."""
    cit = MultilingualCitation(document="Patents Act, 1970", section="3(p)",
                                source="India Code", chunk_id="abc")
    ans = GroundedAnswer(
        answer_text="Section 3(p) explains...",
        citations=[cit],
        confidence="high",
    )
    # Simulate translation (only answer_text changes)
    ans.translated_text = "धारा 3(p) बताती है..."
    ans.language = Language.HINDI

    assert len(ans.citations) == 1
    assert ans.citations[0].document == "Patents Act, 1970"
    assert ans.citations[0].section  == "3(p)"

def test_grounded_answer_confidence_unchanged():
    """Confidence must not change after translation."""
    ans = GroundedAnswer(answer_text="test", citations=[], confidence="medium")
    ans.translated_text = "अनुवाद"
    ans.language = Language.HINDI
    assert ans.confidence == "medium"   # unchanged


# ─────────────────────────────────────────────────────────────────────────────
# 2. Language detection tests
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_english():
    d = LanguageDetector()
    r = d.detect("What does Section 3(p) of the Patents Act say?")
    assert r.language == Language.ENGLISH

def test_detect_hindi_keyword():
    """Hindi query with clear keywords — keyword fallback should fire."""
    result = _keyword_detect("पेटेंट अधिनियम की धारा 3(p) क्या कहती है?")
    assert result == Language.HINDI

def test_detect_kannada_keyword():
    result = _keyword_detect("ಪೇಟೆಂಟ್ ಕಾಯ್ದೆಯ ಸೆಕ್ಷನ್ 3(p) ಏನು ಹೇಳುತ್ತದೆ?")
    assert result == Language.KANNADA

def test_detect_override_english():
    d = LanguageDetector()
    r = d.detect("नीम और हल्दी", override="en")
    assert r.language   == Language.ENGLISH
    assert r.confidence == 1.0
    assert r.method     == "override"

def test_detect_override_hindi():
    d = LanguageDetector()
    r = d.detect("What is patent?", override="hi")
    assert r.language   == Language.HINDI
    assert r.confidence == 1.0

def test_detect_override_kannada():
    d = LanguageDetector()
    r = d.detect("What is patent?", override="kn")
    assert r.language == Language.KANNADA

def test_detect_override_case_insensitive():
    d = LanguageDetector()
    r = d.detect("test", override="HI")
    assert r.language == Language.HINDI

def test_detect_override_full_name():
    d = LanguageDetector()
    r = d.detect("test", override="hindi")
    assert r.language == Language.HINDI

def test_detect_default_to_english():
    d = LanguageDetector()
    r = d.detect("xyz 123")   # ambiguous — should default to English
    assert r.language == Language.ENGLISH


# ─────────────────────────────────────────────────────────────────────────────
# 3. Protected-term tests
# ─────────────────────────────────────────────────────────────────────────────

def test_protected_terms_loaded():
    assert len(_PROTECTED_TERMS) > 5

def test_protect_section_3p():
    text = "Section 3(p) of the Patents Act, 1970"
    protected, mapping = protect_terms(text)
    assert "Section 3(p)" not in protected or "__TERM_" in protected
    restored = restore_terms(protected, mapping)
    assert restored == text

def test_protect_pct_trips():
    text = "PCT applications must comply with TRIPS standards."
    protected, mapping = protect_terms(text)
    restored = restore_terms(protected, mapping)
    assert restored == text

def test_protect_multiple_terms():
    text = "Patents Act, 1970 Section 3(d) and TKDL and PCT all apply here."
    protected, mapping = protect_terms(text)
    restored = restore_terms(protected, mapping)
    assert restored == text

def test_protect_does_not_alter_non_terms():
    text = "The invention involves neem extract."
    protected, mapping = protect_terms(text)
    # "neem extract" is not a protected term — it should stay unchanged
    assert "neem extract" in protected


# ─────────────────────────────────────────────────────────────────────────────
# 4. Provider tests (no LLM — only structural)
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_provider_english_passthrough():
    """English → English should return text unchanged without any LLM call."""
    from multilingual.provider import LLMTranslationProvider
    p = LLMTranslationProvider(llm=None)   # no LLM needed for English
    result = p.translate_to_english("Hello world", Language.ENGLISH)
    assert result == "Hello world"

def test_llm_provider_from_english_passthrough():
    from multilingual.provider import LLMTranslationProvider
    p = LLMTranslationProvider(llm=None)
    result = p.translate_from_english("Section 3(p) is relevant.", Language.ENGLISH)
    assert result == "Section 3(p) is relevant."

def test_bhashini_not_available_without_config():
    """BhashiniProvider.is_available() must be False without env vars."""
    import os
    # Temporarily clear any existing env vars
    uid = os.environ.pop("BHASHINI_USER_ID", None)
    key = os.environ.pop("BHASHINI_API_KEY", None)
    try:
        from multilingual.bhashini import BhashiniProvider
        bp = BhashiniProvider()
        assert bp.is_available() is False
    finally:
        if uid: os.environ["BHASHINI_USER_ID"] = uid
        if key: os.environ["BHASHINI_API_KEY"] = key

def test_get_provider_returns_llm_by_default():
    from multilingual.provider import get_provider, LLMTranslationProvider
    p = get_provider(prefer_bhashini=False)
    assert isinstance(p, LLMTranslationProvider)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Translator tests (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def test_translator_english_query_unchanged():
    """English query must not be translated."""
    from multilingual.translator import MultilingualTranslator
    from multilingual.provider import LLMTranslationProvider

    translator = MultilingualTranslator(provider=LLMTranslationProvider(llm=None))
    lang_result = LanguageResult(language=Language.ENGLISH, confidence=1.0)
    result = translator.normalize_query(
        "What does Section 3(p) say?", lang_result
    )
    assert result == "What does Section 3(p) say?"

def test_translator_disclaimer_in_hindi():
    """After translation to Hindi, disclaimer must be the Hindi version."""
    from multilingual.translator import MultilingualTranslator, _DISCLAIMERS
    from multilingual.provider import LLMTranslationProvider

    translator = MultilingualTranslator(provider=LLMTranslationProvider(llm=None))
    ans = GroundedAnswer(
        answer_text="Section 3(p) excludes traditional knowledge.",
        citations=[],
        confidence="high",
        sufficient=False,   # triggers "insufficient" path — no LLM call
    )
    result = translator.translate_answer(ans, Language.HINDI)
    assert result.language     == Language.HINDI
    assert result.disclaimer   == _DISCLAIMERS[Language.HINDI]
    assert "कानूनी सलाह" in result.disclaimer

def test_translator_disclaimer_in_kannada():
    from multilingual.translator import MultilingualTranslator, _DISCLAIMERS
    from multilingual.provider import LLMTranslationProvider

    translator = MultilingualTranslator(provider=LLMTranslationProvider(llm=None))
    ans = GroundedAnswer(
        answer_text="Section 3(p) excludes traditional knowledge.",
        citations=[],
        confidence="high",
        sufficient=False,
    )
    result = translator.translate_answer(ans, Language.KANNADA)
    assert result.language   == Language.KANNADA
    assert result.disclaimer == _DISCLAIMERS[Language.KANNADA]

def test_translator_insufficient_hindi():
    """'I don't know' in Hindi is returned without LLM call when sufficient=False."""
    from multilingual.translator import MultilingualTranslator, _INSUFFICIENT_RESPONSES
    from multilingual.provider import LLMTranslationProvider

    translator = MultilingualTranslator(provider=LLMTranslationProvider(llm=None))
    ans = GroundedAnswer(
        answer_text="I could not find sufficient information.",
        citations=[],
        confidence="low",
        sufficient=False,
    )
    result = translator.translate_answer(ans, Language.HINDI)
    assert result.translated_text == _INSUFFICIENT_RESPONSES[Language.HINDI]
    assert "पर्याप्त" in result.translated_text

def test_translator_citations_never_modified():
    """Citations must pass through translation completely unchanged."""
    from multilingual.translator import MultilingualTranslator
    from multilingual.provider import LLMTranslationProvider

    cit = MultilingualCitation(
        document="Patents Act, 1970",
        section="3(p)",
        source="India Code",
        chunk_id="test_chunk_1",
    )
    translator = MultilingualTranslator(provider=LLMTranslationProvider(llm=None))
    ans = GroundedAnswer(
        answer_text="test", citations=[cit], confidence="high", sufficient=False
    )
    result = translator.translate_answer(ans, Language.HINDI)

    assert len(result.citations) == 1
    assert result.citations[0].document == "Patents Act, 1970"
    assert result.citations[0].section  == "3(p)"
    assert result.citations[0].chunk_id == "test_chunk_1"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_all() -> None:
    tests = [
        (test_language_enum_values,              "Schema: Language enum values"),
        (test_grounded_answer_citations_untouched,"Schema: citations survive translation"),
        (test_grounded_answer_confidence_unchanged,"Schema: confidence unchanged"),
        (test_detect_english,                    "Detect: English query"),
        (test_detect_hindi_keyword,              "Detect: Hindi keyword"),
        (test_detect_kannada_keyword,            "Detect: Kannada keyword"),
        (test_detect_override_english,           "Detect: override en"),
        (test_detect_override_hindi,             "Detect: override hi"),
        (test_detect_override_kannada,           "Detect: override kn"),
        (test_detect_override_case_insensitive,  "Detect: override case-insensitive"),
        (test_detect_override_full_name,         "Detect: override full name"),
        (test_detect_default_to_english,         "Detect: default to English"),
        (test_protected_terms_loaded,            "Terms: DB loaded"),
        (test_protect_section_3p,                "Terms: protect Section 3(p)"),
        (test_protect_pct_trips,                 "Terms: protect PCT TRIPS"),
        (test_protect_multiple_terms,            "Terms: protect multiple terms"),
        (test_protect_does_not_alter_non_terms,  "Terms: non-terms unchanged"),
        (test_llm_provider_english_passthrough,  "Provider: EN→EN passthrough"),
        (test_llm_provider_from_english_passthrough, "Provider: EN→EN from_english"),
        (test_bhashini_not_available_without_config, "Bhashini: not available without creds"),
        (test_get_provider_returns_llm_by_default,   "Provider: factory returns LLM"),
        (test_translator_english_query_unchanged,"Translator: EN query unchanged"),
        (test_translator_disclaimer_in_hindi,    "Translator: Hindi disclaimer"),
        (test_translator_disclaimer_in_kannada,  "Translator: Kannada disclaimer"),
        (test_translator_insufficient_hindi,     "Translator: insufficient→Hindi"),
        (test_translator_citations_never_modified,"Translator: citations unchanged"),
    ]

    print("=" * 65)
    print("IP-SAKTI  Phase 8 — Multilingual Tests")
    print("=" * 65)

    passed = failed = 0
    for fn, label in tests:
        try:
            fn()
            print(f"  ✓  {label}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {label}  →  {e}")
            failed += 1

    print(f"\n{'─'*65}")
    print(f"  TOTAL  {passed}/{passed+failed}  ({passed/(passed+failed)*100:.0f}%)")
    if failed == 0:
        print("\n  ✓ All Phase 8 tests passed.")
    else:
        print(f"\n  ⚠  {failed} test(s) failed.")


if __name__ == "__main__":
    _run_all()
