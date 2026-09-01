"""
Phase 8 — Multilingual Evaluation
────────────────────────────────────
Measures three things for each language:
  1. Language detection accuracy      — is the query correctly identified?
  2. Query normalisation accuracy     — does English translation preserve intent?
  3. Protected-term preservation      — are Section numbers / Act names kept intact?

The retrieval-consistency test (same Section retrieved for all languages)
requires a running Qdrant instance — it is marked [LIVE] and skipped when
the DB is unavailable.

Usage (from project root):
    python evaluation/multilingual/evaluate.py
    python evaluation/multilingual/evaluate.py --lang hi
    python evaluation/multilingual/evaluate.py --skip-live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from multilingual.schemas import Language
from multilingual.detector import LanguageDetector
from multilingual.provider import protect_terms, restore_terms, _PROTECTED_TERMS
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "src" / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _check_terms_preserved(text: str, expected_terms: list[str]) -> tuple[bool, list[str]]:
    """Return (all_preserved, missing_list)."""
    missing = [t for t in expected_terms if t not in text]
    return len(missing) == 0, missing


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Language detection
# ─────────────────────────────────────────────────────────────────────────────

def test_detection(cases: list[dict], expected_lang: Language) -> tuple[int, int]:
    detector = LanguageDetector()
    passed = failed = 0
    for case in cases:
        result = detector.detect(case["question"])
        ok = result.language == expected_lang
        symbol = "✓" if ok else "✗"
        print(
            f"    {symbol} [{result.language.value:2s} {result.confidence:.0%} "
            f"{result.method:18s}]  {case['question'][:55]}"
        )
        if ok:
            passed += 1
        else:
            print(f"         expected: {expected_lang.value}")
            failed += 1
    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Protected-term preservation (translation output)
# ─────────────────────────────────────────────────────────────────────────────

def test_protect_restore() -> tuple[int, int]:
    """Unit test: protect → restore round-trip."""
    passed = failed = 0
    test_texts = [
        "Section 3(p) of the Patents Act, 1970 relates to traditional knowledge.",
        "PCT application under TRIPS must comply with TKDL requirements.",
        "Trade Marks Act, 1999 Section 25(1)(k) and Copyright Act, 1957.",
        "IPC code A61K is relevant for Biological Diversity Act, 2002.",
    ]
    for text in test_texts:
        protected, mapping = protect_terms(text)
        restored = restore_terms(protected, mapping)
        ok = restored == text
        symbol = "✓" if ok else "✗"
        print(f"    {symbol} Protect/restore: {text[:60]}")
        if ok:
            passed += 1
        else:
            print(f"         got: {restored[:60]}")
            failed += 1
    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Normalisation (translation to English, keyword-free check)
# ─────────────────────────────────────────────────────────────────────────────

def test_normalisation(
    cases: list[dict],
    use_llm: bool = False,
) -> tuple[int, int]:
    """
    Check that after normalisation to English, the expected English keywords
    appear in the output.
    """
    if not use_llm:
        print("    [SKIP] Normalisation tests require --use-llm flag.")
        return 0, 0

    from multilingual.translator import MultilingualTranslator
    translator = MultilingualTranslator()
    detector   = LanguageDetector()
    passed = failed = 0

    for case in cases:
        lang_result   = detector.detect(case["question"])
        english_query = translator.normalize_query(case["question"], lang_result)
        english_equiv = case["english_equivalent"]

        # Check that important keywords from the English equivalent appear
        key_words = [w for w in english_equiv.lower().split() if len(w) > 4][:4]
        hits = sum(1 for kw in key_words if kw in english_query.lower())
        ok   = hits >= max(1, len(key_words) // 2)

        symbol = "✓" if ok else "✗"
        print(
            f"    {symbol} [{hits}/{len(key_words)} kw] "
            f"{case['question'][:40]} → {english_query[:50]}"
        )
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Live retrieval consistency (requires Qdrant + BM25 index)
# ─────────────────────────────────────────────────────────────────────────────

def test_live_retrieval(
    cases: list[dict],
    expected_lang: Language,
) -> tuple[int, int]:
    """
    Translate each query to English, run retrieval, check that the expected
    section / document is retrieved.
    """
    try:
        from multilingual.translator import MultilingualTranslator
        from multilingual.detector import LanguageDetector
        from retrieval.retriever import HybridRetriever
        from langchain_huggingface import HuggingFaceEmbeddings

        emb       = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever = HybridRetriever(embeddings=emb)
        translator= MultilingualTranslator()
        detector  = LanguageDetector()
    except Exception as e:
        print(f"    [SKIP] Live retrieval unavailable: {e}")
        return 0, 0

    passed = failed = 0
    for case in cases:
        expected_doc  = case.get("expected_document_id", "")
        expected_sec  = case.get("expected_section", "")

        lang_result   = detector.detect(case["question"])
        english_query = translator.normalize_query(case["question"], lang_result)
        result        = retriever.retrieve(english_query)
        chunks        = result["chunks"]

        # Check whether expected doc / section appears in top chunks
        doc_hit = any(expected_doc in getattr(c, "document_id", "") for c in chunks)
        sec_hit = (not expected_sec) or any(
            expected_sec in (getattr(c, "subsection", "") or "")
            or expected_sec in (getattr(c, "section", "") or "")
            for c in chunks
        )
        ok = doc_hit and sec_hit
        symbol = "✓" if ok else "✗"
        print(
            f"    {symbol} doc={doc_hit} sec={sec_hit}  "
            f"{case['question'][:50]}"
        )
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    lang_filter: Optional[str] = None,
    skip_live  : bool = False,
    use_llm    : bool = False,
) -> None:
    base = Path(__file__).parent
    corpus: dict[Language, str] = {
        Language.ENGLISH : str(base / "english.json"),
        Language.HINDI   : str(base / "hindi.json"),
        Language.KANNADA : str(base / "kannada.json"),
    }

    if lang_filter:
        try:
            target_lang = Language(lang_filter)
            corpus = {target_lang: corpus[target_lang]}
        except (ValueError, KeyError):
            print(f"Unknown language: {lang_filter}")
            sys.exit(1)

    print("=" * 70)
    print("IP-SAKTI  Phase 8 — Multilingual Evaluation")
    print("=" * 70)

    # Test 2: protect/restore is language-agnostic — run once
    print("\n── Protected-term round-trip ──────────────────────────────────────")
    pp, pf = test_protect_restore()
    print(f"  Result: {pp}/{pp+pf}")

    total_p = pp
    total_f = pf

    for lang, path in corpus.items():
        cases = _load_cases(path)
        print(f"\n── {lang.value.upper()} ({Language(lang.value).name}) — {len(cases)} cases ──────────")

        # Test 1: Detection
        print("  [Detection]")
        dp, df = test_detection(cases, lang)
        print(f"  Detection: {dp}/{dp+df}")

        # Test 3: Normalisation
        if lang != Language.ENGLISH:
            print("  [Normalisation to English]")
            np_, nf = test_normalisation(cases, use_llm=use_llm)
            print(f"  Normalisation: {np_}/{np_+nf}" if (np_+nf) > 0 else "  Normalisation: skipped")
        else:
            np_, nf = 0, 0

        # Test 4: Live retrieval
        if not skip_live:
            print("  [Live retrieval — requires Qdrant]")
            lp, lf = test_live_retrieval(cases, lang)
            print(f"  Live retrieval: {lp}/{lp+lf}" if (lp+lf) > 0 else "  Live retrieval: skipped")
        else:
            lp, lf = 0, 0

        total_p += dp + np_ + lp
        total_f += df + nf + lf

    total = total_p + total_f
    print(f"\n{'═'*70}")
    print(f"OVERALL: {total_p}/{total}  ({total_p/total*100:.0f}%)" if total > 0 else "OVERALL: 0 tests run")

    # Phase 8 success criteria
    detect_total = sum(
        len(_load_cases(p)) for p in corpus.values()
    )
    if total_p >= detect_total * 0.85:
        print("\n✓ Phase 8 detection target met (≥ 85%)")
    else:
        print("\n⚠  Below target — review language detection and keyword patterns")

    # Save results
    out = base / "eval_results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"total_passed": total_p, "total_failed": total_f}, fh, indent=2)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8 multilingual evaluation")
    parser.add_argument("--lang",       default=None, help="Filter: en | hi | kn")
    parser.add_argument("--skip-live",  action="store_true", help="Skip live Qdrant retrieval tests")
    parser.add_argument("--use-llm",    action="store_true", help="Enable LLM normalisation tests")
    args = parser.parse_args()
    run_evaluation(args.lang, args.skip_live, args.use_llm)
