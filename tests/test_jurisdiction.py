"""
Phase 6 — Jurisdiction Router tests
──────────────────────────────────────
All tests use the keyword-fallback classifier only — no LLM, no network.

Run from project root:
    python tests/test_jurisdiction.py
    python -m pytest tests/test_jurisdiction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from routing.jurisdiction import _keyword_classify, Jurisdiction, JurisdictionRouter


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

INDIA_CASES: list[str] = [
    "What does Section 3(p) of the Patents Act say?",
    "How do I register a trademark in India?",
    "What are the patentability requirements under Indian law?",
    "Explain the Copyright Act 1957.",
    "What is the TKDL and how is it used by the Indian Patent Office?",
    "Can I register a geographical indication in India?",
    "What does IP India say about design registration?",
    "Which Indian court handles patent disputes?",
    "What does AYUSH say about traditional formulations?",
    "Explain Section 29 of the Trade Marks Act 1999.",
]

INTERNATIONAL_CASES: list[str] = [
    "What is the PCT and how does it work?",
    "Explain the TRIPS Agreement.",
    "How does the Paris Convention protect industrial property?",
    "What is the Madrid System for trademark registration?",
    "Explain the Berne Convention for copyright.",
    "How do I file an international patent application with WIPO?",
    "What are the requirements for PCT national phase entry?",
    "Does the TRIPS Agreement cover traditional knowledge?",
    "What is the Hague System for international design registration?",
    "How does the Madrid Protocol differ from the Madrid Agreement?",
]

BOTH_CASES: list[str] = [
    "Compare Indian patent law with the PCT.",
    "How does India's copyright protection compare with the Berne Convention?",
    "What are the differences between Indian trademark law and the Madrid System?",
    "How does India's Patents Act relate to TRIPS obligations?",
    "Compare Indian GI protection with international geographical indication rules.",
    "How does Indian traditional knowledge protection interact with international IP law?",
]

OVERRIDE_CASES: list[tuple[str, str, Jurisdiction]] = [
    # (query, override, expected)
    ("What is patentability?",         "india",         Jurisdiction.INDIA),
    ("What is patentability?",         "international", Jurisdiction.INTERNATIONAL),
    ("How does IP law work?",          "both",          Jurisdiction.BOTH),
    ("Explain trademark registration", "INDIA",         Jurisdiction.INDIA),
    ("PCT application procedure",      "India",         Jurisdiction.INDIA),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_cases(
    cases: list[str],
    expected: Jurisdiction,
    label: str,
) -> tuple[int, int]:
    passed = failed = 0
    print(f"\n── {label} ──────────────────────────────────────────────────")
    for query in cases:
        result = _keyword_classify(query)
        ok     = result.jurisdiction == expected
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} [{result.jurisdiction.value:13s}  {result.confidence:.0%}]  {query[:60]}")
        if ok:
            passed += 1
        else:
            print(f"       expected: {expected.value}")
            failed += 1
    return passed, failed


def _run_override_cases() -> tuple[int, int]:
    passed = failed = 0
    router = JurisdictionRouter(llm=None)
    print("\n── User override (explicit jurisdiction) ────────────────────")
    for query, override, expected in OVERRIDE_CASES:
        result = router.classify(query, override=override)
        ok     = result.jurisdiction == expected
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} override='{override:13s}' → [{result.jurisdiction.value}]  {query[:45]}")
        if ok:
            passed += 1
        else:
            print(f"       expected: {expected.value}")
            failed += 1
    return passed, failed


def main() -> None:
    print("=" * 65)
    print("IP-SAKTI  Phase 6 — Jurisdiction Router Tests")
    print("=" * 65)

    ip, if_ = _run_cases(INDIA_CASES,         Jurisdiction.INDIA,         "India queries")
    pp, pf  = _run_cases(INTERNATIONAL_CASES, Jurisdiction.INTERNATIONAL, "International queries")
    bp, bf  = _run_cases(BOTH_CASES,           Jurisdiction.BOTH,          "Both / comparison queries")
    op, of_ = _run_override_cases()

    total_p = ip + pp + bp + op
    total_f = if_ + pf + bf + of_
    total   = total_p + total_f

    print(f"\n{'─'*65}")
    print(f"India               : {ip}/{ip+if_}")
    print(f"International       : {pp}/{pp+pf}")
    print(f"Both / comparison   : {bp}/{bp+bf}")
    print(f"Override            : {op}/{op+of_}")
    print(f"{'─'*65}")
    print(f"TOTAL               : {total_p}/{total}  ({total_p/total*100:.0f}%)")

    if total_f == 0:
        print("\n✓ All jurisdiction tests passed.")
    else:
        print(f"\n⚠  {total_f} test(s) failed.")


if __name__ == "__main__":
    main()


# ── pytest-compatible test functions ─────────────────────────────────────────

def test_india_section3p():
    r = _keyword_classify("What does Section 3(p) of the Patents Act say?")
    assert r.jurisdiction == Jurisdiction.INDIA

def test_india_trademark_act():
    r = _keyword_classify("Explain Section 29 of the Trade Marks Act 1999.")
    assert r.jurisdiction == Jurisdiction.INDIA

def test_india_ayush():
    r = _keyword_classify("What does AYUSH say about traditional formulations?")
    assert r.jurisdiction == Jurisdiction.INDIA

def test_india_tkdl():
    r = _keyword_classify("What is the TKDL and how is it used by the Indian Patent Office?")
    assert r.jurisdiction == Jurisdiction.INDIA

def test_intl_pct():
    r = _keyword_classify("What is the PCT and how does it work?")
    assert r.jurisdiction == Jurisdiction.INTERNATIONAL

def test_intl_trips():
    r = _keyword_classify("Explain the TRIPS Agreement.")
    assert r.jurisdiction == Jurisdiction.INTERNATIONAL

def test_intl_paris():
    r = _keyword_classify("How does the Paris Convention protect industrial property?")
    assert r.jurisdiction == Jurisdiction.INTERNATIONAL

def test_intl_wipo():
    r = _keyword_classify("How do I file an international patent application with WIPO?")
    assert r.jurisdiction == Jurisdiction.INTERNATIONAL

def test_both_compare_pct():
    r = _keyword_classify("Compare Indian patent law with the PCT.")
    assert r.jurisdiction == Jurisdiction.BOTH

def test_both_trips_india():
    r = _keyword_classify("How does India's Patents Act relate to TRIPS obligations?")
    assert r.jurisdiction == Jurisdiction.BOTH

def test_override_india():
    router = JurisdictionRouter(llm=None)
    r = router.classify("What is patentability?", override="india")
    assert r.jurisdiction == Jurisdiction.INDIA
    assert r.confidence == 1.0

def test_override_international():
    router = JurisdictionRouter(llm=None)
    r = router.classify("What is patentability?", override="international")
    assert r.jurisdiction == Jurisdiction.INTERNATIONAL
    assert r.confidence == 1.0

def test_override_both():
    router = JurisdictionRouter(llm=None)
    r = router.classify("How does IP law work?", override="both")
    assert r.jurisdiction == Jurisdiction.BOTH
    assert r.confidence == 1.0

def test_override_case_insensitive():
    router = JurisdictionRouter(llm=None)
    r = router.classify("Patent law", override="INDIA")
    assert r.jurisdiction == Jurisdiction.INDIA

def test_default_to_india_when_no_signal():
    r = _keyword_classify("What is the best way to protect my invention?")
    # No strong signal → defaults to india
    assert r.jurisdiction == Jurisdiction.INDIA
