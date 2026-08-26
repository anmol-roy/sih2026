"""
Phase 5 — IP Router smoke tests
────────────────────────────────
Tests the router's keyword fallback (no LLM call needed) so these
run instantly without network access.

Run from the project root:
    python tests/test_router.py
    python -m pytest tests/test_router.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from routing.ip_router import _keyword_classify
from routing.schemas import IPType
from formulation.classifier import _keyword_classify as _form_kw
from formulation.schemas import FormulationType


# ─────────────────────────────────────────────────────────────────────────────
# Router keyword tests
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_CASES: list[tuple[str, IPType]] = [
    # (query, expected primary IPType)
    ("Can I patent my herbal formulation?",                          IPType.PATENT),
    ("What does Section 3(p) of the Patents Act say?",               IPType.PATENT),
    ("Is neem turmeric formulation traditional knowledge?",          IPType.TRADITIONAL_KNOWLEDGE),
    ("Can I register my company name as a trademark?",               IPType.TRADEMARK),
    ("Who owns copyright in the software I wrote at work?",          IPType.COPYRIGHT),
    ("Can I protect the shape of my product bottle?",                IPType.DESIGN),
    ("Can Darjeeling tea receive geographical indication protection?",IPType.GI),
    ("What is the TKDL and how does it relate to patents?",          IPType.TRADITIONAL_KNOWLEDGE),
    ("Is this Ayurvedic composition patentable?",                    IPType.PATENT),
    ("How do I register a trademark for my brand logo?",             IPType.TRADEMARK),
    ("What rights does a copyright holder have in India?",           IPType.COPYRIGHT),
    ("How is a design registered under the Designs Act?",            IPType.DESIGN),
    ("What products can be registered as geographical indications?",  IPType.GI),
    ("Does Unani medicine qualify as traditional knowledge?",         IPType.TRADITIONAL_KNOWLEDGE),
]

# ─────────────────────────────────────────────────────────────────────────────
# Formulation classifier keyword tests
# ─────────────────────────────────────────────────────────────────────────────

FORMULATION_CASES: list[tuple[str, FormulationType]] = [
    (
        "A formulation containing neem, turmeric and ashwagandha traditionally "
        "used for skin inflammation.",
        FormulationType.AYURVEDA,
    ),
    (
        "A synthetic drug compound based on ibuprofen with modified salt form.",
        FormulationType.MODERN_PHARMACEUTICAL,
    ),
    (
        "A Siddha preparation using local Tamil herbs for fever treatment.",
        FormulationType.SIDDHA,
    ),
    (
        "A Unani formulation for digestive disorders based on Greek medicine.",
        FormulationType.UNANI,
    ),
    (
        "A traditional knowledge preparation used by tribal communities "
        "for wound healing.",
        FormulationType.TRADITIONAL_KNOWLEDGE,
    ),
    (
        "A herbal extract from a plant species found in the Western Ghats.",
        FormulationType.BIOLOGICAL_RESOURCE,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Multi-domain detection test (via IPRouter keyword fallback)
# ─────────────────────────────────────────────────────────────────────────────

MULTI_DOMAIN_CASES: list[tuple[str, list[IPType]]] = [
    (
        "Can I trademark the name of my patented herbal product?",
        [IPType.PATENT, IPType.TRADEMARK],
    ),
    (
        "Is the TKDL prior art that prevents me from patenting this traditional formulation?",
        [IPType.TRADITIONAL_KNOWLEDGE, IPType.PATENT],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_router_tests() -> tuple[int, int]:
    passed = failed = 0
    print("\n── IP Router (keyword fallback) ──────────────────────────────")
    for query, expected in ROUTER_CASES:
        result = _keyword_classify(query)
        got    = result[0]
        ok     = got == expected
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} [{got.value:22s}] {query[:65]}")
        if ok:
            passed += 1
        else:
            print(f"       expected: {expected.value}")
            failed += 1
    return passed, failed


def _run_formulation_tests() -> tuple[int, int]:
    passed = failed = 0
    print("\n── Formulation Classifier (keyword fallback) ─────────────────")
    for desc, expected in FORMULATION_CASES:
        result = _form_kw(desc)
        got    = result.formulation_type
        ok     = got == expected
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} [{got.value:25s}] {desc[:55]}")
        if ok:
            passed += 1
        else:
            print(f"       expected: {expected.value}")
            failed += 1
    return passed, failed


def _run_multi_domain_tests() -> tuple[int, int]:
    passed = failed = 0
    print("\n── Multi-domain detection (keyword fallback) ─────────────────")
    for query, expected_types in MULTI_DOMAIN_CASES:
        result = _keyword_classify(query)
        # Check all expected types are somewhere in result
        found  = all(t in result for t in expected_types)
        symbol = "✓" if found else "✗"
        print(f"  {symbol} [{', '.join(t.value for t in result)}]")
        print(f"      {query[:65]}")
        if found:
            passed += 1
        else:
            print(f"      expected: {[t.value for t in expected_types]}")
            failed += 1
    return passed, failed


def main() -> None:
    print("=" * 65)
    print("IP-SAKTI  Phase 5 — Router & Formulation Classifier Tests")
    print("=" * 65)

    rp, rf = _run_router_tests()
    fp, ff = _run_formulation_tests()
    mp, mf = _run_multi_domain_tests()

    total_p = rp + fp + mp
    total_f = rf + ff + mf
    total   = total_p + total_f

    print(f"\n{'─'*65}")
    print(f"Router            : {rp}/{rp+rf}")
    print(f"Formulation       : {fp}/{fp+ff}")
    print(f"Multi-domain      : {mp}/{mp+mf}")
    print(f"{'─'*65}")
    print(f"TOTAL             : {total_p}/{total}  ({total_p/total*100:.0f}%)")

    if total_f == 0:
        print("\n✓ All keyword-fallback tests passed.")
    else:
        print(f"\n⚠  {total_f} test(s) failed — review keyword patterns in ip_router.py")


if __name__ == "__main__":
    main()


# ── pytest-compatible test functions ─────────────────────────────────────────

def test_patent_routing():
    result = _keyword_classify("Can I patent my herbal formulation?")
    assert IPType.PATENT in result

def test_trademark_routing():
    result = _keyword_classify("How do I register a trademark for my brand?")
    assert IPType.TRADEMARK in result

def test_copyright_routing():
    result = _keyword_classify("Who owns copyright in my software?")
    assert IPType.COPYRIGHT in result

def test_design_routing():
    result = _keyword_classify("Can I protect the shape of my bottle?")
    assert IPType.DESIGN in result

def test_gi_routing():
    result = _keyword_classify("Can Darjeeling tea get geographical indication?")
    assert IPType.GI in result

def test_tk_routing():
    result = _keyword_classify("Is this Ayurvedic formula traditional knowledge?")
    assert IPType.TRADITIONAL_KNOWLEDGE in result

def test_formulation_ayurveda():
    result = _form_kw("Neem, turmeric and ashwagandha for skin inflammation")
    assert result.formulation_type in (
        FormulationType.AYURVEDA, FormulationType.TRADITIONAL_KNOWLEDGE,
        FormulationType.BIOLOGICAL_RESOURCE, FormulationType.MIXED
    )

def test_formulation_modern():
    result = _form_kw("Synthetic ibuprofen salt form pharmaceutical compound")
    assert result.formulation_type == FormulationType.MODERN_PHARMACEUTICAL
