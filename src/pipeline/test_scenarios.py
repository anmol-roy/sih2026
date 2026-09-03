"""
Phase 11 — 6 Representative Manual Test Scenarios
──────────────────────────────────────────────────
These tests verify the routing, scope, and tool-selection layers
without making LLM or network calls.

Each scenario asserts the deterministic parts of the pipeline:
  - language detection
  - scope classification
  - IP type routing
  - jurisdiction routing
  - tool selection

Run from project root:
    python src/pipeline/test_scenarios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from multilingual.detector      import LanguageDetector
from multilingual.schemas       import Language
from guardrails.scope           import ScopeChecker, Scope
from routing.ip_router          import IPRouter, _keyword_classify as ip_kw
from routing.jurisdiction       import JurisdictionRouter, Jurisdiction, _keyword_classify as jur_kw
from agents.orchestrator        import select_tools


# ─────────────────────────────────────────────────────────────────────────────
# Shared instances
# ─────────────────────────────────────────────────────────────────────────────

_detector  = LanguageDetector()
_scope     = ScopeChecker(llm=None)
_jur       = JurisdictionRouter(llm=None)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner helper
# ─────────────────────────────────────────────────────────────────────────────

def _run(
    name          : str,
    query         : str,
    expect_scope  : Scope,
    expect_lang   : Language,
    expect_jur    : Jurisdiction,
    expect_tools  : list[str],          # must be present
    no_tools      : list[str] | None = None,
) -> bool:
    print(f"\n{'─'*60}")
    print(f"  Scenario: {name}")
    print(f"  Query   : {query[:75]}")

    passed = True
    results: list[str] = []

    # Language
    lang = _detector.detect(query)
    ok   = lang.language == expect_lang
    results.append(f"  lang={lang.language.value}  {'✓' if ok else f'✗ expected {expect_lang.value}'}")
    if not ok: passed = False

    # Scope
    scope = _scope.check(query)
    ok    = scope.scope == expect_scope
    results.append(f"  scope={scope.scope.value}  {'✓' if ok else f'✗ expected {expect_scope.value}'}")
    if not ok: passed = False

    # Jurisdiction (only test when scope is in_scope)
    if expect_scope != Scope.OUT_OF_SCOPE:
        jur = _jur.classify(query)
        ok  = jur.jurisdiction == expect_jur
        results.append(f"  jur={jur.jurisdiction.value}  {'✓' if ok else f'✗ expected {expect_jur.value}'}")
        if not ok: passed = False

        # Tools
        selected = [t.split(":")[0] for t in select_tools(query)]
        missing  = [t for t in expect_tools  if t not in selected]
        extra    = [t for t in (no_tools or []) if t in selected]
        results.append(f"  tools={selected}")
        if missing:
            results.append(f"  ✗ MISSING tools: {missing}")
            passed = False
        if extra:
            results.append(f"  ✗ UNEXPECTED tools: {extra}")
            passed = False
        if not missing and not extra:
            results.append("  tools: ✓")

    for r in results:
        print(r)

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# The 6 scenarios
# ─────────────────────────────────────────────────────────────────────────────

def scenario_1_legal_query() -> bool:
    """Test 1 — Simple legal query → Legal RAG only."""
    return _run(
        name        = "1. Simple legal — Section 3(p)",
        query       = "What does Section 3(p) of the Patents Act say?",
        expect_scope= Scope.IN_SCOPE,
        expect_lang = Language.ENGLISH,
        expect_jur  = Jurisdiction.INDIA,
        expect_tools= ["legal_search"],
        no_tools    = ["formulation_classifier", "international_search", "abs_check"],
    )


def scenario_2_herbal_formulation() -> bool:
    """Test 2 — Herbal formulation → Patent+Formulation+TK+PriorArt+Legal."""
    return _run(
        name        = "2. Herbal formulation patentability",
        query       = "Can I patent my herbal formulation containing neem and turmeric?",
        expect_scope= Scope.IN_SCOPE,
        expect_lang = Language.ENGLISH,
        expect_jur  = Jurisdiction.INDIA,
        expect_tools= ["legal_search", "formulation_classifier", "tk_search", "patent_search"],
        no_tools    = ["international_search"],
    )


def scenario_3_pct_international() -> bool:
    """Test 3 — PCT question → International search only."""
    return _run(
        name        = "3. International — PCT",
        query       = "What is the PCT and how does it work for international patent filing?",
        expect_scope= Scope.IN_SCOPE,
        expect_lang = Language.ENGLISH,
        expect_jur  = Jurisdiction.INTERNATIONAL,
        expect_tools= ["international_search"],
        no_tools    = ["formulation_classifier", "abs_check", "tk_search"],
    )


def scenario_4_hindi_query() -> bool:
    """Test 4 — Hindi query → detected as Hindi, scope in_scope."""
    query = "पेटेंट अधिनियम की धारा 3(p) क्या कहती है?"
    print(f"\n{'─'*60}")
    print(f"  Scenario: 4. Hindi formulation question")
    print(f"  Query   : {query}")

    passed = True

    lang = _detector.detect(query)
    ok   = lang.language == Language.HINDI
    print(f"  lang={lang.language.value}  {'✓' if ok else '✗ expected hi'}")
    if not ok: passed = False

    scope = _scope.check(query)
    ok    = scope.scope == Scope.IN_SCOPE
    print(f"  scope={scope.scope.value}  {'✓' if ok else '✗ expected in_scope'}")
    if not ok: passed = False

    # After normalisation to English it should route to India + patent
    english = "What does Section 3(p) of the Patents Act say?"
    jur   = _jur.classify(english)
    ok    = jur.jurisdiction == Jurisdiction.INDIA
    print(f"  jur (after normalise)={jur.jurisdiction.value}  {'✓' if ok else '✗ expected india'}")
    if not ok: passed = False

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}")
    return passed


def scenario_5_out_of_scope() -> bool:
    """Test 5 — Random unrelated question → OUT_OF_SCOPE."""
    query = "Who will win today's cricket match?"
    print(f"\n{'─'*60}")
    print(f"  Scenario: 5. Out of scope — cricket")
    print(f"  Query   : {query}")

    scope  = _scope.check(query)
    ok     = scope.scope == Scope.OUT_OF_SCOPE
    result = "✓ PASS" if ok else "✗ FAIL"
    print(f"  scope={scope.scope.value}  {result}")
    return ok


def scenario_6_abstention_trigger() -> bool:
    """Test 6 — Very obscure query → low confidence path (tool selection check)."""
    query = "What is the exact wording of sub-rule 3 of Rule 138A of the Patent Rules 2003?"
    print(f"\n{'─'*60}")
    print(f"  Scenario: 6. Obscure specific rule — should route to legal_search")
    print(f"  Query   : {query[:70]}")

    scope = _scope.check(query)
    ok1   = scope.scope == Scope.IN_SCOPE
    print(f"  scope={scope.scope.value}  {'✓' if ok1 else '✗'}")

    tools    = [t.split(":")[0] for t in select_tools(query)]
    ok2      = "legal_search" in tools
    print(f"  tools={tools}  legal_search={'✓' if ok2 else '✗'}")

    passed = ok1 and ok2
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios() -> None:
    print("=" * 60)
    print("  IP-SAKTI Phase 11 — 6 Representative Scenarios")
    print("  (Deterministic routing tests — no LLM / no network)")
    print("=" * 60)

    scenarios = [
        scenario_1_legal_query,
        scenario_2_herbal_formulation,
        scenario_3_pct_international,
        scenario_4_hindi_query,
        scenario_5_out_of_scope,
        scenario_6_abstention_trigger,
    ]

    results = [fn() for fn in scenarios]

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total}  ({passed/total*100:.0f}%)")
    if passed == total:
        print("  ✓ All 6 representative scenarios passed.")
    else:
        print(f"  ⚠  {total-passed} scenario(s) failed.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_scenarios()
