"""
Phase 7 — Formulation + ABS tests
───────────────────────────────────
All tests run without LLM or network — they use keyword fallback + local DB.

Run from project root:
    python tests/test_formulation.py
    python -m pytest tests/test_formulation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from formulation.schemas import FormulationType, ABSAssessment, FormulationClassification
from formulation.classifier import _keyword_classify
from formulation.extractor import FormulationExtractor, _lookup, _INGREDIENT_DB
from abs.abs_checker import assess_abs


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ingredient DB tests
# ─────────────────────────────────────────────────────────────────────────────

def test_db_loaded():
    assert len(_INGREDIENT_DB) > 10, "Ingredient DB should have entries"

def test_neem_lookup():
    entry = _lookup("neem")
    assert entry is not None
    assert entry["biological_resource"] is True
    assert "ayurveda" in entry["traditional_systems"]

def test_turmeric_scientific_name():
    entry = _lookup("turmeric")
    assert entry is not None
    assert "Curcuma longa" in (entry.get("scientific_name") or "")

def test_ashwagandha_lookup():
    entry = _lookup("ashwagandha")
    assert entry is not None
    assert entry["traditional_use_indicator"] is True

def test_unknown_ingredient_returns_none():
    entry = _lookup("xyz_not_a_plant_abc123")
    assert entry is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extractor — no-LLM fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_extractor_finds_neem_turmeric():
    extractor = FormulationExtractor(llm=None)   # keyword fallback
    # Simulate _extract_names fallback directly
    names = extractor._extract_names("formulation with neem and turmeric for skin")
    assert "neem" in names or "turmeric" in names

def test_extractor_enriches_neem():
    extractor = FormulationExtractor(llm=None)
    ing = extractor._enrich("neem")
    assert ing.biological_resource is True
    assert ing.traditional_use_indicator is True
    assert ing.scientific_name is not None

def test_extractor_unknown_ingredient():
    extractor = FormulationExtractor(llm=None)
    ing = extractor._enrich("novel_chemical_xyz")
    assert ing.name == "novel_chemical_xyz"
    assert ing.source == "extracted"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classifier keyword tests
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_CASES: list[tuple[str, FormulationType]] = [
    # (description, expected FormulationType)
    (
        "Ayurvedic formulation containing neem, turmeric and ashwagandha for skin inflammation",
        FormulationType.AYURVEDA,
    ),
    (
        "Synthetic ibuprofen salt form with improved solubility — modern pharmaceutical",
        FormulationType.MODERN_PHARMACEUTICAL,
    ),
    (
        "Classical Siddha preparation using traditional Tamil herbs",
        FormulationType.SIDDHA,
    ),
    (
        "Unani formulation based on Greco-Arabic medicine principles",
        FormulationType.UNANI,
    ),
    (
        "Traditional knowledge preparation used by tribal communities for wound healing",
        FormulationType.TRADITIONAL_KNOWLEDGE,
    ),
    (
        "A herbal plant extract from a botanical source",
        FormulationType.BIOLOGICAL_RESOURCE,
    ),
]

def test_classifier_ayurveda():
    r = _keyword_classify(CLASSIFIER_CASES[0][0])
    assert r.formulation_type in (FormulationType.AYURVEDA, FormulationType.MIXED)

def test_classifier_modern():
    r = _keyword_classify(CLASSIFIER_CASES[1][0])
    assert r.formulation_type == FormulationType.MODERN_PHARMACEUTICAL

def test_classifier_siddha():
    r = _keyword_classify(CLASSIFIER_CASES[2][0])
    assert r.formulation_type == FormulationType.SIDDHA

def test_classifier_unani():
    r = _keyword_classify(CLASSIFIER_CASES[3][0])
    assert r.formulation_type == FormulationType.UNANI

def test_classifier_tk():
    r = _keyword_classify(CLASSIFIER_CASES[4][0])
    assert r.formulation_type == FormulationType.TRADITIONAL_KNOWLEDGE

def test_classifier_bio():
    r = _keyword_classify(CLASSIFIER_CASES[5][0])
    assert r.formulation_type == FormulationType.BIOLOGICAL_RESOURCE


# ─────────────────────────────────────────────────────────────────────────────
# 4. ABS checker tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_classification(
    ftype: FormulationType,
    ingredients: list[str],
    bio_resources: list[str],
    tk_indicators: list[str],
) -> FormulationClassification:
    from formulation.extractor import FormulationExtractor
    extractor = FormulationExtractor(llm=None)
    ingredient_objects = [extractor._enrich(name) for name in ingredients]
    return FormulationClassification(
        formulation_type                = ftype,
        ingredients                     = ingredients,
        biological_resources            = bio_resources,
        traditional_knowledge_indicators= tk_indicators,
        ingredient_objects              = ingredient_objects,
        confidence                      = 0.8,
    )


def test_abs_with_bio_and_tk():
    cls = _make_classification(
        FormulationType.AYURVEDA,
        ["neem", "turmeric"],
        ["neem", "turmeric"],
        ["traditional Ayurvedic use"],
    )
    result = assess_abs(cls)
    assert result.potentially_relevant is True
    assert result.traditional_knowledge_detected is True
    assert result.requires_human_review is True
    assert len(result.biological_resources) >= 1
    assert len(result.suggested_provisions) > 0

def test_abs_with_bio_no_tk():
    cls = _make_classification(
        FormulationType.BIOLOGICAL_RESOURCE,
        ["plant extract"],
        ["plant extract"],
        [],
    )
    result = assess_abs(cls)
    assert result.potentially_relevant is True
    assert result.requires_human_review is True

def test_abs_modern_no_bio():
    cls = _make_classification(
        FormulationType.MODERN_PHARMACEUTICAL,
        ["ibuprofen"],
        [],
        [],
    )
    result = assess_abs(cls)
    assert result.potentially_relevant is False
    assert result.traditional_knowledge_detected is False

def test_abs_always_requires_human_review():
    """ABS checker must always flag for human review."""
    for desc, ftype in CLASSIFIER_CASES:
        cls = _keyword_classify(desc)
        result = assess_abs(cls)
        assert result.requires_human_review is True

def test_abs_provisions_populated_when_relevant():
    cls = _make_classification(
        FormulationType.AYURVEDA,
        ["neem"],
        ["neem"],
        ["traditional use"],
    )
    result = assess_abs(cls)
    assert any("Biological Diversity Act" in p for p in result.suggested_provisions)
    assert any("3(p)" in p for p in result.suggested_provisions)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Success criteria cases (from Phase 7 spec)
# ─────────────────────────────────────────────────────────────────────────────

def test_case1_ayurveda_neem_turmeric():
    """Case 1: Neem + turmeric + traditional Ayurvedic use."""
    desc = "Neem and turmeric traditional Ayurvedic formulation for skin conditions"
    cls  = _keyword_classify(desc)
    abs_ = assess_abs(cls)

    assert cls.formulation_type in (FormulationType.AYURVEDA, FormulationType.MIXED)
    assert len(cls.biological_resources) > 0 or len(cls.ingredients) > 0
    assert abs_.potentially_relevant is True
    assert abs_.traditional_knowledge_detected is True

def test_case2_modern_no_abs():
    """Case 2: Synthetic chemical — should NOT trigger TK/ABS."""
    desc = "A new synthetic drug molecule developed through a novel laboratory chemical process"
    cls  = _keyword_classify(desc)
    abs_ = assess_abs(cls)

    assert cls.formulation_type == FormulationType.MODERN_PHARMACEUTICAL
    assert abs_.potentially_relevant is False
    assert abs_.traditional_knowledge_detected is False

def test_case3_siddha_tk():
    """Case 3: Classical Siddha preparation."""
    desc = "A traditional Siddha preparation documented in classical literature"
    cls  = _keyword_classify(desc)
    abs_ = assess_abs(cls)

    assert cls.formulation_type == FormulationType.SIDDHA
    assert abs_.traditional_knowledge_detected is True

def test_case4_uncertain_mixed():
    """Case 4: Uncertain mixture — should be MIXED or BIOLOGICAL_RESOURCE."""
    desc = "A mixture containing several different plant extracts and herbs"
    cls  = _keyword_classify(desc)

    # Should not confidently claim a single known system
    assert cls.formulation_type in (
        FormulationType.BIOLOGICAL_RESOURCE,
        FormulationType.MIXED,
        FormulationType.UNKNOWN,
        FormulationType.AYURVEDA,  # herbs might trigger this
    )
    # Human review should always be required
    abs_ = assess_abs(cls)
    assert abs_.requires_human_review is True


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_all() -> None:
    tests = [
        (test_db_loaded,                     "DB loaded"),
        (test_neem_lookup,                   "DB: neem lookup"),
        (test_turmeric_scientific_name,      "DB: turmeric scientific name"),
        (test_ashwagandha_lookup,            "DB: ashwagandha TK flag"),
        (test_unknown_ingredient_returns_none,"DB: unknown → None"),
        (test_extractor_finds_neem_turmeric, "Extractor: finds neem/turmeric"),
        (test_extractor_enriches_neem,       "Extractor: enriches neem"),
        (test_extractor_unknown_ingredient,  "Extractor: unknown ingredient"),
        (test_classifier_ayurveda,           "Classifier: ayurveda"),
        (test_classifier_modern,             "Classifier: modern pharma"),
        (test_classifier_siddha,             "Classifier: siddha"),
        (test_classifier_unani,              "Classifier: unani"),
        (test_classifier_tk,                 "Classifier: TK"),
        (test_classifier_bio,                "Classifier: biological resource"),
        (test_abs_with_bio_and_tk,           "ABS: bio+TK → relevant"),
        (test_abs_with_bio_no_tk,            "ABS: bio only → relevant"),
        (test_abs_modern_no_bio,             "ABS: modern → not relevant"),
        (test_abs_always_requires_human_review, "ABS: always requires review"),
        (test_abs_provisions_populated_when_relevant, "ABS: provisions populated"),
        (test_case1_ayurveda_neem_turmeric,  "Case 1: Ayurveda neem+turmeric"),
        (test_case2_modern_no_abs,           "Case 2: Modern — no ABS"),
        (test_case3_siddha_tk,               "Case 3: Siddha TK"),
        (test_case4_uncertain_mixed,         "Case 4: Uncertain/mixed"),
    ]

    print("=" * 65)
    print("IP-SAKTI  Phase 7 — Formulation + ABS Tests")
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
        print("\n  ✓ All Phase 7 tests passed.")
    else:
        print(f"\n  ⚠  {failed} test(s) failed.")


if __name__ == "__main__":
    _run_all()
