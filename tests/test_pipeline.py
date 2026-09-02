"""
Phase 11 — Pipeline tests  (no LLM / no network)

Run from project root:
    python tests/test_pipeline.py
    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.response_model    import PipelineResponse, CitationRecord, ConflictRecord
from pipeline.citation_verifier import (
    build_citations, build_sourced_context, verify_citations,
    extract_cited_ids,
)
from pipeline.confidence_engine import calculate_confidence, confidence_band
from pipeline.conflict_detector import detect_conflicts, prefer_authoritative
from evidence.store import EvidenceItem, EvidenceType, Authority


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _item(sid="ev1", stype=EvidenceType.LEGAL, authority="primary",
          score=0.8, section=None, subsection=None, text="",
          version=None) -> EvidenceItem:
    item = EvidenceItem(
        source_id=sid, source_type=stype, authority=authority,
        similarity_score=score, text=text or f"Evidence from {sid}.",
        section=section, subsection=subsection,
        title=f"Doc {sid}", source_name=f"Source {sid}",
        chunk_id=f"chunk_{sid}",
    )
    return item


# ─────────────────────────────────────────────────────────────────────────────
# 1. Response model tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_response_defaults():
    r = PipelineResponse(status="answered")
    assert r.status == "answered"
    assert r.citations == []
    assert r.confidence == 0.0
    assert r.needs_human_review is False

def test_citation_record_verified_default():
    c = CitationRecord(source_id="s1", document="Doc", source_name="SRC", jurisdiction="india", authority="primary")
    assert c.verified is True

def test_conflict_record_fields():
    c = ConflictRecord(source_a="s1", source_b="s2", description="test conflict")
    assert c.resolution == "prefer_latest_version"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Citation verifier tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_citations_from_items():
    items = [_item("patents_act_3p", section="Section 3", subsection="3(p)")]
    cits  = build_citations(items)
    assert len(cits) == 1
    assert cits[0].source_id == "patents_act_3p"
    assert cits[0].verified is True

def test_build_citations_deduplicates():
    items = [_item("ev1"), _item("ev1")]  # duplicate chunk_id
    cits  = build_citations(items)
    assert len(cits) == 1

def test_build_sourced_context_has_source_id():
    items = [_item("patents_act_1970_s3p", text="Section 3(p) traditional knowledge")]
    ctx   = build_sourced_context(items)
    assert "SOURCE_ID: PATENTS_ACT_1970_S3P" in ctx
    assert "Section 3(p)" in ctx

def test_extract_cited_ids_from_answer():
    answer = "Section 3(p) excludes TK. [SOURCE_ID: PATENTS_ACT_1970_S3P] Also relevant: [SOURCE_ID: TKDL_001]"
    ids    = extract_cited_ids(answer)
    assert "PATENTS_ACT_1970_S3P" in ids
    assert "TKDL_001" in ids

def test_verify_citations_all_found():
    items  = [_item("patents_act_1970_s3p")]
    answer = "Section 3(p) is relevant. [SOURCE_ID: PATENTS_ACT_1970_S3P]"
    cits, coverage = verify_citations(answer, items)
    assert coverage == 1.0
    assert all(c.verified for c in cits)

def test_verify_citations_not_found():
    items  = [_item("ev1")]
    answer = "Section 3(p) is relevant. [SOURCE_ID: INVENTED_SOURCE]"
    cits, coverage = verify_citations(answer, items)
    # Coverage < 1.0 because INVENTED_SOURCE is not in items
    assert coverage < 1.0
    invented = [c for c in cits if not c.verified]
    assert len(invented) > 0

def test_verify_citations_no_ids_in_answer():
    items  = [_item("ev1"), _item("ev2")]
    answer = "Section 3(p) is relevant."   # no SOURCE_IDs
    cits, coverage = verify_citations(answer, items)
    # No citations in answer → build from all items, coverage=1.0
    assert coverage == 1.0
    assert len(cits) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confidence engine tests
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_high_all_signals():
    items = [_item("l1", authority="primary", score=0.9),
             _item("l2", authority="primary", score=0.85)]
    cits  = [CitationRecord(source_id="l1", document="D", source_name="S",
                            jurisdiction="india", authority="primary", verified=True)]
    score, band, _ = calculate_confidence(items, "high", 1.0, cits, 0)
    assert band == "high"
    assert score >= 0.80

def test_confidence_low_empty():
    score, band, _ = calculate_confidence([], "low", 0.0, [], 0)
    assert band == "low"

def test_confidence_reduced_by_conflicts():
    items  = [_item("l1", authority="primary", score=0.9)]
    score_no_conflict, _, _ = calculate_confidence(items, "high", 1.0, [], 0)
    score_conflict,    _, _ = calculate_confidence(items, "high", 1.0, [], 3)
    assert score_conflict < score_no_conflict

def test_confidence_band_boundaries():
    assert confidence_band(0.85) == "high"
    assert confidence_band(0.80) == "high"
    assert confidence_band(0.79) == "medium"
    assert confidence_band(0.60) == "medium"
    assert confidence_band(0.59) == "low"

def test_confidence_penalised_by_unverified_citations():
    items = [_item("l1", authority="primary", score=0.9)]
    good_cit = CitationRecord(source_id="l1", document="D", source_name="S",
                              jurisdiction="india", authority="primary", verified=True)
    bad_cit  = CitationRecord(source_id="x",  document="X", source_name="?",
                              jurisdiction="unknown", authority="unknown", verified=False)
    score_good, _, _ = calculate_confidence(items, "high", 1.0, [good_cit], 0)
    score_bad,  _, _ = calculate_confidence(items, "high", 1.0, [bad_cit],  0)
    assert score_good > score_bad


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conflict detector tests
# ─────────────────────────────────────────────────────────────────────────────

def test_no_conflicts_clean_items():
    items    = [_item("l1", text="Section 3(p) is relevant."),
                _item("l2", text="Traditional knowledge is excluded.")]
    conflicts = detect_conflicts(items)
    assert len(conflicts) == 0

def test_conflict_detected_contradictory_statements():
    items = [
        _item("s1", text="The invention is patentable under Indian law."),
        _item("s2", text="The invention is not patentable under Indian law."),
    ]
    conflicts = detect_conflicts(items)
    assert len(conflicts) >= 1

def test_prefer_authoritative_sorts_correctly():
    items = [
        _item("secondary", authority="secondary", score=0.9),
        _item("primary",   authority="primary",   score=0.5),
    ]
    sorted_items = prefer_authoritative(items)
    assert sorted_items[0].authority == "primary"

def test_conflict_deduplication():
    """Same source pair should produce only one conflict."""
    items = [
        _item("s1", text="invention is patentable"),
        _item("s2", text="invention is not patentable"),
    ]
    conflicts = detect_conflicts(items)
    pairs = [(c.source_a, c.source_b) for c in conflicts]
    unique_pairs = list({frozenset(p) for p in pairs})
    assert len(unique_pairs) == len(conflicts)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline response model serialisation
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_response_serialises():
    r = PipelineResponse(
        status="answered",
        answer="Section 3(p) is relevant.",
        language="en",
        confidence=0.85,
        confidence_band="high",
        citations=[
            CitationRecord(source_id="ev1", document="Patents Act",
                           source_name="India Code", jurisdiction="india",
                           authority="primary")
        ],
    )
    d = r.model_dump()
    assert d["status"] == "answered"
    assert d["confidence"] == 0.85
    assert len(d["citations"]) == 1
    assert d["citations"][0]["verified"] is True

def test_abstained_response_has_no_answer():
    r = PipelineResponse(status="abstained", needs_human_review=True)
    assert r.answer is None
    assert r.needs_human_review is True

def test_out_of_scope_response():
    r = PipelineResponse(status="out_of_scope", needs_human_review=False)
    assert r.needs_human_review is False


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_all() -> None:
    tests = [
        (test_pipeline_response_defaults,              "Response: defaults"),
        (test_citation_record_verified_default,        "Response: citation verified=True"),
        (test_conflict_record_fields,                  "Response: conflict record"),
        (test_build_citations_from_items,              "Citation: build from items"),
        (test_build_citations_deduplicates,            "Citation: deduplicates"),
        (test_build_sourced_context_has_source_id,     "Citation: context has SOURCE_ID"),
        (test_extract_cited_ids_from_answer,           "Citation: extract IDs"),
        (test_verify_citations_all_found,              "Citation: all found → 100%"),
        (test_verify_citations_not_found,              "Citation: not found → unverified"),
        (test_verify_citations_no_ids_in_answer,       "Citation: no IDs → build all"),
        (test_confidence_high_all_signals,             "Confidence: high all signals"),
        (test_confidence_low_empty,                    "Confidence: low empty"),
        (test_confidence_reduced_by_conflicts,         "Confidence: reduced by conflicts"),
        (test_confidence_band_boundaries,              "Confidence: band boundaries"),
        (test_confidence_penalised_by_unverified_citations, "Confidence: unverified penalty"),
        (test_no_conflicts_clean_items,                "Conflict: clean → no conflicts"),
        (test_conflict_detected_contradictory_statements, "Conflict: contradiction detected"),
        (test_prefer_authoritative_sorts_correctly,    "Conflict: prefer_authoritative"),
        (test_conflict_deduplication,                  "Conflict: deduplication"),
        (test_pipeline_response_serialises,            "Pipeline: serialises to dict"),
        (test_abstained_response_has_no_answer,        "Pipeline: abstained has no answer"),
        (test_out_of_scope_response,                   "Pipeline: out_of_scope"),
    ]

    print("=" * 65)
    print("IP-SAKTI  Phase 11 — Pipeline Tests")
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
        print("\n  ✓ All Phase 11 tests passed.")
    else:
        print(f"\n  ⚠  {failed} test(s) failed.")


if __name__ == "__main__":
    _run_all()
