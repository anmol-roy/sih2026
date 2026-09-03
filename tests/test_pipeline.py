"""
Phase 11 — Complete Pipeline Tests  (no LLM / no network)

Covers both halves:
  - Part 1 (first half): response model, citations, confidence, conflicts
  - Part 2 (second half): version filter, answer formatter, debug logger, 6 scenarios
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
    build_citations, build_sourced_context, verify_citations, extract_cited_ids,
)
from pipeline.confidence_engine import calculate_confidence, confidence_band
from pipeline.conflict_detector import detect_conflicts, prefer_authoritative
from pipeline.version_filter    import filter_by_version, is_historical_query, prefer_current_versions
from pipeline.answer_formatter  import (
    extract_sections, has_disclaimer_section, ensure_disclaimer_section,
    STRUCTURED_ANSWER_SYSTEM,
)
from pipeline.debug_logger      import PipelineDebugLogger, make_logger
from evidence.store             import EvidenceItem, EvidenceType, Authority


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _item(sid="ev1", stype=EvidenceType.LEGAL, authority="primary",
          score=0.8, section=None, subsection=None, text="") -> EvidenceItem:
    return EvidenceItem(
        source_id=sid, source_type=stype, authority=authority,
        similarity_score=score, text=text or f"Evidence from {sid}.",
        section=section, subsection=subsection,
        title=f"Doc {sid}", source_name=f"Source {sid}",
        chunk_id=f"chunk_{sid}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Response model
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_response_defaults():
    r = PipelineResponse(status="answered")
    assert r.status == "answered"
    assert r.citations == []
    assert r.confidence == 0.0
    assert r.needs_human_review is False

def test_citation_record_verified_default():
    c = CitationRecord(source_id="s1", document="Doc", source_name="SRC",
                       jurisdiction="india", authority="primary")
    assert c.verified is True

def test_conflict_record_fields():
    c = ConflictRecord(source_a="s1", source_b="s2", description="test conflict")
    assert c.resolution == "prefer_latest_version"


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Citation verifier
# ─────────────────────────────────────────────────────────────────────────────

def test_build_citations_from_items():
    items = [_item("patents_act_3p", section="Section 3", subsection="3(p)")]
    cits  = build_citations(items)
    assert len(cits) == 1
    assert cits[0].source_id == "patents_act_3p"
    assert cits[0].verified is True

def test_build_citations_deduplicates():
    cits = build_citations([_item("ev1"), _item("ev1")])
    assert len(cits) == 1

def test_build_sourced_context_has_source_id():
    items = [_item("patents_act_1970_s3p", text="Section 3(p) traditional knowledge")]
    ctx   = build_sourced_context(items)
    assert "SOURCE_ID: PATENTS_ACT_1970_S3P" in ctx

def test_extract_cited_ids_from_answer():
    answer = "Relevant. [SOURCE_ID: PATENTS_ACT_1970_S3P] Also: [SOURCE_ID: TKDL_001]"
    ids    = extract_cited_ids(answer)
    assert "PATENTS_ACT_1970_S3P" in ids
    assert "TKDL_001" in ids

def test_verify_citations_all_found():
    items = [_item("patents_act_1970_s3p")]
    cits, coverage = verify_citations("Relevant. [SOURCE_ID: PATENTS_ACT_1970_S3P]", items)
    assert coverage == 1.0
    assert all(c.verified for c in cits)

def test_verify_citations_not_found():
    cits, coverage = verify_citations("[SOURCE_ID: INVENTED_SOURCE]", [_item("ev1")])
    assert coverage < 1.0
    assert any(not c.verified for c in cits)

def test_verify_citations_no_ids():
    cits, coverage = verify_citations("No IDs here.", [_item("ev1"), _item("ev2")])
    assert coverage == 1.0
    assert len(cits) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Confidence engine
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_high_all_signals():
    items = [_item("l1", authority="primary", score=0.9),
             _item("l2", authority="primary", score=0.85)]
    cit   = CitationRecord(source_id="l1", document="D", source_name="S",
                           jurisdiction="india", authority="primary", verified=True)
    score, band, _ = calculate_confidence(items, "high", 1.0, [cit], 0)
    assert band == "high"
    assert score >= 0.80

def test_confidence_low_empty():
    score, band, _ = calculate_confidence([], "low", 0.0, [], 0)
    assert band == "low"

def test_confidence_reduced_by_conflicts():
    items = [_item("l1", authority="primary", score=0.9)]
    s0, _, _ = calculate_confidence(items, "high", 1.0, [], 0)
    s3, _, _ = calculate_confidence(items, "high", 1.0, [], 3)
    assert s3 < s0

def test_confidence_band_boundaries():
    assert confidence_band(0.85) == "high"
    assert confidence_band(0.80) == "high"
    assert confidence_band(0.79) == "medium"
    assert confidence_band(0.60) == "medium"
    assert confidence_band(0.59) == "low"

def test_confidence_penalised_by_unverified():
    items   = [_item("l1", authority="primary", score=0.9)]
    good    = CitationRecord(source_id="l1", document="D", source_name="S",
                             jurisdiction="india", authority="primary", verified=True)
    bad     = CitationRecord(source_id="x",  document="X", source_name="?",
                             jurisdiction="unknown", authority="unknown", verified=False)
    sg, _, _ = calculate_confidence(items, "high", 1.0, [good], 0)
    sb, _, _ = calculate_confidence(items, "high", 1.0, [bad],  0)
    assert sg > sb


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Conflict detector
# ─────────────────────────────────────────────────────────────────────────────

def test_no_conflicts_clean_items():
    conflicts = detect_conflicts([_item("l1", text="Section 3(p) is relevant."),
                                  _item("l2", text="Traditional knowledge excluded.")])
    assert len(conflicts) == 0

def test_conflict_detected():
    conflicts = detect_conflicts([
        _item("s1", text="The invention is patentable under Indian law."),
        _item("s2", text="The invention is not patentable under Indian law."),
    ])
    assert len(conflicts) >= 1

def test_prefer_authoritative_sorts():
    sorted_items = prefer_authoritative([
        _item("sec", authority="secondary", score=0.9),
        _item("pri", authority="primary",   score=0.5),
    ])
    assert sorted_items[0].authority == "primary"

def test_conflict_deduplication():
    conflicts = detect_conflicts([
        _item("s1", text="invention is patentable"),
        _item("s2", text="invention is not patentable"),
    ])
    pairs = list({frozenset([c.source_a, c.source_b]) for c in conflicts})
    assert len(pairs) == len(conflicts)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Response serialisation
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_response_serialises():
    r = PipelineResponse(
        status="answered", answer="Section 3(p) is relevant.",
        language="en", confidence=0.85, confidence_band="high",
        citations=[CitationRecord(source_id="ev1", document="Patents Act",
                                  source_name="India Code", jurisdiction="india",
                                  authority="primary")],
    )
    d = r.model_dump()
    assert d["status"] == "answered"
    assert d["confidence"] == 0.85
    assert d["citations"][0]["verified"] is True

def test_abstained_has_no_answer():
    r = PipelineResponse(status="abstained", needs_human_review=True)
    assert r.answer is None
    assert r.needs_human_review is True

def test_out_of_scope():
    r = PipelineResponse(status="out_of_scope", needs_human_review=False)
    assert r.needs_human_review is False


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Version filter
# ─────────────────────────────────────────────────────────────────────────────

def test_historical_query_detection():
    assert is_historical_query("what was the original rule before 2021") is True
    assert is_historical_query("what does the current act say")          is False
    assert is_historical_query("old version of the Patents Act")         is True
    assert is_historical_query("Section 3(p) of the Patents Act")        is False

def test_filter_by_version_no_cutoff():
    result = filter_by_version([_item("ev1"), _item("ev2")], "What does Section 3(p) say?")
    assert len(result) == 2

def test_filter_by_version_with_cutoff():
    """Items published after cutoff_date must be excluded."""
    i_old = EvidenceItem(
        source_id="ev_old", source_type=EvidenceType.LEGAL, authority="primary",
        text="old text", title="Old Doc", source_name="Src",
        chunk_id="c1", similarity_score=0.8, publication_date="2015-06-01",
    )
    i_new = EvidenceItem(
        source_id="ev_new", source_type=EvidenceType.LEGAL, authority="primary",
        text="new text", title="New Doc", source_name="Src",
        chunk_id="c2", similarity_score=0.8, publication_date="2025-01-01",
    )
    result = filter_by_version([i_old, i_new], "prior art", cutoff_date="2020-01-01")
    ids = [r.source_id for r in result]
    assert "ev_old" in ids
    assert "ev_new" not in ids

def test_prefer_current_versions_sorts():
    """Higher authority + higher similarity → comes first."""
    i_weak   = EvidenceItem(
        source_id="weak", source_type=EvidenceType.LEGAL, authority="secondary",
        text="secondary text", title="Secondary", source_name="Src",
        chunk_id="c1", similarity_score=0.4,
    )
    i_strong = EvidenceItem(
        source_id="strong", source_type=EvidenceType.LEGAL, authority="primary",
        text="primary text", title="Primary", source_name="Src",
        chunk_id="c2", similarity_score=0.9,
    )
    result = prefer_current_versions([i_weak, i_strong])
    assert result[0].source_id == "strong"

def test_filter_empty_items():
    assert filter_by_version([], "any query") == []


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Answer formatter
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE = """\
### Preliminary Finding
Section 3(p) excludes traditional knowledge. [SOURCE_ID: PATENTS_ACT_1970_S3P]

### Why
The Patents Act says so. [SOURCE_ID: PATENTS_ACT_1970_S3P]

### Prior-Art Evidence
None retrieved.

### Traditional Knowledge
Found in TKDL. [SOURCE_ID: TKDL_001]

### Relevant Law
- Patents Act 1970, Section 3(p) [SOURCE_ID: PATENTS_ACT_1970_S3P]

### Confidence
High — primary source retrieved.

### Important
This is preliminary information only and does not constitute legal advice.
Always consult a qualified IP professional.
"""

def test_extract_sections_all_present():
    sections = extract_sections(_SAMPLE)
    for s in ["Preliminary Finding", "Why", "Relevant Law", "Important"]:
        assert s in sections

def test_has_disclaimer_true():
    assert has_disclaimer_section(_SAMPLE) is True

def test_has_disclaimer_false():
    assert has_disclaimer_section("### Preliminary Finding\nSomething.") is False

def test_ensure_disclaimer_adds():
    result = ensure_disclaimer_section("### Preliminary Finding\nSomething.")
    assert has_disclaimer_section(result) is True
    assert "legal advice" in result.lower()

def test_ensure_disclaimer_hindi():
    result = ensure_disclaimer_section("### Preliminary Finding\nSomething.", "hi")
    assert "कानूनी सलाह" in result

def test_ensure_disclaimer_not_duplicated():
    assert ensure_disclaimer_section(_SAMPLE).count("### Important") == 1

def test_structured_prompt_has_required_sections():
    for s in ["Preliminary Finding", "Why", "Relevant Law", "Confidence", "Important"]:
        assert s in STRUCTURED_ANSWER_SYSTEM


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Debug logger
# ─────────────────────────────────────────────────────────────────────────────

def test_debug_logger_disabled():
    log = PipelineDebugLogger(enabled=False)
    log.step("language", lang="en")
    assert len(log.get_steps()) == 0

def test_debug_logger_records_steps():
    import io, contextlib
    log = PipelineDebugLogger(enabled=True, request_id="TEST-001")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        log.step("language", lang="en", method="auto")
        log.step("scope",    scope="in_scope")
        log.step("tools",    tools=["legal_search"])
    assert len(log.get_steps()) == 3
    assert log.get_steps()[0]["step"] == "language"

def test_debug_logger_request_id():
    log = PipelineDebugLogger(enabled=True, request_id="REQ-XYZ")
    assert log.request_id == "REQ-XYZ"

def test_make_logger_default_disabled():
    log = make_logger()
    log.step("test", x=1)
    assert len(log.get_steps()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — 6 representative scenarios
# ─────────────────────────────────────────────────────────────────────────────

def test_scenarios_all_pass():
    import io, contextlib
    from pipeline.test_scenarios import (
        scenario_1_legal_query, scenario_2_herbal_formulation,
        scenario_3_pct_international, scenario_4_hindi_query,
        scenario_5_out_of_scope, scenario_6_abstention_trigger,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r1 = scenario_1_legal_query()
        r2 = scenario_2_herbal_formulation()
        r3 = scenario_3_pct_international()
        r4 = scenario_4_hindi_query()
        r5 = scenario_5_out_of_scope()
        r6 = scenario_6_abstention_trigger()
    assert r1 and r2 and r3 and r4 and r5 and r6


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_all() -> None:
    tests = [
        # ── Part 1 ────────────────────────────────────────────────────────
        (test_pipeline_response_defaults,          "Response: defaults"),
        (test_citation_record_verified_default,    "Response: citation verified=True"),
        (test_conflict_record_fields,              "Response: conflict record"),
        (test_build_citations_from_items,          "Citation: build from items"),
        (test_build_citations_deduplicates,        "Citation: deduplicates"),
        (test_build_sourced_context_has_source_id, "Citation: context has SOURCE_ID"),
        (test_extract_cited_ids_from_answer,       "Citation: extract IDs"),
        (test_verify_citations_all_found,          "Citation: all found → 100%"),
        (test_verify_citations_not_found,          "Citation: not found → unverified"),
        (test_verify_citations_no_ids,             "Citation: no IDs → build all"),
        (test_confidence_high_all_signals,         "Confidence: high all signals"),
        (test_confidence_low_empty,                "Confidence: low empty"),
        (test_confidence_reduced_by_conflicts,     "Confidence: reduced by conflicts"),
        (test_confidence_band_boundaries,          "Confidence: band boundaries"),
        (test_confidence_penalised_by_unverified,  "Confidence: unverified penalty"),
        (test_no_conflicts_clean_items,            "Conflict: clean → no conflicts"),
        (test_conflict_detected,                   "Conflict: contradiction detected"),
        (test_prefer_authoritative_sorts,          "Conflict: prefer_authoritative"),
        (test_conflict_deduplication,              "Conflict: deduplication"),
        (test_pipeline_response_serialises,        "Pipeline: serialises to dict"),
        (test_abstained_has_no_answer,             "Pipeline: abstained has no answer"),
        (test_out_of_scope,                        "Pipeline: out_of_scope"),
        # ── Part 2 ────────────────────────────────────────────────────────
        (test_historical_query_detection,          "Version: historical detection"),
        (test_filter_by_version_no_cutoff,         "Version: no cutoff → pass all"),
        (test_filter_by_version_with_cutoff,       "Version: cutoff excludes future"),
        (test_prefer_current_versions_sorts,       "Version: prefer current"),
        (test_filter_empty_items,                  "Version: empty list"),
        (test_extract_sections_all_present,        "Formatter: extract sections"),
        (test_has_disclaimer_true,                 "Formatter: disclaimer present"),
        (test_has_disclaimer_false,                "Formatter: disclaimer absent"),
        (test_ensure_disclaimer_adds,              "Formatter: adds disclaimer"),
        (test_ensure_disclaimer_hindi,             "Formatter: Hindi disclaimer"),
        (test_ensure_disclaimer_not_duplicated,    "Formatter: no duplicate"),
        (test_structured_prompt_has_required_sections, "Formatter: prompt has sections"),
        (test_debug_logger_disabled,               "Logger: disabled records nothing"),
        (test_debug_logger_records_steps,          "Logger: records steps"),
        (test_debug_logger_request_id,             "Logger: request ID"),
        (test_make_logger_default_disabled,        "Logger: default disabled"),
        (test_scenarios_all_pass,                  "Scenarios: all 6 pass"),
    ]

    print("=" * 65)
    print("IP-SAKTI  Phase 11 — Complete Pipeline Tests")
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
