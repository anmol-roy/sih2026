"""
Phase 10 — Orchestration tests  (no LLM / no network)

Tests cover:
  - Tool selection heuristics (deterministic)
  - EvidenceItem normalisation
  - Evidence ranking + deduplication
  - Knowledge graph store + builder
  - Evidence fusion

Run from project root:
    python tests/test_orchestration.py
    python -m pytest tests/test_orchestration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.orchestrator import select_tools
from evidence.store import (
    EvidenceItem, EvidenceType, Authority,
    from_legal_chunk, from_patent_chunk, from_tk_match,
)
from evidence.ranking import rank_evidence, combined_score, deduplicate
from evidence.fusion  import fuse
from graph.graph_store import KnowledgeGraph
from graph.graph_builder import GraphBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_item(source_id="ev1", source_type=EvidenceType.LEGAL,
               authority="primary", score=0.8, section=None) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id, source_type=source_type,
        authority=authority, similarity_score=score,
        text="Test evidence text.", section=section,
        title=f"Test {source_id}", source_name="Test Source",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tool selection tests
# ─────────────────────────────────────────────────────────────────────────────

def test_select_simple_legal():
    tools = [t.split(":")[0] for t in select_tools("What does Section 3(p) say?")]
    assert "legal_search" in tools
    assert "formulation_classifier" not in tools
    assert "international_search" not in tools

def test_select_trademark():
    tools = [t.split(":")[0] for t in select_tools("How do I register a trademark in India?")]
    assert "legal_search" in tools
    # Should use trademark domain
    raw = select_tools("How do I register a trademark in India?")
    assert any("trademark" in t for t in raw)

def test_select_formulation_triggers_multiple():
    tools = [t.split(":")[0] for t in select_tools(
        "Can I patent my herbal formulation with neem and turmeric?"
    )]
    assert "formulation_classifier" in tools
    assert "tk_search" in tools
    assert "legal_search" in tools
    assert "patent_search" in tools

def test_select_abs():
    tools = [t.split(":")[0] for t in select_tools(
        "Does using ashwagandha create ABS compliance issues?"
    )]
    assert "abs_check" in tools
    assert "formulation_classifier" in tools

def test_select_international_pct():
    tools = [t.split(":")[0] for t in select_tools("What is the PCT?")]
    assert "international_search" in tools
    assert "formulation_classifier" not in tools

def test_select_both_jurisdictions():
    tools = [t.split(":")[0] for t in select_tools(
        "How does Indian patent law compare with PCT and TRIPS?"
    )]
    assert "legal_search" in tools
    assert "international_search" in tools

def test_select_tk_without_formulation():
    tools = [t.split(":")[0] for t in select_tools(
        "What does TKDL say about traditional knowledge prior art?"
    )]
    assert "tk_search" in tools

def test_select_no_duplicates():
    tools = [t.split(":")[0] for t in select_tools(
        "Neem turmeric herbal formulation patent TK ABS biodiversity"
    )]
    assert len(tools) == len(set(tools))

def test_select_complex_query():
    tools = [t.split(":")[0] for t in select_tools(
        "I developed an Ayurvedic formulation using neem and turmeric. "
        "Can I patent it in India, and could traditional knowledge affect my application?"
    )]
    assert "formulation_classifier" in tools
    assert "legal_search" in tools
    assert "patent_search" in tools
    assert "tk_search" in tools


# ─────────────────────────────────────────────────────────────────────────────
# 2. EvidenceItem tests
# ─────────────────────────────────────────────────────────────────────────────

def test_evidence_item_authority_score():
    item = _make_item(authority="primary")
    assert item.authority_score() == 1.0

def test_evidence_item_authority_score_secondary():
    item = _make_item(authority="secondary")
    assert item.authority_score() == 0.70

def test_evidence_item_to_dict_keys():
    item = _make_item()
    d = item.to_dict()
    assert "source_id" in d
    assert "source_type" in d
    assert "authority" in d
    assert "text" in d

def test_evidence_item_text_truncated():
    long_text = "x" * 1000
    item = EvidenceItem(source_id="x", source_type=EvidenceType.LEGAL,
                        text=long_text, authority="primary", source_name="T", title="T")
    d = item.to_dict()
    assert len(d["text"]) <= 404   # 400 + "…"


class _FakeChunk:
    chunk_id = "chunk1"
    document_id = "patents_act_1970"
    title = "The Patents Act, 1970"
    source = "India Code"
    source_url = None
    document_type = "act"
    domain = "patent"
    authority_level = "primary"
    chapter = "Chapter II"
    section = "Section 3"
    subsection = "3(p)"
    page = 8
    language = "english"
    version = "current"
    effective_date = None
    last_verified = "2026-08-25"
    jurisdiction = "india"
    text = "Section 3(p) excludes traditional knowledge."


def test_from_legal_chunk():
    item = from_legal_chunk(_FakeChunk(), score=0.85)
    assert item.source_type == EvidenceType.LEGAL
    assert item.authority == "primary"
    assert item.section == "Section 3"
    assert item.similarity_score == 0.85

def test_from_patent_chunk():
    class FakePatent:
        chunk_id = "pat1"
        publication_number = "IN123456"
        title = "Herbal formulation"
        source = "IP India"
        text = "Claims..."
        page = 3
        filing_date = "2020-01-01"
        publication_date = "2021-06-15"
    item = from_patent_chunk(FakePatent(), score=0.75)
    assert item.source_type == EvidenceType.PATENT
    assert item.authority == Authority.REGISTRY

def test_from_tk_match():
    match = {
        "chunk": _FakeChunk(),
        "score": 0.90,
        "matched_components": ["neem"],
        "page": 23,
    }
    item = from_tk_match(match)
    assert item.source_type == EvidenceType.TRADITIONAL_KNOWLEDGE
    assert item.similarity_score == 0.90
    assert "neem" in item.entities


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ranking tests
# ─────────────────────────────────────────────────────────────────────────────

def test_combined_score_primary_high():
    item = _make_item(authority="primary", score=0.9)
    s = combined_score(item)
    assert s >= 0.95   # 1.0*0.6 + 0.9*0.4 = 0.96

def test_combined_score_secondary_low():
    item = _make_item(authority="secondary", score=0.3)
    s = combined_score(item)
    assert s <= 0.55

def test_rank_evidence_sorted():
    items = [
        _make_item("ev1", authority="secondary", score=0.3),
        _make_item("ev2", authority="primary",   score=0.9),
        _make_item("ev3", authority="registry",  score=0.7),
    ]
    ranked = rank_evidence(items)
    assert ranked[0].source_id == "ev2"   # primary+high sim should be first

def test_rank_evidence_top_k():
    items = [_make_item(f"ev{i}", authority="secondary", score=0.5) for i in range(20)]
    ranked = rank_evidence(items, top_k=5)
    assert len(ranked) <= 5

def test_deduplicate_keeps_best():
    items = [
        _make_item("ev1", authority="secondary", score=0.3),
        _make_item("ev1", authority="primary",   score=0.9),   # same ID, higher score
    ]
    deduped = deduplicate(items)
    assert len(deduped) == 1
    assert deduped[0].authority == "primary"

def test_rank_min_score_filter():
    items = [
        _make_item("ev1", authority="general", score=0.1),
        _make_item("ev2", authority="primary", score=0.9),
    ]
    ranked = rank_evidence(items, min_score=0.5)
    assert all(combined_score(i) >= 0.5 for i in ranked)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evidence fusion tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fuse_separates_by_type():
    items = [
        _make_item("l1", EvidenceType.LEGAL,                "primary", 0.9, "3(p)"),
        _make_item("p1", EvidenceType.PATENT,               "registry", 0.7),
        _make_item("t1", EvidenceType.TRADITIONAL_KNOWLEDGE,"official", 0.8),
    ]
    fused = fuse(items)
    assert len(fused.legal) >= 1
    assert len(fused.prior_art) >= 1
    assert len(fused.traditional_knowledge) >= 1

def test_fuse_detects_3p_issue():
    item = EvidenceItem(
        source_id="l1", source_type=EvidenceType.LEGAL,
        authority="primary", similarity_score=0.9,
        text="Section 3(p) traditional knowledge exclusion.",
        section="Section 3", subsection="3(p)",
        title="Patents Act", source_name="India Code",
    )
    fused = fuse([item])
    assert any("3(p)" in iss for iss in fused.issues)

def test_fuse_tk_triggers_issue():
    items = [_make_item("t1", EvidenceType.TRADITIONAL_KNOWLEDGE, "official", 0.8)]
    fused = fuse(items)
    assert len(fused.issues) > 0

def test_fuse_confidence_high_with_primary():
    items = [
        _make_item("l1", EvidenceType.LEGAL, "primary", 0.9),
        _make_item("l2", EvidenceType.LEGAL, "primary", 0.85),
        _make_item("p1", EvidenceType.PATENT, "registry", 0.7),
    ]
    fused = fuse(items)
    assert fused.overall_confidence in ("high", "medium")

def test_fuse_confidence_low_empty():
    fused = fuse([])
    assert fused.overall_confidence == "low"

def test_fuse_tool_status_preserved():
    fused = fuse([], tool_status={"legal_search": "success", "abs_check": "failed"})
    assert fused.tool_status["abs_check"] == "failed"

def test_fuse_context_blocks_non_empty():
    items = [_make_item("l1", EvidenceType.LEGAL, "primary", 0.9)]
    fused = fuse(items)
    blocks = fused.context_blocks()
    assert len(blocks) > 0
    assert any("LEGAL" in b for b in blocks)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Knowledge graph tests
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_add_node():
    g = KnowledgeGraph()
    g.add_node("neem", "ingredient", {"scientific_name": "Azadirachta indica"})
    assert g.has_node("neem")
    assert g.get_node("neem")["type"] == "ingredient"

def test_graph_add_relationship():
    g = KnowledgeGraph()
    g.add_node("neem", "ingredient")
    g.add_node("tk_001", "traditional_knowledge")
    g.add_relationship("neem", "documented_in", "tk_001")
    related = g.find_related("neem", "documented_in")
    assert "tk_001" in related

def test_graph_no_duplicate_edges():
    g = KnowledgeGraph()
    g.add_node("a", "ingredient")
    g.add_node("b", "patent")
    g.add_relationship("a", "appears_in", "b")
    g.add_relationship("a", "appears_in", "b")
    assert len(g.relationships) == 1

def test_graph_find_sources():
    g = KnowledgeGraph()
    g.add_node("sec_3p", "legal_section")
    g.add_node("tk_001", "traditional_knowledge")
    g.add_relationship("tk_001", "relevant_to", "sec_3p")
    sources = g.find_sources("sec_3p", "relevant_to")
    assert "tk_001" in sources

def test_graph_neighbours():
    g = KnowledgeGraph()
    g.add_node("neem", "ingredient")
    g.add_node("tk_1", "traditional_knowledge")
    g.add_node("pat_1", "patent")
    g.add_relationship("neem", "documented_in", "tk_1")
    g.add_relationship("neem", "appears_in",    "pat_1")
    n = g.neighbours("neem")
    assert "tk_1" in n
    assert "pat_1" in n

def test_graph_save_load(tmp_path):
    g = KnowledgeGraph()
    g.add_node("neem", "ingredient")
    g.add_node("tk_1", "traditional_knowledge")
    g.add_relationship("neem", "documented_in", "tk_1")
    path = tmp_path / "test_graph.json"
    g.save(path)
    g2 = KnowledgeGraph.load(path)
    assert g2.has_node("neem")
    assert "tk_1" in g2.find_related("neem", "documented_in")

def test_graph_repr():
    g = KnowledgeGraph()
    g.add_node("n1", "ingredient")
    g.add_node("n2", "patent")
    g.add_relationship("n1", "appears_in", "n2")
    r = repr(g)
    assert "2 nodes" in r
    assert "1 relationships" in r


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph builder tests
# ─────────────────────────────────────────────────────────────────────────────

def test_builder_populates_legal_item():
    g = KnowledgeGraph()
    b = GraphBuilder(g)
    item = _make_item("patents_act_1970_sec3p", EvidenceType.LEGAL, "primary", 0.9)
    item.section = "Section 3"
    item.subsection = "3(p)"
    b.build_from_evidence([item])
    assert g.has_node("patents_act_1970_sec3p")

def test_builder_tk_links_to_section_3p():
    g = KnowledgeGraph()
    b = GraphBuilder(g)
    item = _make_item("tk_neem", EvidenceType.TRADITIONAL_KNOWLEDGE, "official", 0.8)
    b.build_from_evidence([item])
    assert g.has_node("section_3p")
    related = g.find_sources("section_3p", "relevant_to")
    assert "tk_neem" in related

def test_builder_ingredient_to_patent():
    g = KnowledgeGraph()
    b = GraphBuilder(g)
    item = _make_item("pat_123", EvidenceType.PATENT, "registry", 0.7)
    item.text = "formulation containing neem extract"
    b.build_from_evidence([item], query_entities=["neem"])
    assert "pat_123" in g.find_related("neem", "appears_in")

def test_builder_ingredient_to_tk():
    g = KnowledgeGraph()
    b = GraphBuilder(g)
    item = _make_item("tk_001", EvidenceType.TRADITIONAL_KNOWLEDGE, "official", 0.8)
    item.text = "turmeric traditionally used for skin conditions"
    b.build_from_evidence([item], query_entities=["turmeric"])
    assert "tk_001" in g.find_related("turmeric", "documented_in")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Seed graph load test
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_graph_loads():
    g = KnowledgeGraph.load()
    assert g.has_node("neem")
    assert g.has_node("turmeric")
    assert g.has_node("section_3p")
    assert "tkdl_overview" in g.find_related("neem", "documented_in")

def test_seed_graph_section_3p_belongs_to_act():
    g = KnowledgeGraph.load()
    assert "patents_act_1970" in g.find_related("section_3p", "belongs_to")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Tool selection 20-case eval (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def test_all_cases_tool_selection():
    """Run all 20 evaluation cases through select_tools()."""
    import json
    cases_path = Path(__file__).parent.parent / "evaluation" / "orchestration" / "cases.json"
    with open(cases_path, encoding="utf-8") as fh:
        cases = json.load(fh)

    failures = []
    for case in cases:
        selected = [t.split(":")[0] for t in select_tools(case["query"])]
        missing  = [t for t in case.get("expected_tools", []) if t not in selected]
        if missing:
            failures.append(f"{case['id']}: missing {missing}")

    assert not failures, "\n".join(failures)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_all() -> None:
    tests = [
        (test_select_simple_legal,               "Select: simple legal"),
        (test_select_trademark,                  "Select: trademark domain"),
        (test_select_formulation_triggers_multiple,"Select: formulation → 4 tools"),
        (test_select_abs,                        "Select: ABS"),
        (test_select_international_pct,          "Select: PCT → international"),
        (test_select_both_jurisdictions,         "Select: both jurisdictions"),
        (test_select_tk_without_formulation,     "Select: TK without formulation"),
        (test_select_no_duplicates,              "Select: no duplicates"),
        (test_select_complex_query,              "Select: complex query"),
        (test_evidence_item_authority_score,     "Evidence: primary score 1.0"),
        (test_evidence_item_authority_score_secondary, "Evidence: secondary score"),
        (test_evidence_item_to_dict_keys,        "Evidence: to_dict keys"),
        (test_evidence_item_text_truncated,      "Evidence: text truncated"),
        (test_from_legal_chunk,                  "Evidence: from_legal_chunk"),
        (test_from_patent_chunk,                 "Evidence: from_patent_chunk"),
        (test_from_tk_match,                     "Evidence: from_tk_match"),
        (test_combined_score_primary_high,       "Ranking: primary high score"),
        (test_combined_score_secondary_low,      "Ranking: secondary low score"),
        (test_rank_evidence_sorted,              "Ranking: sorted by score"),
        (test_rank_evidence_top_k,               "Ranking: top_k respected"),
        (test_deduplicate_keeps_best,            "Ranking: dedup keeps best"),
        (test_rank_min_score_filter,             "Ranking: min_score filter"),
        (test_fuse_separates_by_type,            "Fusion: separates by type"),
        (test_fuse_detects_3p_issue,             "Fusion: detects 3(p) issue"),
        (test_fuse_tk_triggers_issue,            "Fusion: TK triggers issue"),
        (test_fuse_confidence_high_with_primary, "Fusion: confidence high"),
        (test_fuse_confidence_low_empty,         "Fusion: confidence low empty"),
        (test_fuse_tool_status_preserved,        "Fusion: tool status preserved"),
        (test_fuse_context_blocks_non_empty,     "Fusion: context blocks non-empty"),
        (test_graph_add_node,                    "Graph: add node"),
        (test_graph_add_relationship,            "Graph: add relationship"),
        (test_graph_no_duplicate_edges,          "Graph: no duplicate edges"),
        (test_graph_find_sources,                "Graph: find_sources"),
        (test_graph_neighbours,                  "Graph: neighbours"),
        (test_graph_repr,                        "Graph: repr"),
        (test_builder_populates_legal_item,      "Builder: legal item"),
        (test_builder_tk_links_to_section_3p,    "Builder: TK → section_3p"),
        (test_builder_ingredient_to_patent,      "Builder: ingredient → patent"),
        (test_builder_ingredient_to_tk,          "Builder: ingredient → TK"),
        (test_seed_graph_loads,                  "Seed graph: loads"),
        (test_seed_graph_section_3p_belongs_to_act, "Seed graph: 3p → act"),
        (test_all_cases_tool_selection,          "All 20 cases: tool selection"),
    ]

    # graph save/load needs tmp_path — skip in manual runner
    skip_need_tmpdir = {test_graph_save_load}

    print("=" * 65)
    print("IP-SAKTI  Phase 10 — Orchestration Tests")
    print("=" * 65)

    passed = failed = 0
    for fn, label in tests:
        if fn in skip_need_tmpdir:
            print(f"  ~  {label}  (skipped — needs tmp_path)")
            continue
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
        print("\n  ✓ All Phase 10 tests passed.")
    else:
        print(f"\n  ⚠  {failed} test(s) failed.")


if __name__ == "__main__":
    _run_all()
