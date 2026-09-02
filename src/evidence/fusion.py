"""
Evidence Fusion  (Phase 10)
─────────────────────────────
Merges evidence from all tools (legal, patent, TK, ABS, international)
into one structured package that the orchestrator passes to the LLM.

This is the Phase 10 upgrade of Phase 3/4's evidence_fusion.py —
it works with EvidenceItem objects instead of raw chunk tuples, and
it incorporates knowledge-graph relationships.
"""

from __future__ import annotations

from typing import Optional

from evidence.store import EvidenceItem, EvidenceType
from evidence.ranking import rank_evidence, rank_by_type


# ─────────────────────────────────────────────────────────────────────────────
# Fused evidence package
# ─────────────────────────────────────────────────────────────────────────────

class FusedEvidence:
    """
    Container for all evidence collected by the orchestrator.

    Attributes
    ----------
    legal           : ranked legal provisions
    prior_art       : ranked patent / prior-art records
    traditional_knowledge : ranked TK / AYUSH sources
    abs             : ABS-relevant legal/regulatory items
    international   : international treaty / WIPO material
    graph_relations : relationships extracted from the knowledge graph
    tool_status     : which tools succeeded / failed / were not called
    issues          : automatically detected issues (e.g. Section 3(p))
    overall_confidence : "high" | "medium" | "low"
    """

    def __init__(self):
        self.legal                : list[EvidenceItem] = []
        self.prior_art            : list[EvidenceItem] = []
        self.traditional_knowledge: list[EvidenceItem] = []
        self.abs                  : list[EvidenceItem] = []
        self.international        : list[EvidenceItem] = []
        self.graph_relations      : list[dict]         = []
        self.tool_status          : dict[str, str]     = {}   # tool → "success"|"failed"|"not_called"
        self.issues               : list[str]          = []
        self.overall_confidence   : str                = "low"

    def all_items(self) -> list[EvidenceItem]:
        return (
            self.legal + self.prior_art +
            self.traditional_knowledge + self.abs + self.international
        )

    def to_dict(self) -> dict:
        return {
            "legal"             : [i.to_dict() for i in self.legal],
            "prior_art"         : [i.to_dict() for i in self.prior_art],
            "traditional_knowledge": [i.to_dict() for i in self.traditional_knowledge],
            "abs"               : [i.to_dict() for i in self.abs],
            "international"     : [i.to_dict() for i in self.international],
            "graph_relations"   : self.graph_relations,
            "tool_status"       : self.tool_status,
            "issues"            : self.issues,
            "overall_confidence": self.overall_confidence,
            "total_sources"     : len(self.all_items()),
        }

    def context_blocks(self) -> list[str]:
        """Build LLM-ready context blocks, separated by type."""
        blocks: list[str] = []

        def _add_section(title: str, items: list[EvidenceItem]) -> None:
            if not items:
                return
            blocks.append(f"=== {title} ===")
            for i, item in enumerate(items[:5], 1):
                header = f"[{i}] {item.title or item.source_id}"
                if item.section:
                    header += f" | {item.section}"
                if item.subsection:
                    header += f" § {item.subsection}"
                if item.page:
                    header += f" | p.{item.page}"
                header += f" | {item.source_name} ({item.authority})"
                blocks.append(header)
                blocks.append(item.text[:500])
                blocks.append("")

        _add_section("LEGAL PROVISIONS", self.legal)
        _add_section("PRIOR ART — PATENTS", self.prior_art)
        _add_section("TRADITIONAL KNOWLEDGE", self.traditional_knowledge)
        _add_section("ABS / BIODIVERSITY", self.abs)
        _add_section("INTERNATIONAL SOURCES", self.international)

        return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Issue detection
# ─────────────────────────────────────────────────────────────────────────────

_ISSUE_MAP: list[tuple[str, str]] = [
    ("3(p)",  "Section 3(p) — traditional knowledge exclusion may apply."),
    ("3(d)",  "Section 3(d) — new form of known substance without enhanced efficacy."),
    ("3(e)",  "Section 3(e) — mere mixture of known ingredients may apply."),
    ("3(i)",  "Section 3(i) — method of treatment exclusion may apply."),
    ("3(j)",  "Section 3(j) — plant/animal/biological process exclusion may apply."),
    ("3(k)",  "Section 3(k) — mathematical method / software / business method exclusion."),
]


def _detect_issues(legal: list[EvidenceItem], tk: list[EvidenceItem]) -> list[str]:
    issues: list[str] = []
    seen: set[str] = set()
    for item in legal:
        sec = (item.subsection or item.section or "").lower()
        for trigger, msg in _ISSUE_MAP:
            if trigger.lower() in sec and msg not in seen:
                issues.append(msg)
                seen.add(msg)
    if tk:
        msg = "Traditional-knowledge material retrieved. Section 3(p) relevance should be assessed."
        if msg not in seen:
            issues.append(msg)
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confidence(fused: FusedEvidence) -> str:
    has_primary = any(
        item.authority in ("primary", "registry")
        for item in fused.all_items()
    )
    total = len(fused.all_items())
    if has_primary and total >= 3:
        return "high"
    if has_primary or total >= 2:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Main fusion function
# ─────────────────────────────────────────────────────────────────────────────

def fuse(
    raw_items   : list[EvidenceItem],
    tool_status : Optional[dict[str, str]] = None,
    graph_relations: Optional[list[dict]]  = None,
) -> FusedEvidence:
    """
    Rank all items, separate by type, detect issues, compute confidence.

    Parameters
    ----------
    raw_items      : all EvidenceItem objects collected by all tools
    tool_status    : {"legal_search": "success", "abs_check": "failed", …}
    graph_relations: relationships from KnowledgeGraph
    """
    fused = FusedEvidence()
    fused.tool_status    = tool_status or {}
    fused.graph_relations= graph_relations or []

    # Rank all items globally first, then split by type
    ranked = rank_evidence(raw_items, top_k=30)
    by_type = rank_by_type(ranked)

    fused.legal               = by_type.get(EvidenceType.LEGAL,                [])
    fused.prior_art           = by_type.get(EvidenceType.PATENT,               [])
    fused.traditional_knowledge = by_type.get(EvidenceType.TRADITIONAL_KNOWLEDGE, [])
    fused.abs                 = by_type.get(EvidenceType.ABS,                  [])
    fused.international       = by_type.get(EvidenceType.INTERNATIONAL,        [])

    fused.issues             = _detect_issues(fused.legal, fused.traditional_knowledge)
    fused.overall_confidence = _compute_confidence(fused)

    return fused
