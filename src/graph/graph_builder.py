"""
Graph Builder  (Phase 10)
──────────────────────────
Populates the KnowledgeGraph from retrieved evidence.

Populating from retrieved evidence (not the entire corpus) keeps the
graph small and query-relevant.

Example: after retrieving for a neem+turmeric formulation, the builder adds:
  neem  ──(appears_in)──►  patent_123
  neem  ──(documented_in)──►  tk_001
  tk_001 ──(relevant_to)──►  section_3p
  section_3p ──(belongs_to)──►  patents_act_1970
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from graph.graph_store import KnowledgeGraph
from evidence.store import EvidenceItem, EvidenceType


# ─────────────────────────────────────────────────────────────────────────────
# Section extractor
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r"(?:Section|Sec\.?)\s*(\d+[A-Z]?(?:\([a-z]+\))?)",
    re.IGNORECASE,
)


def _extract_sections(text: str) -> list[str]:
    return [f"section_{m.group(1).lower().replace('(','').replace(')','')}"
            for m in _SECTION_RE.finditer(text)]


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Incrementally populates a KnowledgeGraph from EvidenceItem objects.

    Call build_from_evidence() after each orchestration run to keep the
    graph up to date.
    """

    def __init__(self, graph: KnowledgeGraph):
        self._g = graph

    def build_from_evidence(
        self,
        items         : list[EvidenceItem],
        query_entities: list[str] | None = None,
    ) -> None:
        """
        Insert nodes and relationships for all evidence items.

        Parameters
        ----------
        items          : ranked evidence from the fusion layer
        query_entities : ingredient / component names from the user's query
                         (used to create formulation → ingredient edges)
        """
        for item in items:
            self._add_item(item)

        # Formulation → ingredient edges
        if query_entities:
            for entity in query_entities:
                entity_id = entity.lower().strip()
                if not self._g.has_node(entity_id):
                    self._g.add_node(entity_id, "ingredient")
                # Connect entity to evidence items that mention it
                for item in items:
                    if entity_id in item.text.lower() or entity_id in item.entities:
                        target_id = item.source_id.lower()
                        if item.source_type == EvidenceType.PATENT:
                            self._g.add_relationship(entity_id, "appears_in", target_id)
                        elif item.source_type == EvidenceType.TRADITIONAL_KNOWLEDGE:
                            self._g.add_relationship(entity_id, "documented_in", target_id)

    def _add_item(self, item: EvidenceItem) -> None:
        item_id = item.source_id.lower()

        # Add the item as a node
        node_type_map = {
            EvidenceType.LEGAL               : "legal_section",
            EvidenceType.PATENT              : "patent",
            EvidenceType.TRADITIONAL_KNOWLEDGE: "traditional_knowledge",
            EvidenceType.ABS                 : "legal_section",
            EvidenceType.INTERNATIONAL       : "legal_section",
        }
        node_type = node_type_map.get(item.source_type, "document")
        self._g.add_node(item_id, node_type, {
            "title"     : item.title,
            "source"    : item.source_name,
            "authority" : item.authority,
            "section"   : item.section or "",
        })

        # Legal items: connect section → act
        if item.source_type in (EvidenceType.LEGAL, EvidenceType.ABS):
            act_id = item.source_id.split("_")[0] if "_" in item.source_id else item.source_id
            if item.section:
                sec_id = f"section_{item.section.lower().replace(' ','_').replace('(','').replace(')','')}"
                self._g.add_node(sec_id, "legal_section")
                self._g.add_relationship(sec_id, "belongs_to", act_id)
                self._g.add_relationship(item_id, "relevant_to", sec_id)

            # Connect sections extracted from text
            for sec_id in _extract_sections(item.text):
                self._g.add_node(sec_id, "legal_section")
                self._g.add_relationship(item_id, "contains", sec_id)

        # TK items: connect to relevant legal sections
        if item.source_type == EvidenceType.TRADITIONAL_KNOWLEDGE:
            # Traditional knowledge is always potentially relevant to 3(p)
            self._g.add_node("section_3p", "legal_section", {
                "title": "Section 3(p) — traditional knowledge exclusion"
            })
            self._g.add_relationship(item_id, "relevant_to", "section_3p")

        # Patent items: connect to jurisdiction
        if item.source_type == EvidenceType.PATENT:
            jur_id = item.jurisdiction or "india"
            self._g.add_node(jur_id, "jurisdiction")
            self._g.add_relationship(item_id, "has_jurisdiction", jur_id)

    def get_graph(self) -> KnowledgeGraph:
        return self._g
