"""
Knowledge Graph Store  (Phase 10)
────────────────────────────────────
In-memory knowledge graph.

No Neo4j required — a plain Python dict-based graph is enough for the
prototype. The structure can be persisted to data/graph/knowledge.json
and loaded on startup.

Node types  : ingredient | formulation | patent | traditional_knowledge
              | legal_section | act | jurisdiction | organization
Relation types: contains | appears_in | documented_in | relevant_to
                | belongs_to | has_jurisdiction | similar_to | potentially_related_to
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "graph" / "knowledge.json"


class KnowledgeGraph:
    """
    In-memory graph: nodes (id → {type, properties}) + relationships list.

    Supports:
      add_node / add_relationship
      find_related(node_id, relation) → list of target node IDs
      neighbours(node_id) → all related node IDs
      save / load (JSON)
    """

    def __init__(self):
        self.nodes: dict[str, dict]  = {}
        self.relationships: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id   : str,
        node_type : str,
        properties: Optional[dict] = None,
    ) -> None:
        """Add or update a node."""
        key = node_id.lower().strip()
        if key not in self.nodes:
            self.nodes[key] = {"type": node_type, "properties": properties or {}}
        else:
            # Merge properties if node already exists
            self.nodes[key]["properties"].update(properties or {})

    def add_relationship(
        self,
        source  : str,
        relation: str,
        target  : str,
        properties: Optional[dict] = None,
    ) -> None:
        """Add a directed relationship. Ensures nodes exist first."""
        src = source.lower().strip()
        tgt = target.lower().strip()
        # Don't add duplicate edges
        for rel in self.relationships:
            if rel["source"] == src and rel["relation"] == relation and rel["target"] == tgt:
                return
        self.relationships.append({
            "source"    : src,
            "relation"  : relation,
            "target"    : tgt,
            "properties": properties or {},
        })

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def find_related(
        self,
        node_id : str,
        relation: Optional[str] = None,
    ) -> list[str]:
        """
        Return target node IDs reachable from *node_id* via *relation*.
        If relation is None, return all targets.
        """
        src = node_id.lower().strip()
        results = []
        for rel in self.relationships:
            if rel["source"] == src:
                if relation is None or rel["relation"] == relation:
                    results.append(rel["target"])
        return results

    def find_sources(
        self,
        node_id : str,
        relation: Optional[str] = None,
    ) -> list[str]:
        """Reverse lookup: find nodes that point TO node_id."""
        tgt = node_id.lower().strip()
        results = []
        for rel in self.relationships:
            if rel["target"] == tgt:
                if relation is None or rel["relation"] == relation:
                    results.append(rel["source"])
        return results

    def neighbours(self, node_id: str) -> list[str]:
        """All nodes directly connected (either direction)."""
        return list(set(
            self.find_related(node_id) + self.find_sources(node_id)
        ))

    def get_node(self, node_id: str) -> Optional[dict]:
        return self.nodes.get(node_id.lower().strip())

    def has_node(self, node_id: str) -> bool:
        return node_id.lower().strip() in self.nodes

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": nid, "type": data["type"], "properties": data["properties"]}
                for nid, data in self.nodes.items()
            ],
            "relationships": self.relationships,
        }

    def save(self, path: Optional[Path] = None) -> None:
        p = path or _DEFAULT_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "KnowledgeGraph":
        p = path or _DEFAULT_PATH
        g = cls()
        if not p.exists():
            return g
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        for node in data.get("nodes", []):
            g.add_node(node["id"], node["type"], node.get("properties", {}))
        for rel in data.get("relationships", []):
            g.add_relationship(
                rel["source"], rel["relation"], rel["target"],
                rel.get("properties", {})
            )
        return g

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph("
            f"{len(self.nodes)} nodes, {len(self.relationships)} relationships)"
        )
