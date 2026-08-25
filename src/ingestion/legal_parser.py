"""
Legal structure-aware parser.

Recognises the hierarchy inside Indian legal documents:
  CHAPTER I / Chapter 1 / CHAPTER - I
    └── 1. Short title ...
         └── (a) ...
              └── (i) ...

Produces LegalChunk objects with chapter / section / subsection metadata.
Each subsection (or section when no subsections exist) becomes one chunk.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader

from .schema import LegalChunk


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "CHAPTER I", "CHAPTER - II", "Chapter III", "CHAPTER 4"
_CHAPTER_RE = re.compile(
    r"^(CHAPTER|Chapter)\s*[-–]?\s*([IVXLCDM]+|\d+)\b(.*)?$",
    re.IGNORECASE,
)

# "3. What are not inventions.—"  /  "10A. Some heading"
_SECTION_RE = re.compile(
    r"^(\d+[A-Z]?)\.\s+(.+?)[\.\—\-–]?\s*$",
    re.IGNORECASE,
)

# "(a)", "(ab)", "(1)", "(p)", "(iv)"  — subsection labels
_SUBSEC_RE = re.compile(
    r"^\(([a-zA-Z]{1,3}|[ivxlcdm]+|\d{1,3})\)\s*(.*)$",
    re.IGNORECASE,
)

# "Explanation" / "Provided that" / "Explanation.—"
_EXPLANATION_RE = re.compile(
    r"^(Explanation\s*[\.\—]?|Provided\s+that\s*[\.\—]?)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal data structure built during parsing
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    kind: str            # "chapter" | "section" | "subsection" | "explanation"
    label: str           # raw label, e.g. "Chapter II", "3", "(p)"
    heading: str         # heading text (may be empty)
    page: int
    lines: list[str] = field(default_factory=list)
    # structural context (filled in after parsing)
    chapter_label: Optional[str] = None
    section_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Public parser class
# ---------------------------------------------------------------------------

class LegalDocumentParser:
    """
    Parse a legal PDF into structured LegalChunk objects.

    Parameters
    ----------
    doc_meta : dict
        Required keys: document_id, title, source, document_type, domain,
                       authority_level
        Optional keys: source_url, version, effective_date, last_verified
    """

    def __init__(self, doc_meta: dict):
        self.meta = doc_meta
        self._chunk_counter: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def parse_pdf(self, pdf_path: str) -> list[LegalChunk]:
        """Load *pdf_path*, parse its text, return a list of LegalChunk."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()   # list of Document, one per page

        # Build a flat list of (page_number, line) pairs
        page_lines: list[tuple[int, str]] = []
        for doc in pages:
            page_num = doc.metadata.get("page", 0)
            if isinstance(page_num, int):
                page_num += 1   # PyPDF is 0-indexed
            for raw_line in doc.page_content.splitlines():
                line = self._clean(raw_line)
                if line:
                    page_lines.append((page_num, line))

        nodes = self._segment(page_lines)
        return self._build_chunks(nodes)

    def parse_text(
        self,
        text: str,
        page_override: int = 1,
    ) -> list[LegalChunk]:
        """Parse a plain string (useful for testing)."""
        page_lines = [(page_override, self._clean(ln))
                      for ln in text.splitlines()
                      if self._clean(ln)]
        nodes = self._segment(page_lines)
        return self._build_chunks(nodes)

    # ------------------------------------------------------------------
    # Step 1 — segmentation
    # ------------------------------------------------------------------

    def _segment(self, page_lines: list[tuple[int, str]]) -> list[_Node]:
        """Walk every line and emit _Node objects."""
        nodes: list[_Node] = []
        current_chapter = ""
        current_section = ""
        current_node: Optional[_Node] = None

        def _flush():
            if current_node is not None:
                nodes.append(current_node)

        for page, line in page_lines:

            # ── Chapter heading ──────────────────────────────────────────────
            m = _CHAPTER_RE.match(line)
            if m:
                _flush()
                current_chapter = f"Chapter {m.group(2).upper()}"
                heading = (m.group(3) or "").strip(" .—–-")
                current_node = _Node(
                    kind="chapter",
                    label=current_chapter,
                    heading=heading,
                    page=page,
                    chapter_label=current_chapter,
                )
                continue

            # ── Section heading ──────────────────────────────────────────────
            m = _SECTION_RE.match(line)
            if m:
                _flush()
                current_section = m.group(1)
                current_node = _Node(
                    kind="section",
                    label=current_section,
                    heading=m.group(2).strip(),
                    page=page,
                    chapter_label=current_chapter,
                    section_label=current_section,
                )
                # Add the heading text as first line
                current_node.lines.append(line)
                continue

            # ── Explanation / Proviso ────────────────────────────────────────
            if _EXPLANATION_RE.match(line):
                _flush()
                current_node = _Node(
                    kind="explanation",
                    label=line[:20],
                    heading="",
                    page=page,
                    chapter_label=current_chapter,
                    section_label=current_section,
                )
                current_node.lines.append(line)
                continue

            # ── Subsection label ─────────────────────────────────────────────
            m = _SUBSEC_RE.match(line)
            if m and current_section:
                _flush()
                sub_label = m.group(1)
                current_node = _Node(
                    kind="subsection",
                    label=sub_label,
                    heading="",
                    page=page,
                    chapter_label=current_chapter,
                    section_label=current_section,
                )
                current_node.lines.append(line)
                continue

            # ── Continuation line ────────────────────────────────────────────
            if current_node is not None:
                current_node.lines.append(line)

        _flush()
        return nodes

    # ------------------------------------------------------------------
    # Step 2 — build LegalChunk objects
    # ------------------------------------------------------------------

    def _build_chunks(self, nodes: list[_Node]) -> list[LegalChunk]:
        chunks: list[LegalChunk] = []

        for node in nodes:
            text = node.heading + "\n" + "\n".join(node.lines)
            text = text.strip()
            if not text:
                continue

            # Build section / subsection strings
            section_str = (
                f"Section {node.section_label}" if node.section_label else None
            )
            subsection_str: Optional[str] = None
            if node.kind == "subsection":
                base = node.section_label or ""
                subsection_str = f"{base}({node.label})" if base else f"({node.label})"
            elif node.kind == "explanation":
                # Attach explanation to the parent section
                subsection_str = None

            chapter_str = node.chapter_label or None

            chunk_id = self._make_id(
                node.chapter_label,
                node.section_label,
                node.label if node.kind == "subsection" else None,
                node.page,
            )

            chunk = LegalChunk(
                chunk_id=chunk_id,
                document_id=self.meta["document_id"],
                title=self.meta["title"],
                source=self.meta["source"],
                source_url=self.meta.get("source_url"),
                document_type=self.meta["document_type"],
                domain=self.meta["domain"],
                authority_level=self.meta["authority_level"],
                chapter=chapter_str,
                section=section_str,
                subsection=subsection_str,
                page=node.page,
                language=self.meta.get("language", "english"),
                version=self.meta.get("version"),
                effective_date=self.meta.get("effective_date"),
                last_verified=self.meta.get("last_verified", "2026-08-25"),
                text=text,
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_id(
        self,
        chapter: Optional[str],
        section: Optional[str],
        subsection: Optional[str],
        page: int,
    ) -> str:
        doc_id = self.meta["document_id"]
        parts = [doc_id]
        if chapter:
            parts.append(re.sub(r"\s+", "_", chapter).lower())
        if section:
            parts.append(f"sec{section}")
        if subsection:
            # "(p)" → "p"
            clean = re.sub(r"[^a-zA-Z0-9]", "", subsection)
            parts.append(clean)
        parts.append(f"page{page}")
        base = "_".join(parts)

        # Guarantee uniqueness within this parse run
        count = self._chunk_counter.get(base, 0)
        self._chunk_counter[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise unicode, strip leading/trailing whitespace."""
        text = unicodedata.normalize("NFKC", text)
        return text.strip()
