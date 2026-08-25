"""
Patent document parser.

Recognises the major sections of an Indian patent document:
  Abstract / Background / Summary / Description / Claims / Classification

Each section becomes one or more PatentChunk objects.
Claims are split individually (Claim 1, Claim 2, …) because they carry
the most weight in a prior-art assessment.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader

from .schema import PatentChunk


# ---------------------------------------------------------------------------
# Section boundary patterns
# ---------------------------------------------------------------------------

_SECTION_MARKERS: list[tuple[str, re.Pattern]] = [
    ("abstract",       re.compile(r"^(ABSTRACT|Abstract)\s*$")),
    ("background",     re.compile(r"^(BACKGROUND|Background\s+of\s+(the\s+)?[Ii]nvention|FIELD\s+OF\s+THE\s+INVENTION)\s*$")),
    ("summary",        re.compile(r"^(SUMMARY|Summary\s+of\s+(the\s+)?[Ii]nvention|BRIEF\s+SUMMARY)\s*$")),
    ("description",    re.compile(r"^(DETAILED\s+DESCRIPTION|Description|DESCRIPTION\s+OF\s+(EMBODIMENTS|THE\s+INVENTION))\s*$")),
    ("claims",         re.compile(r"^(CLAIMS|Claims)\s*$")),
    ("classification", re.compile(r"^(CLASSIFICATION|IPC\s+Classification|CPC\s+Classification)\s*$")),
]

# Individual claim: "1." or "1. A composition …"
_CLAIM_START_RE = re.compile(r"^(\d+)\.\s+(.*)")

# Bibliographic fields that often appear at the top of a patent PDF
_BIB_RE: dict[str, re.Pattern] = {
    "publication_number": re.compile(r"(Publication\s+No\.?|Patent\s+No\.?)\s*[:\-]?\s*(\S+)", re.I),
    "application_number": re.compile(r"Application\s+No\.?\s*[:\-]?\s*(\S+)", re.I),
    "filing_date":        re.compile(r"(Filing|Filed)\s+Date\s*[:\-]?\s*(\S+)", re.I),
    "publication_date":   re.compile(r"Publication\s+Date\s*[:\-]?\s*(\S+)", re.I),
    "applicant":          re.compile(r"Applicant\s*[:\-]?\s*(.+)", re.I),
    "inventor":           re.compile(r"Inventor\s*[:\-]?\s*(.+)", re.I),
    "classification":     re.compile(r"(IPC|CPC)\s*[:\-]?\s*([A-Z]\d{2}[A-Z]?\s*\d+/\d+[\d/]*)", re.I),
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class PatentDocumentParser:
    """
    Parse a patent PDF into structured PatentChunk objects.

    Parameters
    ----------
    doc_meta : dict
        Required keys  : publication_number, title
        Optional keys  : application_number, applicant, inventor,
                         filing_date, publication_date, classification,
                         status, source, source_url
    max_description_chars : int
        Split long description sections into chunks of this size.
    """

    def __init__(
        self,
        doc_meta: dict,
        max_description_chars: int = 1500,
    ):
        self.meta = doc_meta
        self.max_desc = max_description_chars
        self._counter: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def parse_pdf(self, pdf_path: str) -> list[PatentChunk]:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        page_lines: list[tuple[int, str]] = []
        for doc in pages:
            pnum = doc.metadata.get("page", 0)
            if isinstance(pnum, int):
                pnum += 1
            for raw in doc.page_content.splitlines():
                line = _clean(raw)
                if line:
                    page_lines.append((pnum, line))

        # Try to extract bibliographic info from first 60 lines
        bib = dict(self.meta)
        self._extract_bib(page_lines[:60], bib)

        sections = self._segment(page_lines)
        return self._build_chunks(sections, bib)

    # ------------------------------------------------------------------
    # Step 1 – segmentation
    # ------------------------------------------------------------------

    def _extract_bib(self, lines: list[tuple[int, str]], bib: dict) -> None:
        for _, line in lines:
            for field, pattern in _BIB_RE.items():
                if field in bib and bib[field]:
                    continue
                m = pattern.search(line)
                if m:
                    bib[field] = m.group(m.lastindex or 1).strip()

    def _segment(
        self,
        page_lines: list[tuple[int, str]],
    ) -> list[dict]:
        """Return list of {section_type, page, lines[]}."""
        sections: list[dict] = []
        current: Optional[dict] = None

        def _flush():
            if current and current["lines"]:
                sections.append(current)

        for page, line in page_lines:
            matched_section = None
            for sec_type, pattern in _SECTION_MARKERS:
                if pattern.match(line):
                    matched_section = sec_type
                    break

            if matched_section:
                _flush()
                current = {"section_type": matched_section, "page": page, "lines": []}
            else:
                if current is None:
                    current = {"section_type": "description", "page": page, "lines": []}
                current["lines"].append((page, line))

        _flush()
        return sections

    # ------------------------------------------------------------------
    # Step 2 – build PatentChunk objects
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        sections: list[dict],
        bib: dict,
    ) -> list[PatentChunk]:
        chunks: list[PatentChunk] = []

        pub = bib.get("publication_number", "UNKNOWN")
        doc_id = re.sub(r"[^a-zA-Z0-9]", "_", pub).lower()

        for sec in sections:
            sec_type = sec["section_type"]
            lines_with_pages = sec["lines"]

            if sec_type == "claims":
                chunks.extend(self._split_claims(lines_with_pages, bib, doc_id, pub))
            else:
                text = "\n".join(l for _, l in lines_with_pages).strip()
                if not text:
                    continue
                # Split long description sections
                for sub_text, page in self._split_text(text, lines_with_pages):
                    chunk_id = self._make_id(doc_id, sec_type)
                    chunks.append(
                        PatentChunk(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            publication_number=pub,
                            application_number=bib.get("application_number"),
                            title=bib.get("title", "Unknown Patent"),
                            applicant=bib.get("applicant"),
                            inventor=bib.get("inventor"),
                            filing_date=bib.get("filing_date"),
                            publication_date=bib.get("publication_date"),
                            classification=bib.get("classification"),
                            status=bib.get("status"),
                            source=bib.get("source", "IP India"),
                            source_url=bib.get("source_url"),
                            section_type=sec_type,
                            page=page,
                            text=sub_text,
                        )
                    )

        return chunks

    def _split_claims(
        self,
        lines_with_pages: list[tuple[int, str]],
        bib: dict,
        doc_id: str,
        pub: str,
    ) -> list[PatentChunk]:
        """Parse claims section and emit one chunk per individual claim."""
        chunks: list[PatentChunk] = []
        current_claim_num: Optional[int] = None
        current_lines: list[str] = []
        current_page: int = 1

        def _flush_claim():
            if current_claim_num is not None and current_lines:
                text = " ".join(current_lines).strip()
                chunk_id = self._make_id(doc_id, f"claim{current_claim_num}")
                chunks.append(
                    PatentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        publication_number=pub,
                        application_number=bib.get("application_number"),
                        title=bib.get("title", "Unknown Patent"),
                        applicant=bib.get("applicant"),
                        inventor=bib.get("inventor"),
                        filing_date=bib.get("filing_date"),
                        publication_date=bib.get("publication_date"),
                        classification=bib.get("classification"),
                        status=bib.get("status"),
                        source=bib.get("source", "IP India"),
                        source_url=bib.get("source_url"),
                        section_type="claim",
                        claim_number=current_claim_num,
                        page=current_page,
                        text=text,
                    )
                )

        for page, line in lines_with_pages:
            m = _CLAIM_START_RE.match(line)
            if m:
                _flush_claim()
                current_claim_num = int(m.group(1))
                current_lines = [line]
                current_page = page
            else:
                current_lines.append(line)

        _flush_claim()
        return chunks

    def _split_text(
        self,
        text: str,
        lines_with_pages: list[tuple[int, str]],
    ) -> list[tuple[str, int]]:
        """Yield (sub_text, page) pairs no longer than max_description_chars."""
        if len(text) <= self.max_desc:
            page = lines_with_pages[0][0] if lines_with_pages else 1
            return [(text, page)]

        results = []
        start = 0
        while start < len(text):
            chunk = text[start : start + self.max_desc]
            # find the page of the first character
            cumulative = 0
            page = lines_with_pages[0][0] if lines_with_pages else 1
            for p, l in lines_with_pages:
                if cumulative >= start:
                    page = p
                    break
                cumulative += len(l) + 1  # +1 for newline
            results.append((chunk.strip(), page))
            start += self.max_desc

        return results

    def _make_id(self, doc_id: str, label: str) -> str:
        base = f"{doc_id}_{label}"
        count = self._counter.get(base, 0)
        self._counter[base] = count + 1
        return base if count == 0 else f"{base}_{count}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()
