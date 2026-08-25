"""
Query Generator
────────────────
Takes a structured Invention and generates 5-10 focused search queries
targeting different angles: components, use, technology, legal relevance.

These queries are used to drive both vector and BM25 retrieval across
the legal corpus, patent corpus, and TK/AYUSH corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention


def generate_queries(invention: Invention) -> list[str]:
    """
    Generate a deduplicated list of search queries from a structured Invention.

    Strategy
    --------
    1. Component combinations           — ingredient/part focused
    2. Use-case + components            — problem/solution focused
    3. Technical field + use            — domain focused
    4. Legal relevance                  — patent law exclusion checks
    5. Traditional knowledge angle      — TK/AYUSH overlap check
    6. Keyword combinations             — broad sweep
    """
    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)

    comps = invention.components
    use   = invention.intended_use or ""
    field = invention.technical_field or ""
    kws   = invention.keywords
    title = invention.title

    # ── 1. Component queries ──────────────────────────────────────────────
    if comps:
        _add(" ".join(comps))
        if len(comps) >= 2:
            _add(f"{comps[0]} {comps[1]} formulation")
        if len(comps) >= 3:
            _add(f"{comps[0]} {comps[1]} {comps[2]} composition")

    # ── 2. Use + components ───────────────────────────────────────────────
    if use and comps:
        _add(f"{' '.join(comps[:3])} {use}")
        _add(f"formulation for {use} {''.join(comps[:2])}")

    # ── 3. Technical field + use ──────────────────────────────────────────
    if field and use:
        _add(f"{field} {use}")
        _add(f"{field} formulation {use}")

    # ── 4. Legal relevance queries ────────────────────────────────────────
    # Always check patentability exclusions for any invention
    _add(f"patentability {field or title}")
    _add(f"Section 3 Patents Act {field or title}")
    if comps:
        _add(f"traditional knowledge {comps[0]}")
        _add(f"Section 3(p) traditional knowledge {use or comps[0]}")

    # ── 5. TK / AYUSH angle ───────────────────────────────────────────────
    if comps:
        _add(f"Ayurvedic {' '.join(comps[:3])}")
        _add(f"traditional medicine {use or comps[0]}")
    _add(f"TKDL {title}")

    # ── 6. Keyword sweep ──────────────────────────────────────────────────
    if kws:
        _add(" ".join(kws[:5]))
        for kw in kws[:4]:
            _add(kw)

    # Title itself
    _add(title)

    return queries[:12]   # cap at 12 to keep retrieval manageable
