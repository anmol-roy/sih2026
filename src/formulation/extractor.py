"""
Formulation Extractor  (Phase 7)
──────────────────────────────────
Converts a free-text formulation description into a list of
enriched Ingredient objects.

Two-step pipeline:
  1. LLM extracts raw ingredient names from the text
  2. Local ingredients.json enriches each name with scientific name,
     biological-resource flag, traditional-use indicator, and systems

This hybrid approach avoids hallucinated scientific names while still
handling ingredients that aren't in the local DB.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from formulation.schemas import Ingredient

# ─────────────────────────────────────────────────────────────────────────────
# Local ingredient knowledge base
# ─────────────────────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "formulation" / "ingredients.json"

def _load_db() -> dict:
    if _DB_PATH.exists():
        with open(_DB_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}

_INGREDIENT_DB: dict = _load_db()


def _lookup(name: str) -> Optional[dict]:
    """Case-insensitive lookup in the ingredient DB."""
    name_lower = name.lower().strip()
    # Exact match
    if name_lower in _INGREDIENT_DB:
        return _INGREDIENT_DB[name_lower]
    # Partial match — name_lower is substring of a DB key
    for key, val in _INGREDIENT_DB.items():
        if name_lower in key or key in name_lower:
            return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a pharmaceutical ingredient extraction expert.

Given a formulation description, extract ALL ingredient names mentioned.

Return ONLY a JSON array of strings — one string per ingredient name.
Use the common name as mentioned in the text (not the scientific name).

Example:
["neem", "turmeric", "ashwagandha"]

Rules:
- Output ONLY a valid JSON array. No markdown. No explanation. 
- If no ingredients are mentioned, return [].
- Do not add ingredients not explicitly mentioned.
- Use lowercase names.
"""

_USER = "Formulation description:\n\n{text}\n\nJSON array of ingredient names:"


# ─────────────────────────────────────────────────────────────────────────────
# FormulationExtractor
# ─────────────────────────────────────────────────────────────────────────────

class FormulationExtractor:
    """
    Extract and enrich ingredients from a formulation description.

    Parameters
    ----------
    llm : optional shared ChatGroq instance.
          If None, keyword-only extraction is used (no LLM call).
    """

    def __init__(self, llm: Optional[ChatGroq] = None):
        self._llm = llm   # may be None — keyword fallback is used in that case

    def extract(self, description: str) -> list[Ingredient]:
        """
        Extract ingredients from *description*.

        Returns a list of Ingredient objects, enriched from the local DB
        where available.
        """
        raw_names = self._extract_names(description)
        return [self._enrich(name) for name in raw_names]

    # ------------------------------------------------------------------
    # Step 1 — LLM name extraction
    # ------------------------------------------------------------------

    def _extract_names(self, text: str) -> list[str]:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER.format(text=text.strip())},
        ]
        try:
            raw = self._llm.invoke(messages).content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            names = json.loads(raw)
            if isinstance(names, list):
                return [str(n).strip().lower() for n in names if n]
        except Exception:
            pass
        # Fallback: look for known plant names in text
        text_lower = text.lower()
        return [k for k in _INGREDIENT_DB if k in text_lower]

    # ------------------------------------------------------------------
    # Step 2 — DB enrichment
    # ------------------------------------------------------------------

    def _enrich(self, name: str) -> Ingredient:
        db_entry = _lookup(name)
        if db_entry:
            return Ingredient(
                name                     = name,
                scientific_name          = db_entry.get("scientific_name"),
                biological_resource      = db_entry.get("biological_resource", False),
                traditional_use_indicator= db_entry.get("traditional_use_indicator", False),
                traditional_systems      = db_entry.get("traditional_systems", []),
                source                   = "db_enriched",
            )
        # Not in DB — make reasonable inferences
        is_bio = bool(re.search(
            r"\bextract\b|\bherbal\b|\bplant\b|\broot\b|\bleaf\b|\bseed\b|\bbark\b",
            name, re.I
        ))
        return Ingredient(
            name                     = name,
            scientific_name          = None,
            biological_resource      = is_bio,
            traditional_use_indicator= False,
            traditional_systems      = [],
            source                   = "extracted",
        )
