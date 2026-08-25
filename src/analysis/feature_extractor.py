"""
Feature Extractor  (Phase 4)
─────────────────────────────
Converts a user's invention description (or a structured Invention object)
into a list of discrete InventionFeature objects.

Each feature gets a stable ID (F1, F2, …) so the rest of the pipeline
can reference them by ID in the feature-match matrix.

Example output for:
  "Herbal formulation containing neem extract, turmeric extract and
   ashwagandha extract in a 2:1:1 ratio for inflammatory skin conditions"

[
  F1  neem extract                     (component)
  F2  turmeric extract                 (component)
  F3  ashwagandha extract              (component)
  F4  2:1:1 composition ratio          (ratio)
  F5  treatment of inflammatory skin conditions  (use)
  F6  herbal formulation               (composition)
]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention, InventionFeature


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert patent analyst specialising in feature decomposition.

Given an invention description, extract EVERY discrete technical feature
and return a JSON array. Each element must have EXACTLY:

{
  "id":       "F1"  (F1, F2, F3, … in order),
  "feature":  "plain-text description of this specific feature",
  "category": one of: component | method | use | composition | ratio | structural | other
}

Rules:
- Each feature must be atomic (one concept per feature).
- Components / ingredients = category "component".
- Proportions / ratios / concentrations = category "ratio".
- Intended medical/technical use = category "use".
- Overall formulation type (tablet, cream, extract) = category "composition".
- Process steps = category "method".
- Structural elements (shape, size) = category "structural".
- Extract 4–10 features. More is better than fewer.
- Output ONLY a valid JSON array. No markdown. No explanation.
"""

_USER = "Invention:\n\n{text}\n\nJSON array of features:"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, llm: Optional[ChatGroq] = None):
        self._llm = llm or ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

    def extract_from_text(self, description: str) -> list[InventionFeature]:
        """Extract features directly from a free-text description."""
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER.format(text=description.strip())},
        ]
        return self._parse_response(self._llm.invoke(messages).content)

    def extract_from_invention(self, invention: Invention) -> list[InventionFeature]:
        """
        Extract features from a structured Invention.
        Builds a rich text representation first so the LLM has full context.
        """
        parts = [f"Title: {invention.title}"]
        if invention.technical_field:
            parts.append(f"Technical field: {invention.technical_field}")
        if invention.problem:
            parts.append(f"Problem: {invention.problem}")
        if invention.solution:
            parts.append(f"Solution: {invention.solution}")
        if invention.components:
            parts.append(f"Components: {', '.join(invention.components)}")
        if invention.intended_use:
            parts.append(f"Intended use: {invention.intended_use}")
        if invention.novelty_claim:
            parts.append(f"Novelty claim: {invention.novelty_claim}")

        return self.extract_from_text("\n".join(parts))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> list[InventionFeature]:
        raw = raw.strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find the first JSON array
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = []
            else:
                data = []

        if not isinstance(data, list):
            data = []

        features: list[InventionFeature] = []
        valid_categories = {
            "component", "method", "use", "composition",
            "ratio", "structural", "other"
        }

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            fid      = item.get("id") or f"F{i+1}"
            feature  = str(item.get("feature", "")).strip()
            category = str(item.get("category", "other")).lower()
            if category not in valid_categories:
                category = "other"
            if not feature:
                continue
            features.append(InventionFeature(
                id=fid,
                feature=feature,
                category=category,
            ))

        # Re-index to ensure F1, F2, … even if LLM returned odd IDs
        for i, f in enumerate(features):
            f.id = f"F{i+1}"

        return features
