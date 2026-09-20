"""
Invention Extractor
────────────────────
Takes a free-text invention description from the user and returns a
structured Invention object using an LLM with JSON output.

Usage
-----
extractor = InventionExtractor(llm)
invention = extractor.extract("I developed an herbal formulation …")
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_mistralai import ChatMistralAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are an expert patent analyst.

Given a free-text invention description, extract structured information and
return a single JSON object with EXACTLY these keys:

{
  "title":          string  — short descriptive title (max 12 words),
  "technical_field": string  — technical domain (e.g. "Herbal medicine"),
  "problem":        string  — problem or need addressed,
  "solution":       string  — how the invention solves the problem,
  "components":     [string] — list of key ingredients / parts / elements,
  "intended_use":   string  — primary use or application,
  "keywords":       [string] — 6-10 search keywords,
  "novelty_claim":  string  — what the inventor claims is new (or null)
}

Rules:
- Output ONLY valid JSON. No markdown. No explanation.
- components and keywords must be JSON arrays of strings.
- If a field cannot be determined, use null.
"""

_USER_TEMPLATE = "Invention description:\n\n{description}\n\nJSON:"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class InventionExtractor:
    def __init__(self, llm: Optional[ChatMistralAI] = None):
        import os
        self._llm = llm or ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

    def extract(self, description: str) -> Invention:
        """
        Parse *description* into a structured Invention.
        Falls back to a minimal Invention if the LLM returns bad JSON.
        """
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TEMPLATE.format(description=description.strip())},
        ]

        response = self._llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract the first JSON object
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        # Coerce list fields
        for field in ("components", "keywords"):
            val = data.get(field)
            if not isinstance(val, list):
                data[field] = [str(val)] if val else []

        # Ensure required title
        if not data.get("title"):
            data["title"] = description[:80].strip()

        return Invention(
            title=data.get("title", "Untitled invention"),
            technical_field=data.get("technical_field"),
            problem=data.get("problem"),
            solution=data.get("solution"),
            components=data.get("components", []),
            intended_use=data.get("intended_use"),
            keywords=data.get("keywords", []),
            novelty_claim=data.get("novelty_claim"),
        )
