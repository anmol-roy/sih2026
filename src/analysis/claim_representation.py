"""
Claim Representation  (Phase 4)
────────────────────────────────
Builds a claim-like technical representation of the invention from its
structured features. This is NOT a legally drafted patent claim — it is
a normalised technical structure used for prior-art comparison.

Also suggests IPC classification codes based on technical field + components.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_mistralai import ChatMistralAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import Invention, InventionFeature, ClaimRepresentation


# ---------------------------------------------------------------------------
# IPC hint table  (common technology areas → IPC codes)
# Expand as needed.
# ---------------------------------------------------------------------------

_IPC_HINTS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"herbal|ayurved|plant|botanical|extract", re.I),    ["A61K 36/00"]),
    (re.compile(r"pharma|drug|medicine|therapeut|treatment", re.I),  ["A61K", "A61P"]),
    (re.compile(r"antibacterial|antimicrobial|antibiotic", re.I),    ["A61P 31/00"]),
    (re.compile(r"anti.?inflam|inflammation|skin", re.I),            ["A61P 17/00", "A61P 29/00"]),
    (re.compile(r"formulation|composition|cream|ointment", re.I),    ["A61K 9/00"]),
    (re.compile(r"nanoparticle|nano", re.I),                         ["B82Y 5/00"]),
    (re.compile(r"computer|software|algorithm|neural", re.I),        ["G06N", "G06F"]),
    (re.compile(r"mechanical|machine|device|apparatus", re.I),       ["B25", "F16"]),
    (re.compile(r"chemical|compound|synthesis", re.I),               ["C07", "C08"]),
    (re.compile(r"food|nutrition|supplement", re.I),                 ["A23L", "A23V"]),
    (re.compile(r"agriculture|pesticide|fertilizer", re.I),          ["A01N", "A01P"]),
    (re.compile(r"semiconductor|circuit|electronic", re.I),          ["H01L", "H03"]),
]


def _suggest_ipc(
    invention: Invention,
    features: list[InventionFeature],
) -> list[str]:
    search_text = " ".join(filter(None, [
        invention.title,
        invention.technical_field or "",
        " ".join(invention.components),
        invention.intended_use or "",
        " ".join(f.feature for f in features),
    ]))

    codes: list[str] = []
    seen: set[str] = set()
    for pattern, ipc_list in _IPC_HINTS:
        if pattern.search(search_text):
            for code in ipc_list:
                if code not in seen:
                    seen.add(code)
                    codes.append(code)
    return codes[:5]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert patent drafter.

Given an invention's technical features, produce a claim-like technical
representation and return a single JSON object with EXACTLY these keys:

{
  "claim_type": one of: composition | method | device | system | process | use,
  "elements":   [string]  — list of technical elements that define the invention,
  "independent_claim": string — one sentence independent claim draft starting with
                                "A [claim_type] comprising …" or "A method for …"
}

Rules:
- Output ONLY valid JSON. No markdown. No explanation.
- elements must list every important technical element, 4-10 items.
- The independent_claim must be ONE sentence, technically precise.
- Do not include legal boilerplate.
"""

_USER = """\
Invention title: {title}
Technical field: {field}
Features:
{features_text}

JSON:"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class ClaimBuilder:
    def __init__(self, llm: Optional[ChatMistralAI] = None):
        import os
        self._llm = llm or ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            api_key=os.getenv("MISTRAL_API_KEY")
        )

    def build(
        self,
        invention: Invention,
        features: list[InventionFeature],
    ) -> ClaimRepresentation:
        features_text = "\n".join(
            f"  {f.id} [{f.category}]: {f.feature}"
            for f in features
        )

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER.format(
                title=invention.title,
                field=invention.technical_field or "Not specified",
                features_text=features_text,
            )},
        ]

        raw = self._llm.invoke(messages).content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}

        valid_types = {"composition", "method", "device", "system", "process", "use"}
        claim_type = data.get("claim_type", "composition")
        if claim_type not in valid_types:
            claim_type = "composition"

        elements = data.get("elements", [])
        if not isinstance(elements, list):
            elements = [str(elements)]

        independent = data.get("independent_claim", "")
        if not independent:
            independent = (
                f"A {claim_type} comprising "
                + ", ".join(f.feature for f in features[:5])
            )

        ipc_codes = _suggest_ipc(invention, features)

        return ClaimRepresentation(
            claim_type=claim_type,
            elements=elements,
            independent_claim=independent,
            ipc_suggested=ipc_codes,
        )
