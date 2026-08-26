"""
IP Type Router  (Phase 5)
──────────────────────────
Classifies a user query into one or more IP domains using an LLM.
Supports multi-domain detection for queries that span patent + TK,
patent + trademark, etc.

Falls back to keyword-based classification if the LLM returns
malformed JSON, ensuring the router never crashes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from routing.schemas import IPType, QueryRoute


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an Indian Intellectual Property classification expert.

Classify the user query into one or more IP categories.

Categories and their definitions:
  patent               — Inventions, products, processes, technical solutions,
                         patentability, novelty, inventive step, prior art,
                         patent application, Section 3 exclusions.
  trademark            — Brand names, logos, marks, slogans, trade dress,
                         trademark registration, passing off, infringement.
  copyright            — Books, software, music, films, paintings, artistic works,
                         authorship, copyright ownership, moral rights, infringement.
  design               — Appearance, shape, configuration, pattern, ornamental
                         features of a product, registered design, design protection.
  gi                   — Geographical Indications, products associated with a
                         specific geographical region (e.g. Darjeeling tea, Basmati).
  traditional_knowledge — AYUSH, Ayurveda, Siddha, Unani, Yoga, traditional
                         medicine, traditional formulations, TKDL, herbal medicine,
                         biological resources, ABS (Access and Benefit Sharing).
  unknown              — Cannot determine from the query.

Return ONLY a JSON object with EXACTLY these keys:
{
  "primary"   : string  — the single most relevant category,
  "ip_types"  : [string] — list of ALL relevant categories (1–3 max),
  "confidence": float between 0 and 1,
  "reason"    : string  — one sentence explanation
}

Rules:
- Output ONLY valid JSON. No markdown, no explanation outside the JSON.
- If the query clearly spans two domains (e.g. "trademark my patented product"),
  include both in ip_types and set primary to the most prominent one.
- Never return more than 3 ip_types.
- If unsure, use "unknown" as primary and set confidence below 0.5.
"""

_USER = "User query: {query}\n\nJSON:"


# ─────────────────────────────────────────────────────────────────────────────
# Keyword fallback (used when LLM fails or returns invalid JSON)
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORD_RULES: list[tuple[re.Pattern, IPType]] = [
    (re.compile(r"\bpatent\b|patentab|inventive\s+step|prior\s+art|novelty|Section\s+3|invent", re.I), IPType.PATENT),
    (re.compile(r"\btrademark\b|trade\s*mark|brand\s*name|logo|slogan|passing\s+off|mark\s+register", re.I), IPType.TRADEMARK),
    (re.compile(r"\bcopyright\b|copy\s*right|authorship|moral\s+right|literary|artistic\s+work|software\s+right", re.I), IPType.COPYRIGHT),
    (re.compile(r"\bdesign\b|shape\s+of|appearance\s+of|ornamental|visual\s+feature|design\s+register", re.I), IPType.DESIGN),
    (re.compile(r"\bgeograph|\bgi\b|geographical\s+indication|darjeeling|basmati|region\s+product", re.I), IPType.GI),
    (re.compile(r"\bayush\b|\bayurved|\bsiddha\b|\bunani\b|\btkdl\b|traditional\s+knowledge|traditional\s+medicine|herbal\s+formula|traditional\s+formula", re.I), IPType.TRADITIONAL_KNOWLEDGE),
]


def _keyword_classify(query: str) -> list[IPType]:
    found: list[IPType] = []
    for pattern, ip_type in _KEYWORD_RULES:
        if pattern.search(query):
            found.append(ip_type)
    return found or [IPType.UNKNOWN]


def _coerce_ip_type(val: str) -> IPType:
    val = val.strip().lower().replace(" ", "_")
    try:
        return IPType(val)
    except ValueError:
        # Fuzzy match
        for member in IPType:
            if member.value in val or val in member.value:
                return member
        return IPType.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# IP Router
# ─────────────────────────────────────────────────────────────────────────────

class IPRouter:
    """
    Classifies a user query into IP domain(s).

    Parameters
    ----------
    llm : optional shared ChatGroq instance
    """

    def __init__(self, llm: Optional[ChatGroq] = None):
        self._llm = llm or ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

    def classify(self, query: str) -> QueryRoute:
        """
        Classify *query* into one or more IP types.

        Returns a QueryRoute with primary type, all types, confidence, reason.
        """
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER.format(query=query.strip())},
        ]

        try:
            raw = self._llm.invoke(messages).content.strip()
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)

            primary   = _coerce_ip_type(data.get("primary", "unknown"))
            raw_types = data.get("ip_types", [data.get("primary", "unknown")])
            if not isinstance(raw_types, list):
                raw_types = [raw_types]

            ip_types = list(dict.fromkeys(            # deduplicate, preserve order
                _coerce_ip_type(t) for t in raw_types
            ))
            if primary not in ip_types:
                ip_types.insert(0, primary)

            confidence = float(data.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            reason     = str(data.get("reason", "LLM classification"))

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback to keyword classification
            ip_types   = _keyword_classify(query)
            primary    = ip_types[0]
            confidence = 0.60
            reason     = "Keyword-based fallback classification"

        return QueryRoute(
            ip_types   = ip_types,
            primary    = primary,
            confidence = confidence,
            reason     = reason,
            is_multi   = len(ip_types) > 1,
        )
