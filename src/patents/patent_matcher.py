"""
Patent Matcher  (Phase 4)
──────────────────────────
Builds the feature-to-patent match matrix.

For each (prior-art candidate, invention feature) pair it determines:
  - EXACT   : the feature text appears verbatim (or near-verbatim) in the chunk
  - SEMANTIC: the feature is semantically present (cosine ≥ threshold)
  - NO_MATCH: no meaningful overlap

Uses three layers:
  1. String containment check (fast, catches exact hits)
  2. Synonym / equivalent lookup for common botanical names
  3. Cosine similarity between feature embedding and chunk embedding

Returns a list of PriorArtCandidate objects each containing:
  - a FeatureMatch per feature
  - aggregate coverage score
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingestion.schema import (
    InventionFeature, PatentChunk, FeatureMatch, PriorArtCandidate
)

# ---------------------------------------------------------------------------
# Botanical / chemical synonyms  (extend as needed)
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, list[str]] = {
    "neem":         ["azadirachta indica", "nimtree", "nim", "neem extract"],
    "turmeric":     ["curcuma longa", "curcumin", "haldi", "turmeric extract"],
    "ashwagandha":  ["withania somnifera", "indian ginseng", "ashwagandha extract"],
    "aloe vera":    ["aloe barbadensis", "aloe gel"],
    "tulsi":        ["ocimum sanctum", "holy basil"],
    "ginger":       ["zingiber officinale", "ginger extract"],
    "amla":         ["phyllanthus emblica", "emblica officinalis", "indian gooseberry"],
    "brahmi":       ["bacopa monnieri", "brahmi extract"],
    "giloy":        ["tinospora cordifolia"],
    "neem oil":     ["azadirachta indica oil"],
}


def _expand_synonyms(text: str) -> set[str]:
    """Return the text plus all its known synonyms as a set of lowercase strings."""
    tl = text.lower()
    expanded = {tl}
    for canonical, syns in _SYNONYMS.items():
        all_forms = [canonical] + syns
        if any(f in tl for f in all_forms):
            expanded.update(f.lower() for f in all_forms)
    return expanded


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

EXACT_THRESHOLD    = 0.92   # cosine → treat as "exact"
SEMANTIC_THRESHOLD = 0.55   # cosine → treat as "semantic"


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# ---------------------------------------------------------------------------
# PatentMatcher
# ---------------------------------------------------------------------------

class PatentMatcher:
    """
    Match every invention feature against every candidate patent chunk.

    Parameters
    ----------
    embeddings : shared HuggingFaceEmbeddings
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self._emb = embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_candidates(
        self,
        features: list[InventionFeature],
        candidates: list[PatentChunk],
    ) -> list[PriorArtCandidate]:
        """
        For each candidate, produce a PriorArtCandidate with
        per-feature FeatureMatch entries.

        Returns list sorted by coverage (descending).
        """
        if not features or not candidates:
            return []

        # Pre-compute feature embeddings once
        feature_embs: dict[str, list[float]] = {
            f.id: self._emb.embed_query(f.feature)
            for f in features
        }

        results: list[PriorArtCandidate] = []

        for chunk in candidates:
            candidate = self._match_one(chunk, features, feature_embs)
            results.append(candidate)

        results.sort(
            key=lambda c: (c.matched_count, c.overall_similarity),
            reverse=True,
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_one(
        self,
        chunk: PatentChunk,
        features: list[InventionFeature],
        feature_embs: dict[str, list[float]],
    ) -> PriorArtCandidate:
        """Build a PriorArtCandidate for a single chunk."""

        chunk_text  = chunk.text
        chunk_lower = chunk_text.lower()
        chunk_emb   = self._emb.embed_query(chunk_text[:2000])  # cap for speed

        feature_matches: dict[str, FeatureMatch] = {}
        matched_count   = 0
        score_sum       = 0.0

        for feat in features:
            fm = self._match_feature(
                feat, chunk_text, chunk_lower, chunk_emb,
                feature_embs[feat.id],
            )
            feature_matches[feat.id] = fm
            if fm.matched:
                matched_count += 1
            score_sum += fm.confidence

        overall = score_sum / len(features) if features else 0.0

        return PriorArtCandidate(
            document_id        = chunk.document_id,
            publication_number = chunk.publication_number,
            title              = chunk.title,
            source             = chunk.source,
            filing_date        = chunk.filing_date,
            publication_date   = chunk.publication_date,
            classification     = chunk.classification,
            ipc_codes          = chunk.ipc_codes,
            chunk_id           = chunk.chunk_id,
            feature_matches    = feature_matches,
            matched_count      = matched_count,
            total_features     = len(features),
            overall_similarity = round(overall, 4),
        )

    def _match_feature(
        self,
        feat: InventionFeature,
        chunk_text: str,
        chunk_lower: str,
        chunk_emb: list[float],
        feature_emb: list[float],
    ) -> FeatureMatch:
        """Determine how well *feat* is present in the chunk."""

        feat_lower = feat.feature.lower()

        # ── Layer 1: String containment (exact) ───────────────────────────
        if feat_lower in chunk_lower:
            return FeatureMatch(
                feature_id    = feat.id,
                feature_text  = feat.feature,
                matched       = True,
                match_type    = "exact",
                confidence    = 1.0,
                evidence_text = self._extract_snippet(chunk_text, feat_lower),
                notes         = "String match",
            )

        # ── Layer 2: Synonym expansion ─────────────────────────────────────
        expanded = _expand_synonyms(feat.feature)
        for synonym in expanded:
            if synonym in chunk_lower and synonym != feat_lower:
                return FeatureMatch(
                    feature_id    = feat.id,
                    feature_text  = feat.feature,
                    matched       = True,
                    match_type    = "exact",
                    confidence    = 0.95,
                    evidence_text = self._extract_snippet(chunk_text, synonym),
                    notes         = f"Synonym match: '{synonym}'",
                )

        # ── Layer 3: Semantic cosine similarity ────────────────────────────
        sim = _cosine(feature_emb, chunk_emb)

        if sim >= EXACT_THRESHOLD:
            return FeatureMatch(
                feature_id   = feat.id,
                feature_text = feat.feature,
                matched      = True,
                match_type   = "exact",
                confidence   = round(sim, 4),
                evidence_text= chunk_text[:300],
            )

        if sim >= SEMANTIC_THRESHOLD:
            return FeatureMatch(
                feature_id   = feat.id,
                feature_text = feat.feature,
                matched      = True,
                match_type   = "semantic",
                confidence   = round(sim, 4),
                evidence_text= chunk_text[:300],
            )

        return FeatureMatch(
            feature_id   = feat.id,
            feature_text = feat.feature,
            matched      = False,
            match_type   = "no_match",
            confidence   = round(sim, 4),
        )

    @staticmethod
    def _extract_snippet(text: str, keyword: str, window: int = 200) -> str:
        """Return a snippet of *text* around the first occurrence of *keyword*."""
        idx = text.lower().find(keyword.lower())
        if idx == -1:
            return text[:window]
        start = max(0, idx - 80)
        end   = min(len(text), idx + window)
        snippet = text[start:end].strip()
        return f"…{snippet}…" if start > 0 else snippet
