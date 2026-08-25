"""
Persistent BM25 index over all ingested LegalChunks.

Serialises the index + chunk list to disk so the retrieval layer
can reload it without re-ingesting every time.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from .schema import LegalChunk


_DEFAULT_PATH = Path(__file__).parent.parent.parent / "bm25_index.pkl"


def _tokenise(text: str) -> list[str]:
    """Lower-case, split on whitespace — good enough for legal text."""
    return text.lower().split()


class BM25Store:
    """
    Wraps rank-bm25 with save/load helpers.

    Usage
    -----
    # Building (in ingest.py)
    store = BM25Store()
    store.build(chunks)
    store.save()

    # Loading (in retriever.py)
    store = BM25Store.load()
    results = store.query("Section 3(p) traditional knowledge", top_k=10)
    """

    def __init__(self):
        self._chunks: list[LegalChunk] = []
        self._index: Optional[BM25Okapi] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[LegalChunk]) -> None:
        self._chunks = chunks
        corpus = [_tokenise(c.text) for c in chunks]
        self._index = BM25Okapi(corpus)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 10) -> list[LegalChunk]:
        if self._index is None or not self._chunks:
            return []
        tokens = _tokenise(question)
        scores = self._index.get_scores(tokens)
        # argsort descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: Path = _DEFAULT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"chunks": self._chunks, "index": self._index}, fh)
        print(f"BM25 index saved → {path}  ({len(self._chunks)} chunks)")

    @classmethod
    def load(cls, path: Path = _DEFAULT_PATH) -> "BM25Store":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run ingest.py first."
            )
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        store = cls()
        store._chunks = data["chunks"]
        store._index = data["index"]
        print(f"BM25 index loaded ← {path}  ({len(store._chunks)} chunks)")
        return store
