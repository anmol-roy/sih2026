"""
Shared QdrantClient singleton.

All components must import `get_qdrant_client()` instead of
creating their own QdrantClient(path=...) instance.
Qdrant local (SQLite) only allows one open handle at a time.
"""

from __future__ import annotations

from pathlib import Path
from qdrant_client import QdrantClient

_client: QdrantClient | None = None

QDRANT_PATH = str(Path(__file__).parent.parent / "qdrant_db")


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client
