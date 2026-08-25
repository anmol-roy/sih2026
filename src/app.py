"""
IP-SAKTI Sahayak — Phase 3
───────────────────────────
This file re-exports the Phase 3 FastAPI app so both of these work:

    uvicorn src.app:app --reload          (legacy entry point)
    uvicorn src.api.main:app --reload     (canonical entry point)

All routes and logic live in src/api/main.py.
"""

from api.main import app  # noqa: F401

__all__ = ["app"]
