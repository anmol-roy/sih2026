"""
Audit Logger  (Phase 9)
────────────────────────
SQLite-based audit log.

What IS logged:
  request_id, timestamp, language, ip_type, jurisdiction,
  retrieval_count, citation_count, confidence, status, human_review,
  scope, escalation_reason

What is NEVER logged:
  ❌ API keys or credentials
  ❌ Raw invention descriptions (in privacy_mode)
  ❌ System prompts
  ❌ Generated answers
  ❌ Internal file paths

Privacy mode (privacy_mode=True):
  - question stored as empty string
  - no invention text stored

Database: data/audit.db  (SQLite, auto-created)
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_DB_PATH = Path(__file__).parent.parent.parent / "data" / "audit.db"
_LOCK    = threading.Lock()   # SQLite isn't thread-safe by default

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT    NOT NULL UNIQUE,
    timestamp         TEXT    NOT NULL,
    language          TEXT    DEFAULT 'en',
    ip_type           TEXT    DEFAULT 'unknown',
    jurisdiction      TEXT    DEFAULT 'india',
    retrieval_count   INTEGER DEFAULT 0,
    citation_count    INTEGER DEFAULT 0,
    confidence        REAL    DEFAULT 0.0,
    status            TEXT    DEFAULT 'answered',
    human_review      INTEGER DEFAULT 0,
    scope             TEXT    DEFAULT 'in_scope',
    escalation_reason TEXT    DEFAULT '',
    privacy_mode      INTEGER DEFAULT 0
)
"""

_CREATE_ESCALATIONS = """
CREATE TABLE IF NOT EXISTS escalation_requests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    reason       TEXT    DEFAULT '',
    ip_type      TEXT    DEFAULT 'unknown',
    jurisdiction TEXT    DEFAULT 'india',
    confidence   REAL    DEFAULT 0.0,
    question     TEXT    DEFAULT '',
    status       TEXT    DEFAULT 'pending',
    language     TEXT    DEFAULT 'en'
)
"""


# ─────────────────────────────────────────────────────────────────────────────
# AuditLogger
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thread-safe SQLite audit logger.

    Usage:
        logger = AuditLogger()
        logger.log(request_id="REQ-2026-ABC123", language="hi", ...)
        logger.log_escalation(escalation_request)
        logger.delete(request_id)      # DPDP-style deletion
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db = db_path or _DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ──────────────────────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with _LOCK:
            con = sqlite3.connect(self._db)
            try:
                con.execute(_CREATE_TABLE)
                con.execute(_CREATE_ESCALATIONS)
                con.commit()
            finally:
                con.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Log a request
    # ──────────────────────────────────────────────────────────────────────────

    def log(
        self,
        request_id      : str,
        language        : str   = "en",
        ip_type         : str   = "unknown",
        jurisdiction    : str   = "india",
        retrieval_count : int   = 0,
        citation_count  : int   = 0,
        confidence      : float = 0.0,
        status          : str   = "answered",
        human_review    : bool  = False,
        scope           : str   = "in_scope",
        escalation_reason: str  = "",
        privacy_mode    : bool  = False,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        row = (
            request_id, ts, language, ip_type, jurisdiction,
            retrieval_count, citation_count, round(confidence, 4),
            status, int(human_review), scope, escalation_reason,
            int(privacy_mode),
        )
        with _LOCK:
            con = sqlite3.connect(self._db)
            try:
                con.execute(
                    """
                    INSERT OR REPLACE INTO audit_logs
                    (request_id, timestamp, language, ip_type, jurisdiction,
                     retrieval_count, citation_count, confidence,
                     status, human_review, scope, escalation_reason, privacy_mode)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    row,
                )
                con.commit()
            finally:
                con.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Log an escalation request
    # ──────────────────────────────────────────────────────────────────────────

    def log_escalation(self, escalation) -> None:
        """Accept an EscalationRequest object."""
        ts = datetime.now(timezone.utc).isoformat()
        with _LOCK:
            con = sqlite3.connect(self._db)
            try:
                con.execute(
                    """
                    INSERT INTO escalation_requests
                    (request_id, timestamp, reason, ip_type, jurisdiction,
                     confidence, question, status, language)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        escalation.request_id, ts,
                        escalation.reason, escalation.ip_type,
                        escalation.jurisdiction, escalation.confidence,
                        escalation.question,   # empty in privacy_mode
                        escalation.status, escalation.language,
                    ),
                )
                con.commit()
            finally:
                con.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieve
    # ──────────────────────────────────────────────────────────────────────────

    def get(self, request_id: str) -> Optional[dict]:
        with _LOCK:
            con = sqlite3.connect(self._db)
            con.row_factory = sqlite3.Row
            try:
                row = con.execute(
                    "SELECT * FROM audit_logs WHERE request_id = ?", (request_id,)
                ).fetchone()
                return dict(row) if row else None
            finally:
                con.close()

    def list_recent(self, limit: int = 20) -> list[dict]:
        with _LOCK:
            con = sqlite3.connect(self._db)
            con.row_factory = sqlite3.Row
            try:
                rows = con.execute(
                    "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                con.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Delete  (DPDP / privacy right-to-erasure)
    # ──────────────────────────────────────────────────────────────────────────

    def delete(self, request_id: str) -> bool:
        """Delete all records for *request_id*. Returns True if rows were deleted."""
        with _LOCK:
            con = sqlite3.connect(self._db)
            try:
                cur1 = con.execute(
                    "DELETE FROM audit_logs WHERE request_id = ?", (request_id,)
                )
                cur2 = con.execute(
                    "DELETE FROM escalation_requests WHERE request_id = ?", (request_id,)
                )
                con.commit()
                return (cur1.rowcount + cur2.rowcount) > 0
            finally:
                con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_logger: Optional[AuditLogger] = None


def get_logger() -> AuditLogger:
    global _logger
    if _logger is None:
        _logger = AuditLogger()
    return _logger
