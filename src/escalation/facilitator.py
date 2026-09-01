"""
Human Escalation  (Phase 9)
────────────────────────────
Generates escalation requests and decides when to escalate.

For the prototype, escalation requests are stored in the audit DB
(via audit.logger) and the API returns a structured escalation response.
A real deployment would route these to a human IP facilitator queue.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class EscalationRequest(BaseModel):
    request_id  : str
    created_at  : str   = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason      : str
    ip_type     : str   = "unknown"
    jurisdiction: str   = "india"
    confidence  : float = 0.0
    question    : str   = ""         # stored only in non-privacy-mode
    status      : str   = "pending"  # pending | assigned | resolved | closed
    language    : str   = "en"


class EscalationResponse(BaseModel):
    request_id         : str
    status             : str
    message            : str
    estimated_response : str   = "A qualified IP facilitator will review this query."


# ─────────────────────────────────────────────────────────────────────────────
# Request-ID generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_request_id() -> str:
    """
    Generate a stable, traceable request ID.
    Format: REQ-YYYY-NNNNNN (year + 6 random hex chars)
    """
    year = datetime.now(timezone.utc).year
    suffix = uuid.uuid4().hex[:6].upper()
    return f"REQ-{year}-{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Escalation decision
# ─────────────────────────────────────────────────────────────────────────────

ESCALATION_THRESHOLD = 0.60   # confidence below this → escalate

def should_escalate(
    confidence          : float,
    evidence_sufficient : bool,
    conflicting_sources : bool = False,
    user_requests       : bool = False,
) -> bool:
    """
    Return True when the query should be escalated to a human facilitator.

    Triggers:
      - User explicitly requests human review
      - Evidence is insufficient (no authoritative source)
      - Confidence below threshold (< 0.60)
      - Conflicting sources detected
    """
    if user_requests:
        return True
    if not evidence_sufficient:
        return True
    if confidence < ESCALATION_THRESHOLD:
        return True
    if conflicting_sources:
        return True
    return False


def escalation_reason(
    confidence          : float,
    evidence_sufficient : bool,
    conflicting_sources : bool = False,
    user_requests       : bool = False,
) -> str:
    """Return a human-readable reason for escalation."""
    if user_requests:
        return "User explicitly requested human review."
    if not evidence_sufficient:
        return "Insufficient authoritative evidence found in available sources."
    if confidence < ESCALATION_THRESHOLD:
        return f"Low retrieval confidence ({confidence:.0%}) — answer reliability uncertain."
    if conflicting_sources:
        return "Conflicting evidence sources detected."
    return "Escalation conditions met."


# ─────────────────────────────────────────────────────────────────────────────
# Escalation request builder
# ─────────────────────────────────────────────────────────────────────────────

def create_escalation_request(
    reason      : str,
    ip_type     : str   = "unknown",
    jurisdiction: str   = "india",
    confidence  : float = 0.0,
    question    : str   = "",
    language    : str   = "en",
    privacy_mode: bool  = False,
    request_id  : Optional[str] = None,
) -> EscalationRequest:
    """Build an EscalationRequest, redacting question in privacy mode."""
    rid = request_id or generate_request_id()
    return EscalationRequest(
        request_id  = rid,
        reason      = reason,
        ip_type     = ip_type,
        jurisdiction= jurisdiction,
        confidence  = confidence,
        question    = "" if privacy_mode else question,
        language    = language,
    )
