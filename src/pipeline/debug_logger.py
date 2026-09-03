"""
Pipeline Debug Logger  (Phase 11 — 2nd half)
──────────────────────────────────────────────
Logs every pipeline step to stdout / a log file for development debugging.

Prints:
  QUERY → LANGUAGE → IP TYPE → JURISDICTION
  → TOOLS USED → SOURCES RETRIEVED → CONFIDENCE → FINAL STATUS

Can be disabled in production via enabled=False.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Optional


class PipelineDebugLogger:
    """
    Lightweight step-by-step logger for the unified pipeline.

    Usage:
        log = PipelineDebugLogger(enabled=True)
        log.step("language",  language="en", method="override")
        log.step("scope",     scope="in_scope", confidence=0.88)
        log.step("tools",     tools=["legal_search", "tk_search"])
        log.step("evidence",  count=7, confidence="medium")
        log.step("status",    status="answered", confidence=0.86)
        log.summary()
    """

    def __init__(self, enabled: bool = True, request_id: Optional[str] = None):
        self.enabled    = enabled
        self.request_id = request_id or "N/A"
        self._steps: list[dict] = []
        self._start = datetime.now(timezone.utc)

    def step(self, name: str, **kwargs) -> None:
        """Record and optionally print a pipeline step."""
        if not self.enabled:
            return
        entry = {"step": name, **kwargs}
        self._steps.append(entry)
        self._print_step(entry)

    def summary(self) -> None:
        """Print the full pipeline trace."""
        if not self.enabled:
            return
        elapsed = (datetime.now(timezone.utc) - self._start).total_seconds()
        print(f"\n{'─'*55}")
        print(f"  REQUEST: {self.request_id}  ({elapsed:.2f}s)")
        print(f"{'─'*55}")
        for s in self._steps:
            name = s.pop("step", "?")
            vals = "  ".join(f"{k}={v}" for k, v in s.items())
            print(f"  [{name:20s}] {vals}")
            s["step"] = name
        print(f"{'─'*55}\n")

    @staticmethod
    def _print_step(entry: dict) -> None:
        name = entry.get("step", "?")
        vals = "  ".join(
            f"{k}={v}" for k, v in entry.items() if k != "step"
        )
        print(f"  [PIPELINE:{name:18s}] {vals}", file=sys.stdout, flush=True)

    def get_steps(self) -> list[dict]:
        return self._steps


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience (disabled by default in production)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_ENABLED = False   # set to True during development


def make_logger(
    request_id: Optional[str] = None,
    enabled   : Optional[bool] = None,
) -> PipelineDebugLogger:
    en = enabled if enabled is not None else _DEFAULT_ENABLED
    return PipelineDebugLogger(enabled=en, request_id=request_id)
