"""Persistent, append-only record of every signal evaluation in the live
demo loop -- not just the ones that became trades.

Mirrors the "decision log" idea from LLM-agent trading frameworks like
TradingAgents (which keeps a running log of every decision and its
reasoning, used for both audit and the system's own memory of past
outcomes): an auditable trail of what the system saw and why it did or
didn't act, so "is this reliable" can be checked against the record
instead of trusted on a summary.

Deliberately not wired into the backtester: writing a log line per bar
there would mean thousands of file-append calls per run and slow backtests
down for no real benefit (the backtest's own trade list + equity curve
already is its record). This is for the live loop, where bars are minutes
apart and every decision is worth being able to look back at.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "decision_log.jsonl"


def record_decision(
    *,
    timestamp: str,
    price: float,
    direction: str,
    confidence: float,
    reasons: list[str],
    approved: bool,
    rejection_reason: str | None,
    action: str,  # "flat" | "opened" | "rejected" | "halted"
) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": timestamp,
        "price": price,
        "direction": direction,
        "confidence": confidence,
        "reasons": reasons,
        "approved": approved,
        "rejection_reason": rejection_reason,
        "action": action,
        "logged_at": time.time(),
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
