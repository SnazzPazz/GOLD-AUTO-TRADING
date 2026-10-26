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
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "decision_log.jsonl"

# One line per poll cycle, indefinitely, with no rotation would grow unbounded over a
# long unattended run (see deploy/README.md). Reusing RotatingFileHandler rather than
# hand-rolling rotation: at a typical ~250-byte line and a 60s poll interval, 10MB is
# roughly a month of history per file; 10 backups is roughly a year of total retention.
_MAX_BYTES = 10_000_000
_BACKUP_COUNT = 10

_logger = logging.getLogger("gold_trader.decision_log")
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Tracked separately from _logger.handlers rather than read back off it: test runners
# (pytest's log-capture plugin, in particular) attach their own extra handlers to named
# loggers for the duration of a test, so "handlers[0]" is not reliably ours to inspect.
# Keeping our own reference means we only ever touch the one handler we added.
_active_handler: RotatingFileHandler | None = None
_active_path: str | None = None


def _get_logger() -> logging.Logger:
    """(Re)points the module logger's handler at the current LOG_PATH.

    Indirection (instead of wiring the handler up once at import time)
    exists so tests can monkeypatch LOG_PATH to a temp file and get a
    handler that actually writes there, rather than whichever path was in
    effect when this module first got imported -- and so importing this
    module has no filesystem side effect until something actually logs.
    """
    global _active_handler, _active_path
    target = str(LOG_PATH)
    if _active_path != target:
        if _active_handler is not None:
            _logger.removeHandler(_active_handler)
            _active_handler.close()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _active_handler = RotatingFileHandler(LOG_PATH, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT)
        _active_handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSON line, no preamble
        _logger.addHandler(_active_handler)
        _active_path = target
    return _logger


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
    _get_logger().info(json.dumps(entry))
