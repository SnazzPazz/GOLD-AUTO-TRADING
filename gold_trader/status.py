"""Local, file-based status snapshot of the live demo loop.

Answers "is the bot running, and what is it doing right now" (equity, halt
state, last signal, cycle count) without needing to read raw logs or any
credentials. Written by the demo loop after every cycle; read by
`python main.py status`.

Deliberately just a snapshot, not a control channel: this doesn't let a
separate process reach into a running demo loop and change its state (clear
a halt, force a trade). A halt still has to be reviewed by restarting the
loop or acting within it -- this only makes that state visible from outside.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

STATUS_PATH = Path(__file__).resolve().parent.parent / "logs" / "status.json"

# If the status file hasn't been updated in longer than this many poll
# intervals, the loop that was writing it has likely stopped or crashed
# rather than just being between polls.
STALE_POLL_MULTIPLE = 3


@dataclass
class StatusSnapshot:
    updated_at: float
    broker: str
    instrument: str
    granularity: str
    poll_interval_seconds: int
    cycle_count: int
    equity: float
    open_position: bool
    can_trade: bool
    halt_reason: str | None
    halted_for_day: bool
    halted_permanently: bool
    halted_catastrophically: bool
    last_action: str  # "flat" | "opened" | "rejected" | "halted" | "regime_skip"
    last_signal_direction: str | None
    last_signal_confidence: float | None
    trades_today: int


def write_status(snapshot: StatusSnapshot) -> None:
    STATUS_PATH.parent.mkdir(exist_ok=True)
    # Write to a temp file and rename (atomic on the same filesystem) so a
    # concurrent reader never sees a half-written/truncated JSON file.
    tmp_path = STATUS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(asdict(snapshot), indent=2))
    tmp_path.replace(STATUS_PATH)


def read_status() -> StatusSnapshot | None:
    if not STATUS_PATH.exists():
        return None
    data = json.loads(STATUS_PATH.read_text())
    return StatusSnapshot(**data)


def is_stale(snapshot: StatusSnapshot, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    threshold = max(snapshot.poll_interval_seconds, 1) * STALE_POLL_MULTIPLE
    return (now - snapshot.updated_at) > threshold
