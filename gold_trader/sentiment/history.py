"""Point-in-time sentiment history.

TradingAgents (github.com/TauricResearch/TradingAgents) treats look-ahead
bugs as serious enough to list individually in its changelog across news,
social, and fundamentals data -- every dated input has to reflect only what
was actually knowable at that point in time, or a backtest is lying to you.

This project had exactly that bug, silently: `backtest/backtester.py` never
passed a sentiment score into the signal engine at all, while the live demo
loop always did. So the backtest never validated the sentiment-dampening
path the demo loop actually runs -- "the backtest proves the strategy" and
"the strategy includes sentiment" were both true, just not of the same
strategy.

There's no honest fix that backdates sentiment before this file existed --
the free RSS feeds this system reads have no historical archive, so there
is no way to know what gold-related headlines said on some date last year.
What this file does instead is start recording: every live sentiment
reading the demo loop takes gets appended here with its timestamp, so a
backtest run over a period *after* recording started can replay the exact,
real, point-in-time sentiment this system itself observed -- no lookahead,
because it genuinely didn't exist yet when each reading was taken.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "sentiment_history.csv"


@dataclass
class SentimentRecord:
    timestamp: float
    score: float
    headline_count: int


def record(score: float, headline_count: int, as_of: float | None = None, path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    as_of = as_of if as_of is not None else time.time()
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "score", "headline_count"])
        writer.writerow([as_of, score, headline_count])


def load_history(path: Path = HISTORY_PATH) -> list[SentimentRecord]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for row in csv.DictReader(f):
            records.append(SentimentRecord(float(row["timestamp"]), float(row["score"]), int(row["headline_count"])))
    return records


class PointInTimeSentimentLookup:
    """Returns the most recent sentiment score recorded at or before a given
    timestamp. Returns None when no qualifying record exists (most commonly:
    the backtest period predates any recorded history) -- callers must treat
    that exactly like "sentiment disabled for this bar", never guess at a
    value, since guessing would reintroduce the lookahead this exists to
    prevent.
    """

    def __init__(self, records: list[SentimentRecord] | None = None, path: Path = HISTORY_PATH):
        self.records = sorted(records if records is not None else load_history(path), key=lambda r: r.timestamp)

    def score_at(self, as_of_timestamp: float) -> float | None:
        result = None
        for rec in self.records:
            if rec.timestamp <= as_of_timestamp:
                result = rec.score
            else:
                break
        return result
