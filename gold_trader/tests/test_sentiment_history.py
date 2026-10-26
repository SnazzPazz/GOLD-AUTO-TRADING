import tempfile
from pathlib import Path

from gold_trader.sentiment.history import PointInTimeSentimentLookup, load_history, record


def test_record_and_load_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sentiment_history.csv"
        record(0.5, 10, as_of=1000.0, path=path)
        record(-0.3, 5, as_of=2000.0, path=path)

        records = load_history(path)
        assert len(records) == 2
        assert records[0].score == 0.5
        assert records[1].score == -0.3


def test_lookup_returns_most_recent_record_at_or_before_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sentiment_history.csv"
        record(0.1, 1, as_of=1000.0, path=path)
        record(0.5, 2, as_of=2000.0, path=path)
        record(-0.2, 3, as_of=3000.0, path=path)

        lookup = PointInTimeSentimentLookup(path=path)
        assert lookup.score_at(2500.0) == 0.5
        assert lookup.score_at(3000.0) == -0.2
        assert lookup.score_at(999.0) is None  # before any recorded history


def test_lookup_never_leaks_future_sentiment():
    """The entire point of this module: a lookup for time T must never see
    a record whose timestamp is after T, since that would be exactly the
    lookahead bug this exists to prevent."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sentiment_history.csv"
        record(-0.9, 1, as_of=5000.0, path=path)  # far in the "future"
        record(0.2, 1, as_of=1000.0, path=path)

        lookup = PointInTimeSentimentLookup(path=path)
        assert lookup.score_at(1500.0) == 0.2  # only the earlier record is visible


def test_empty_history_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nonexistent.csv"
        lookup = PointInTimeSentimentLookup(path=path)
        assert lookup.score_at(1000.0) is None
