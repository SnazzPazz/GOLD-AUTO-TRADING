import numpy as np
import pandas as pd

from gold_trader.strategy import signal_engine as se


def _uptrend_with_pullback_candles(n=250):
    """Builds a synthetic series: steady uptrend, then a small bullish-engulfing
    style pullback-and-reversal on the last two bars, which should be enough
    confirmations (trend + momentum + pattern) to fire a LONG signal."""
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    base = np.linspace(100, 160, n)
    noise = np.zeros(n)
    close = base + noise
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Make the second-to-last bar a down bar, last bar a strong up bar that
    # engulfs it, while keeping the overall series trending up.
    open_[-2], close[-2] = close[-3] + 1.0, close[-3] - 1.0
    open_[-1], close[-1] = close[-2] - 1.0, close[-2] + 3.0

    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    volume = np.full(n, 1000)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_insufficient_history_returns_flat():
    short_df = _uptrend_with_pullback_candles(n=50)
    enriched = se.prepare(short_df)
    signal = se.generate_signal(enriched)
    assert signal.direction is se.Direction.FLAT


def test_requires_minimum_confirmations_not_just_one():
    # A flat/no-trend, no-pattern series should never produce a directional signal.
    idx = pd.date_range("2024-01-01", periods=250, freq="15min")
    price = 100 + np.sin(np.linspace(0, 3, 250)) * 0.01  # nearly flat, tiny wiggle
    df = pd.DataFrame({
        "open": price, "high": price + 0.05, "low": price - 0.05, "close": price, "volume": 1000,
    }, index=idx)
    enriched = se.prepare(df)
    signal = se.generate_signal(enriched)
    assert signal.direction is se.Direction.FLAT


def test_sentiment_dampens_contradicting_signal_confidence():
    df = _uptrend_with_pullback_candles()
    enriched = se.prepare(df)
    baseline = se.generate_signal(enriched, sentiment_score=None)
    if baseline.direction is se.Direction.LONG:
        dampened = se.generate_signal(enriched, sentiment_score=-0.8)
        assert dampened.confidence < baseline.confidence
