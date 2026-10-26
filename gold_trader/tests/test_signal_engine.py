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


def test_generate_signal_matches_row_based_equivalent_at_every_bar():
    """generate_signal(df) is documented as a thin wrapper over
    generate_signal_from_row(df.iloc[-1], len(df)-1, ...) -- the backtester's
    hot loop calls the row-based version directly to avoid re-slicing a
    growing DataFrame every bar. If these two ever diverge, the backtester's
    results would silently stop matching what generate_signal alone predicts,
    so pin the equivalence at several different history lengths, not just one."""
    df = _uptrend_with_pullback_candles(n=260)
    enriched = se.prepare(df)
    for n in (209, 210, 211, 230, 260):
        via_frame = se.generate_signal(enriched.iloc[:n], sentiment_score=0.2, min_confirmations=2)
        via_row = se.generate_signal_from_row(enriched.iloc[n - 1], n - 1, sentiment_score=0.2, min_confirmations=2)
        assert via_frame.direction is via_row.direction
        assert via_frame.confidence == via_row.confidence
        assert via_frame.reasons == via_row.reasons
        assert via_frame.price == via_row.price
        assert via_frame.atr == via_row.atr


def test_generate_signal_from_row_boundary_matches_length_check():
    """bar_index < 209 must reject exactly where len(enriched_df) < 210 used
    to -- this is the off-by-one the row-based refactor had to get right."""
    df = _uptrend_with_pullback_candles(n=260)
    enriched = se.prepare(df)
    row_209 = enriched.iloc[209]  # the 210th row, len() == 210 there -> should NOT be flat-for-insufficient-history
    row_208 = enriched.iloc[208]  # the 209th row, len() == 209 there -> should be flat-for-insufficient-history

    assert se.generate_signal_from_row(row_208, 208).reasons == ["insufficient history for reliable indicators"]
    assert se.generate_signal_from_row(row_209, 209).reasons != ["insufficient history for reliable indicators"]
