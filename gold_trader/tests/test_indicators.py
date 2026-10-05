import numpy as np
import pandas as pd

from gold_trader.analysis import indicators as ind


def _flat_df(n=300, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1000},
        index=idx,
    )


def test_rsi_flat_price_is_neutral():
    df = _flat_df()
    rsi = ind.rsi(df["close"])
    assert np.isclose(rsi.iloc[-1], 50.0, atol=5)


def test_rsi_bounds():
    idx = pd.date_range("2024-01-01", periods=200, freq="15min")
    prices = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 200))
    df = pd.DataFrame({"close": prices}, index=idx)
    rsi = ind.rsi(df["close"])
    assert rsi.min() >= 0 and rsi.max() <= 100


def test_atr_zero_on_flat_series():
    df = _flat_df()
    df["high"] = df["close"]
    df["low"] = df["close"]
    atr = ind.atr(df)
    assert np.isclose(atr.iloc[-1], 0.0, atol=1e-6)


def test_trend_direction_uptrend():
    idx = pd.date_range("2024-01-01", periods=300, freq="15min")
    prices = np.linspace(100, 200, 300)
    df = pd.DataFrame({"close": prices}, index=idx)
    trend = ind.trend_direction(df)
    assert trend.iloc[-1] == 1


def test_compute_all_has_expected_columns():
    df = _flat_df()
    out = ind.compute_all(df)
    for col in ["ema_50", "ema_200", "rsi_14", "macd", "signal", "histogram", "bb_upper", "bb_lower", "atr_14", "adx_14", "trend"]:
        assert col in out.columns
