"""Vectorized candlestick pattern detection.

Each function returns a boolean pandas Series aligned to the candle
DataFrame's index, True where the pattern completes on that bar. These are
intentionally dependency-free (pure pandas/numpy) re-implementations of the
classic patterns rather than a TA-Lib binding, so the project has no
compiled-library install step.

Patterns here only look at the current bar and a small fixed window of
prior bars (no lookahead past the bar being evaluated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0, np.nan)


def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def _bullish(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]


def _bearish(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]


def doji(df: pd.DataFrame, body_ratio: float = 0.1) -> pd.Series:
    return (_body(df) / _range(df)) <= body_ratio


def hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish reversal: small body near the top, long lower wick, little/no upper wick."""
    body = _body(df)
    rng = _range(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    return (lower >= 2 * body) & (upper <= body) & ((body / rng) <= 0.35)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Bearish reversal: small body near the bottom, long upper wick, little/no lower wick."""
    body = _body(df)
    rng = _range(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    return (upper >= 2 * body) & (lower <= body) & ((body / rng) <= 0.35)


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    prev_bearish = prev_close < prev_open
    return (
        prev_bearish
        & _bullish(df)
        & (df["close"] >= prev_open)
        & (df["open"] <= prev_close)
        & (_body(df) > (prev_open - prev_close).abs())
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    prev_bullish = prev_close > prev_open
    return (
        prev_bullish
        & _bearish(df)
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
        & (_body(df) > (prev_open - prev_close).abs())
    )


def morning_star(df: pd.DataFrame) -> pd.Series:
    """Three-bar bullish reversal: big down bar, small indecisive bar, big up bar."""
    c1_bear = _bearish(df.shift(2))
    c1_body = _body(df.shift(2))
    c2_small = _body(df.shift(1)) <= 0.4 * c1_body
    c3_bull = _bullish(df)
    c3_closes_into_c1 = df["close"] >= (df["open"].shift(2) + df["close"].shift(2)) / 2
    return c1_bear & c2_small & c3_bull & c3_closes_into_c1


def evening_star(df: pd.DataFrame) -> pd.Series:
    """Three-bar bearish reversal: big up bar, small indecisive bar, big down bar."""
    c1_bull = _bullish(df.shift(2))
    c1_body = _body(df.shift(2))
    c2_small = _body(df.shift(1)) <= 0.4 * c1_body
    c3_bear = _bearish(df)
    c3_closes_into_c1 = df["close"] <= (df["open"].shift(2) + df["close"].shift(2)) / 2
    return c1_bull & c2_small & c3_bear & c3_closes_into_c1


def piercing_line(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    midpoint = (prev_open + prev_close) / 2
    return (
        (prev_close < prev_open)
        & _bullish(df)
        & (df["open"] < prev_close)
        & (df["close"] > midpoint)
        & (df["close"] < prev_open)
    )


def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    midpoint = (prev_open + prev_close) / 2
    return (
        (prev_close > prev_open)
        & _bearish(df)
        & (df["open"] > prev_close)
        & (df["close"] < midpoint)
        & (df["close"] > prev_open)
    )


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of boolean pattern columns aligned to df's index."""
    return pd.DataFrame({
        "doji": doji(df),
        "hammer": hammer(df),
        "shooting_star": shooting_star(df),
        "bullish_engulfing": bullish_engulfing(df),
        "bearish_engulfing": bearish_engulfing(df),
        "morning_star": morning_star(df),
        "evening_star": evening_star(df),
        "piercing_line": piercing_line(df),
        "dark_cloud_cover": dark_cloud_cover(df),
    }, index=df.index).fillna(False)


BULLISH_PATTERNS = ["hammer", "bullish_engulfing", "morning_star", "piercing_line"]
BEARISH_PATTERNS = ["shooting_star", "bearish_engulfing", "evening_star", "dark_cloud_cover"]
