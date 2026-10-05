"""Vectorized technical indicators (pandas/numpy only, no compiled TA-Lib
dependency, so this runs anywhere Python runs).

Every function takes/returns pandas Series aligned to the input candle
DataFrame's index and performs no lookahead: indicator[i] only ever uses
data from rows <= i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (not direction)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def trend_direction(df: pd.DataFrame, fast_span: int = 50, slow_span: int = 200) -> pd.Series:
    """+1 uptrend, -1 downtrend, 0 no clear trend, based on EMA crossover."""
    fast = ema(df["close"], fast_span)
    slow = ema(df["close"], slow_span)
    return np.sign(fast - slow).fillna(0).astype(int)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every indicator this system uses to a copy of the candle frame."""
    out = df.copy()
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)
    out["rsi_14"] = rsi(out["close"], 14)
    macd_df = macd(out["close"])
    out = out.join(macd_df)
    bb_df = bollinger_bands(out["close"])
    out = out.join(bb_df)
    out["atr_14"] = atr(out, 14)
    out["adx_14"] = adx(out, 14)
    out["trend"] = trend_direction(out)
    return out
