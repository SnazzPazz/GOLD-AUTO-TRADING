"""Combines trend, candlestick patterns, momentum, and (optionally)
news sentiment into a single trade signal.

Design: price action is primary. A trade is only signaled when multiple
independent confirmations line up ("triple confirmation" style), which
cuts down on false positives compared to acting on any single indicator.
Sentiment, when enabled, can only dampen confidence or veto a signal that
contradicts it strongly — it can never originate a signal on its own,
because free news/social data is noisy and frequently lags price.

No amount of confirmation logic makes a leveraged trade risk-free; this
only improves the odds that a given entry has more than one independent
reason behind it. Position sizing and stops are handled downstream by
risk.risk_manager, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from gold_trader.analysis import candlestick_patterns as patterns
from gold_trader.analysis import indicators as ind
from gold_trader.config import settings


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Signal:
    direction: Direction
    confidence: float  # 0..1
    reasons: list[str] = field(default_factory=list)
    price: float = 0.0
    atr: float = 0.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Attach indicators and pattern columns. Call once per fresh candle batch."""
    enriched = ind.compute_all(df)
    pattern_df = patterns.detect_all(df)
    return enriched.join(pattern_df)


def generate_signal(
    enriched_df: pd.DataFrame,
    sentiment_score: float | None = None,
    min_confirmations: int | None = None,
) -> Signal:
    """Evaluate the *last completed* row of an already-`prepare()`d frame."""
    min_confirmations = min_confirmations if min_confirmations is not None else settings.min_confirmations

    if len(enriched_df) < 210:  # need enough bars for EMA-200 etc. to be meaningful
        return Signal(Direction.FLAT, 0.0, ["insufficient history for reliable indicators"])

    row = enriched_df.iloc[-1]
    long_votes: list[str] = []
    short_votes: list[str] = []

    # 1. Trend filter (EMA50 vs EMA200)
    if row["trend"] > 0:
        long_votes.append("uptrend (EMA50 > EMA200)")
    elif row["trend"] < 0:
        short_votes.append("downtrend (EMA50 < EMA200)")

    # 2. Momentum (RSI + MACD histogram)
    if row["rsi_14"] < 70 and row["histogram"] > 0:
        long_votes.append("bullish MACD momentum, RSI not overbought")
    if row["rsi_14"] > 30 and row["histogram"] < 0:
        short_votes.append("bearish MACD momentum, RSI not oversold")

    # 3. Candlestick confirmation
    if any(row.get(p, False) for p in patterns.BULLISH_PATTERNS):
        fired = [p for p in patterns.BULLISH_PATTERNS if row.get(p, False)]
        long_votes.append(f"bullish candlestick pattern: {', '.join(fired)}")
    if any(row.get(p, False) for p in patterns.BEARISH_PATTERNS):
        fired = [p for p in patterns.BEARISH_PATTERNS if row.get(p, False)]
        short_votes.append(f"bearish candlestick pattern: {', '.join(fired)}")

    # 4. Mean-reversion extreme (Bollinger) as a lightweight 4th confirmation
    if row["close"] <= row["bb_lower"]:
        long_votes.append("price at/below lower Bollinger band")
    if row["close"] >= row["bb_upper"]:
        short_votes.append("price at/above upper Bollinger band")

    long_count, short_count = len(long_votes), len(short_votes)

    if long_count >= min_confirmations and long_count > short_count:
        direction, votes, confirmations = Direction.LONG, long_votes, long_count
    elif short_count >= min_confirmations and short_count > long_count:
        direction, votes, confirmations = Direction.SHORT, short_votes, short_count
    else:
        return Signal(
            Direction.FLAT, 0.0,
            [f"not enough confirmations (long={long_count}, short={short_count}, need {min_confirmations})"],
            price=row["close"], atr=row["atr_14"],
        )

    confidence = min(confirmations / 4.0, 1.0)

    # Sentiment can only veto/dampen, never originate, and only if enabled.
    if sentiment_score is not None:
        if direction is Direction.LONG and sentiment_score < -0.4:
            votes.append(f"sentiment strongly bearish ({sentiment_score:.2f}) -> confidence reduced")
            confidence *= 0.5
        elif direction is Direction.SHORT and sentiment_score > 0.4:
            votes.append(f"sentiment strongly bullish ({sentiment_score:.2f}) -> confidence reduced")
            confidence *= 0.5
        else:
            votes.append(f"sentiment={sentiment_score:.2f} (not contradicting)")

    return Signal(direction, confidence, votes, price=row["close"], atr=row["atr_14"])
