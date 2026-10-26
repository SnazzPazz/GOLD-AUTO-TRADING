"""CFTC Commitments of Traders (COT) positioning data for gold futures.

Free, public, no API key, no login, no scraping of anyone's account: the
CFTC publishes this every week as open government data via a public
Socrata API (publicreporting.cftc.gov). It's a legitimate, gold-specific
alternative to generic social-media sentiment -- net speculative
("non-commercial") positioning in gold futures, as a percentage of open
interest -- rather than another indicator computed off the same price
series everything else here already looks at.

Point-in-time correctness: the "as of" date on each report (a Tuesday) is
NOT when it becomes public. The CFTC releases it the following Friday,
roughly 3 days later. A backtest that used a report from the moment its
"as of" date passed, rather than from when it was actually published,
would be leaking information from the future. `available_at` on every
record adds that lag so lookups can never do that. The trailing percentile
is computed the same way -- expanding-window, using only each row and the
ones before it -- for the same reason: ranking a reading against its FULL
history (including rows that hadn't happened yet) would leak the future
into a number meant to describe the past.

This module only computes the signal. Whether extreme positioning actually
predicts anything for gold (the classic heuristic: extreme net-long
speculative positioning precedes pullbacks, extreme net-short precedes
rallies, because commercials/"smart money" are positioned the other way at
those points) is a hypothesis for research/walk_forward.py to test
honestly, not something assumed true here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

COT_API_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
GOLD_MARKET_NAME = "GOLD - COMMODITY EXCHANGE INC."
PUBLISH_LAG_DAYS = 3  # CFTC releases a Tuesday report the following Friday
MIN_PERIODS_FOR_PERCENTILE = 26  # ~half a year of weekly reports before ranking means anything


@dataclass
class CotReading:
    as_of: pd.Timestamp
    available_at: pd.Timestamp
    net_noncomm_pct_oi: float            # (noncomm long - noncomm short) / open interest, in percent
    percentile_in_history: float | None  # 0..100 trailing percentile; None if not enough history yet


def _trailing_percentile(series: pd.Series, min_periods: int = MIN_PERIODS_FOR_PERCENTILE) -> pd.Series:
    def pct_rank(window: pd.Series) -> float:
        return (window.iloc[-1] >= window).mean() * 100

    return series.expanding(min_periods=min_periods).apply(pct_rank, raw=False)


def fetch_cot_history(limit: int = 520, timeout: int = 20) -> pd.DataFrame:
    """Pulls up to `limit` most recent weekly gold COT reports (default
    520 ~= 10 years) and returns a DataFrame indexed by report `as_of`
    date, with `available_at`, `net_noncomm_pct_oi`, and the point-in-time
    trailing `percentile_in_history` already computed.
    """
    resp = requests.get(
        COT_API_URL,
        params={"market_and_exchange_names": GOLD_MARKET_NAME, "$limit": limit, "$order": "report_date_as_yyyy_mm_dd DESC"},
        timeout=timeout,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return pd.DataFrame(columns=["net_noncomm_pct_oi", "available_at", "percentile_in_history"])

    records = []
    for row in rows:
        open_interest = float(row["open_interest_all"])
        if open_interest <= 0:
            continue
        net = float(row["noncomm_positions_long_all"]) - float(row["noncomm_positions_short_all"])
        records.append({
            "as_of": pd.Timestamp(row["report_date_as_yyyy_mm_dd"]),
            "net_noncomm_pct_oi": 100 * net / open_interest,
        })

    df = pd.DataFrame(records).sort_values("as_of").set_index("as_of")
    df["available_at"] = df.index + pd.Timedelta(days=PUBLISH_LAG_DAYS)
    df["percentile_in_history"] = _trailing_percentile(df["net_noncomm_pct_oi"])
    return df


class PointInTimeCotLookup:
    """Same discipline as sentiment/history.py's lookup: returns the most
    recent COT reading whose `available_at` is at or before the given
    timestamp, and None if none qualifies -- never guesses, never looks
    at a report before it was actually public.
    """

    def __init__(self, history: pd.DataFrame | None = None):
        self.history = history if history is not None else fetch_cot_history()

    def reading_at(self, as_of_timestamp: pd.Timestamp) -> CotReading | None:
        if self.history.empty:
            return None
        # CFTC report dates are plain dates (tz-naive); candle timestamps from
        # OANDA/yfinance are tz-aware. Comparing the two directly raises in
        # pandas, so normalize to tz-naive here -- losing timezone precision
        # doesn't meaningfully matter against a multi-day publish lag anyway.
        as_of_timestamp = pd.Timestamp(as_of_timestamp)
        if as_of_timestamp.tzinfo is not None:
            as_of_timestamp = as_of_timestamp.tz_localize(None)
        available = self.history[self.history["available_at"] <= as_of_timestamp]
        if available.empty:
            return None
        row = available.iloc[-1]
        pctl = row["percentile_in_history"]
        return CotReading(
            as_of=available.index[-1],
            available_at=row["available_at"],
            net_noncomm_pct_oi=row["net_noncomm_pct_oi"],
            percentile_in_history=None if pd.isna(pctl) else float(pctl),
        )
