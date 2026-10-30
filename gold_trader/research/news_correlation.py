"""Does real news/economic-event history actually correlate with gold's
subsequent price move? This asks that question directly, with a real
multi-decade event study, instead of assuming a sentiment/news filter
helps just because it sounds plausible.

Why Fed rate decisions, not generic headlines: a genuine "for every
headline, check whether price moved" study needs a free source of dated
historical news text going back years. The two candidates evaluated:

- GDELT's DOC 2.0 API (global news search with sentiment "tone"): only
  indexes English-language full-text search from 2017 onward (it
  explicitly rejects earlier start dates), and in practice rate-limits
  requests heavily -- even single, well-spaced test queries returned
  HTTP 429 ("limit requests to one every 5 seconds or contact us for
  larger queries") during evaluation for this module. Not usable for a
  systematic multi-year pull without a dedicated arrangement with GDELT.
- BLS's own CPI release-calendar page (the one place that publishes exact
  historical release dates, as opposed to the reference month FRED
  tags its data with) returns HTTP 403 to a plain fetch -- bot-blocked.

What *is* free, reliable, and exactly dated going back decades: FRED's
daily federal funds target rate series (DFEDTAR through 2008-12-15,
DFEDTARU from 2008-12-16 on -- the Fed switched from a single target to a
target range that month). Stitched together, this gives an exact date for
every single Fed rate change since 1982, with zero ambiguity about when
the "news" happened -- no release-date guessing required, unlike CPI/NFP.
This is also exactly the "FOMC/interest rate" category the existing (off
by default) sentiment filter's GOLD_KEYWORDS already singles out as
relevant, so it's testing a real, specific hypothesis already embedded in
this codebase, not an arbitrary one.

This module answers "is there a real signal here," not "let's add a
feature." If nothing clears a significance bar, that's the honest
result -- same as how COT positioning was tested and found to hurt (see
README). Nothing here is wired into the live signal engine; integrating
a finding into live trading is a separate, deliberate decision.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
# The Fed switched from a single target rate to a target range on this date;
# DFEDTAR stops and DFEDTARU begins right around here.
TARGET_RANGE_SWITCH_DATE = pd.Timestamp("2008-12-16")

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


def _fetch_fred_series(series_id: str, cache_dir: Path | None, cache_ttl_seconds: float | None) -> pd.Series:
    """Fetches a FRED series via the free, no-API-key fredgraph.csv endpoint
    (the official api.stlouisfed.org JSON API requires a registered API
    key; this CSV export doesn't). Caching follows the same opt-in,
    TTL-based pattern as YFinanceDataProvider -- disabled by default,
    since this is only ever called from research code, not anything
    live."""
    cache_path = (cache_dir / f"fred_{series_id}.csv") if cache_dir else None
    if cache_path is not None and cache_ttl_seconds is not None and cache_path.exists():
        import time
        if time.time() - cache_path.stat().st_mtime < cache_ttl_seconds:
            raw = cache_path.read_text()
        else:
            raw = None
    else:
        raw = None

    if raw is None:
        resp = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=15)
        resp.raise_for_status()
        raw = resp.text
        if cache_path is not None and cache_ttl_seconds is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(raw)

    df = pd.read_csv(io.StringIO(raw), parse_dates=["observation_date"])
    df = df.set_index("observation_date")
    series = df[series_id]
    series = series[series != "."]  # FRED's own missing-value marker
    return series.astype(float)


def fetch_fed_funds_target_rate(
    cache_dir: Path | None = _DEFAULT_CACHE_DIR, cache_ttl_seconds: float | None = 86_400.0
) -> pd.Series:
    """Returns one continuous daily series of the Fed's target rate back to
    1982, stitching DFEDTAR (single target, pre-2008-12-16) and DFEDTARU
    (target range upper bound, 2008-12-16 on) at the point the Fed itself
    switched frameworks. Using the upper bound post-switch is an arbitrary
    but consistent choice -- what matters for this study is detecting
    *when* the target changed, not its exact level.
    """
    pre = _fetch_fred_series("DFEDTAR", cache_dir, cache_ttl_seconds)
    post = _fetch_fred_series("DFEDTARU", cache_dir, cache_ttl_seconds)
    pre = pre[pre.index < TARGET_RANGE_SWITCH_DATE]
    post = post[post.index >= TARGET_RANGE_SWITCH_DATE]
    return pd.concat([pre, post]).sort_index()


@dataclass
class RateChangeEvent:
    event_date: pd.Timestamp
    delta: float          # percentage points; positive = hike, negative = cut
    direction: str        # "HIKE" or "CUT"


def find_rate_change_events(rate_series: pd.Series) -> list[RateChangeEvent]:
    """A 'change day' here means the Fed's target actually moved, not every
    FOMC meeting -- 'hold' decisions don't show up in this series at all,
    since nothing changes. That's a deliberate scope choice, not an
    oversight: a held rate is close to a non-event by definition, and this
    source has zero ambiguity about exactly when a real change happened,
    which a full meeting-by-meeting calendar (not freely available with
    reliable historical dates) couldn't offer without guessing."""
    diffs = rate_series.diff()
    events = []
    for date, delta in diffs.items():
        if pd.isna(delta) or delta == 0:
            continue
        events.append(RateChangeEvent(
            event_date=date, delta=float(delta), direction="HIKE" if delta > 0 else "CUT",
        ))
    return events


@dataclass
class EventReturn:
    event: RateChangeEvent
    anchor_trading_day: pd.Timestamp
    same_day_ret_pct: float
    next_day_ret_pct: float
    two_day_ret_pct: float


def compute_event_returns(events: list[RateChangeEvent], gold_close: pd.Series) -> list[EventReturn]:
    """For each event, finds the nearest trading day at/after the event date
    (FRED's rate series is dated by calendar day; gold only trades on
    trading days) and computes three point-in-time-correct windows:
    same-day (close[t] vs close[t-1], capturing a decision made during
    that day's session), next-day (close[t+1] vs close[t], capturing
    delayed digestion), and two-day (close[t+1] vs close[t-1], the combined
    effect). Events with no trading day on both sides (e.g. right at the
    edge of the available price history) are silently skipped -- not
    enough data to say anything about them.
    """
    trading_days = gold_close.index.sort_values()
    results = []
    for event in events:
        pos = trading_days.searchsorted(event.event_date)
        if pos >= len(trading_days):
            pos = len(trading_days) - 1
        if pos <= 0 or pos >= len(trading_days) - 1:
            continue
        anchor = trading_days[pos]
        close_prev = gold_close.loc[trading_days[pos - 1]]
        close_t = gold_close.loc[anchor]
        close_next = gold_close.loc[trading_days[pos + 1]]
        results.append(EventReturn(
            event=event,
            anchor_trading_day=anchor,
            same_day_ret_pct=(close_t - close_prev) / close_prev * 100,
            next_day_ret_pct=(close_next - close_t) / close_t * 100,
            two_day_ret_pct=(close_next - close_prev) / close_prev * 100,
        ))
    return results


@dataclass
class SignificanceCheck:
    label: str
    window: str
    n: int
    mean_pct: float
    std_pct: float
    baseline_mean_pct: float
    bootstrap_p_value: float | None  # None when n too small to test meaningfully

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.bootstrap_p_value is not None and self.bootstrap_p_value < alpha


def _bootstrap_two_sided_p(sample_mean: float, n: int, baseline: np.ndarray, rng: np.random.Generator,
                            n_boot: int = 10_000) -> float:
    """Resamples n-day random windows from the real baseline distribution of
    daily returns many times, and asks: how often does a random n-day
    sample have a mean at least as extreme as the one actually observed on
    event days? This makes no assumption that returns are normally
    distributed (they aren't -- real markets have fat tails), unlike a
    classic t-test, which is why it's used here instead of/alongside one.
    """
    boot_means = rng.choice(baseline, size=(n_boot, n), replace=True).mean(axis=1)
    return float((np.abs(boot_means) >= abs(sample_mean)).mean())


MIN_SAMPLE_FOR_SIGNIFICANCE_TEST = 3


def check_significance(
    event_returns: list[EventReturn], gold_close: pd.Series, label: str, window: str, seed: int = 0,
) -> SignificanceCheck:
    """window must be one of 'same_day_ret_pct', 'next_day_ret_pct', 'two_day_ret_pct'."""
    vals = np.array([getattr(e, window) for e in event_returns])
    baseline = gold_close.pct_change().dropna().values * 100
    if len(vals) < MIN_SAMPLE_FOR_SIGNIFICANCE_TEST:
        return SignificanceCheck(label, window, len(vals), float(vals.mean()) if len(vals) else float("nan"),
                                  float(vals.std()) if len(vals) else float("nan"), float(baseline.mean()), None)
    rng = np.random.default_rng(seed)
    p_value = _bootstrap_two_sided_p(float(vals.mean()), len(vals), baseline, rng)
    return SignificanceCheck(label, window, len(vals), float(vals.mean()), float(vals.std()),
                              float(baseline.mean()), p_value)


@dataclass
class NewsCorrelationResult:
    events: list[EventReturn]
    checks: list[SignificanceCheck]
    gold_history_start: pd.Timestamp
    gold_history_end: pd.Timestamp
    volatility_checks: list[SignificanceCheck] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Fed rate-change event study: {len(self.events)} events "
            f"within gold price history {self.gold_history_start.date()} -> {self.gold_history_end.date()}",
            "",
            "Directional return vs baseline (bootstrap p-value; p < 0.05 would be a real signal):",
        ]
        for c in self.checks:
            flag = "SIGNIFICANT" if c.is_significant() else "not significant"
            p_str = f"{c.bootstrap_p_value:.4f}" if c.bootstrap_p_value is not None else "n/a (n too small)"
            lines.append(
                f"  {c.label:12s} {c.window:16s} n={c.n:3d}  mean={c.mean_pct:+.4f}%  "
                f"std={c.std_pct:.4f}%  p={p_str}  [{flag}]"
            )
        if self.volatility_checks:
            lines.append("")
            lines.append("Volatility (|return|) vs baseline -- a real effect here means bigger moves,")
            lines.append("not a predictable direction:")
            for c in self.volatility_checks:
                flag = "SIGNIFICANT" if c.is_significant() else "not significant"
                p_str = f"{c.bootstrap_p_value:.4f}" if c.bootstrap_p_value is not None else "n/a"
                lines.append(
                    f"  {c.label:12s} {c.window:16s} n={c.n:3d}  mean|ret|={c.mean_pct:.4f}%  "
                    f"(baseline {c.baseline_mean_pct:.4f}%)  p={p_str}  [{flag}]"
                )
        return "\n".join(lines)


def run_news_correlation_study(
    gold_close: pd.Series,
    cache_dir: Path | None = _DEFAULT_CACHE_DIR,
    cache_ttl_seconds: float | None = 86_400.0,
    seed: int = 0,
) -> NewsCorrelationResult:
    """The entry point: fetches Fed rate history, finds every rate-change
    event inside gold_close's date range, and tests both direction and
    volatility against a real baseline. gold_close must be a datetime-
    indexed Series of daily closes (e.g. YFinanceDataProvider's 'close'
    column)."""
    # yfinance returns a tz-aware index (the exchange's local time); FRED's dates
    # are plain calendar days with no timezone of their own. Only the calendar
    # date matters for a daily event study, so drop tz-awareness rather than try
    # to reconcile two timezones that don't actually mean anything to compare here.
    if gold_close.index.tz is not None:
        gold_close = gold_close.tz_localize(None)
    rate_series = fetch_fed_funds_target_rate(cache_dir, cache_ttl_seconds)
    gold_start, gold_end = gold_close.index.min(), gold_close.index.max()
    events = find_rate_change_events(rate_series)
    events = [e for e in events if gold_start <= e.event_date <= gold_end]
    event_returns = compute_event_returns(events, gold_close)

    windows = ["same_day_ret_pct", "next_day_ret_pct", "two_day_ret_pct"]
    checks = []
    for label, subset in [
        ("ALL", event_returns),
        ("HIKES", [e for e in event_returns if e.event.direction == "HIKE"]),
        ("CUTS", [e for e in event_returns if e.event.direction == "CUT"]),
    ]:
        for window in windows:
            checks.append(check_significance(subset, gold_close, label, window, seed=seed))

    baseline_abs = gold_close.pct_change().dropna().abs().values * 100
    vol_checks = []
    for window in ["same_day_ret_pct", "next_day_ret_pct"]:
        vals = np.abs(np.array([getattr(e, window) for e in event_returns]))
        rng = np.random.default_rng(seed)
        p_value = _bootstrap_two_sided_p(float(vals.mean()), len(vals), baseline_abs, rng) if len(vals) >= 3 else None
        vol_checks.append(SignificanceCheck("ALL", window, len(vals), float(vals.mean()), float(vals.std()),
                                             float(baseline_abs.mean()), p_value))

    return NewsCorrelationResult(
        events=event_returns, checks=checks, gold_history_start=gold_start, gold_history_end=gold_end,
        volatility_checks=vol_checks,
    )
