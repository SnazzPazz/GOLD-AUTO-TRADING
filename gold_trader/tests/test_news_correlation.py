import numpy as np
import pandas as pd
import pytest

from gold_trader.research import news_correlation as nc


def _gold_close(n=200, seed=0, start="2010-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B")  # business days, like real trading days
    returns = rng.normal(0, 0.01, n)
    close = 1000 * np.cumprod(1 + returns)
    return pd.Series(close, index=idx)


def test_find_rate_change_events_detects_hikes_and_cuts_only_on_real_moves():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    rate = pd.Series([1.0, 1.0, 1.25, 1.25, 1.00, 1.00], index=idx)
    events = nc.find_rate_change_events(rate)
    assert len(events) == 2
    assert events[0].direction == "HIKE"
    assert events[0].delta == pytest.approx(0.25)
    assert events[0].event_date == idx[2]
    assert events[1].direction == "CUT"
    assert events[1].delta == pytest.approx(-0.25)


def test_find_rate_change_events_empty_for_flat_series():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    rate = pd.Series([2.0] * 5, index=idx)
    assert nc.find_rate_change_events(rate) == []


def test_compute_event_returns_is_point_in_time_correct():
    """same_day_ret must compare the event's trading day against the PRIOR
    trading day's close -- never leak the event day's own close into its
    own 'before' baseline."""
    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=idx)
    event = nc.RateChangeEvent(event_date=idx[5], delta=0.25, direction="HIKE")

    results = nc.compute_event_returns([event], close)
    assert len(results) == 1
    r = results[0]
    expected_same_day = (close.iloc[5] - close.iloc[4]) / close.iloc[4] * 100
    expected_next_day = (close.iloc[6] - close.iloc[5]) / close.iloc[5] * 100
    expected_two_day = (close.iloc[6] - close.iloc[4]) / close.iloc[4] * 100
    assert r.same_day_ret_pct == pytest.approx(expected_same_day)
    assert r.next_day_ret_pct == pytest.approx(expected_next_day)
    assert r.two_day_ret_pct == pytest.approx(expected_two_day)


def test_compute_event_returns_skips_events_at_the_edge_of_price_history():
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    close = pd.Series([100, 101, 102, 103, 104], index=idx)
    first_day_event = nc.RateChangeEvent(event_date=idx[0], delta=0.25, direction="HIKE")
    last_day_event = nc.RateChangeEvent(event_date=idx[-1], delta=0.25, direction="HIKE")
    middle_event = nc.RateChangeEvent(event_date=idx[2], delta=0.25, direction="HIKE")

    results = nc.compute_event_returns([first_day_event, last_day_event, middle_event], close)
    assert len(results) == 1
    assert results[0].anchor_trading_day == idx[2]


def test_compute_event_returns_maps_event_to_nearest_later_trading_day():
    """A rate-change date that falls on a weekend/holiday (no trading that
    day) must anchor to the next real trading day, not silently vanish or
    misalign. Needs trading days on both sides of the gap to produce a
    result at all (compute_event_returns correctly skips edge-of-history
    events, which a too-short window would hit instead of testing this)."""
    idx = pd.date_range("2021-01-04", periods=10, freq="B")  # two full Mon-Fri weeks
    close = pd.Series(range(100, 110), index=idx, dtype=float)
    saturday_event = nc.RateChangeEvent(event_date=pd.Timestamp("2021-01-09"), delta=0.25, direction="HIKE")

    results = nc.compute_event_returns([saturday_event], close)
    assert len(results) == 1
    assert results[0].anchor_trading_day == idx[5]  # Monday 2021-01-11, the next real trading day


def test_check_significance_returns_none_p_value_when_sample_too_small():
    close = _gold_close()
    event = nc.RateChangeEvent(event_date=close.index[50], delta=0.25, direction="HIKE")
    event_returns = nc.compute_event_returns([event], close)
    check = nc.check_significance(event_returns, close, "HIKES", "same_day_ret_pct")
    assert check.n == 1
    assert check.bootstrap_p_value is None
    assert check.is_significant() is False


def test_check_significance_finds_an_obviously_real_effect():
    """Sanity check for the bootstrap test itself: if 'events' always land
    on days with a huge, consistent return far outside the baseline's
    range, the test must detect that as significant. Otherwise the
    significance check itself can't be trusted for the real study."""
    close = _gold_close(n=300, seed=1)
    # Inject a consistent, large +5% move on 20 specific days -- impossible
    # to confuse with normal ~1%-std noise.
    event_days = close.index[::10][:20]
    close = close.copy()
    for d in event_days:
        pos = close.index.get_loc(d)
        if 0 < pos < len(close) - 1:
            close.iloc[pos] = close.iloc[pos - 1] * 1.05

    events = [nc.RateChangeEvent(event_date=d, delta=0.25, direction="HIKE") for d in event_days]
    event_returns = nc.compute_event_returns(events, close)
    check = nc.check_significance(event_returns, close, "HIKES", "same_day_ret_pct")
    assert check.n >= 15
    assert check.mean_pct > 3.0
    assert check.bootstrap_p_value < 0.01
    assert check.is_significant() is True


def test_check_significance_does_not_flag_pure_noise_as_significant():
    """The flip side: real daily noise around rate-change days (no actual
    injected effect) should NOT come back significant most of the time --
    this guards against the bootstrap test itself being miscalibrated and
    flagging everything."""
    close = _gold_close(n=500, seed=7)
    event_days = close.index[::15][:25]
    events = [nc.RateChangeEvent(event_date=d, delta=0.25, direction="HIKE") for d in event_days]
    event_returns = nc.compute_event_returns(events, close)
    check = nc.check_significance(event_returns, close, "HIKES", "same_day_ret_pct", seed=3)
    assert check.bootstrap_p_value > 0.05  # no real effect was injected here


def test_run_news_correlation_study_end_to_end_with_mocked_fetch(monkeypatch):
    close = _gold_close(n=400, seed=2, start="2015-01-01")

    def fake_fetch(series_id, cache_dir, cache_ttl_seconds):
        idx = pd.date_range("2010-01-01", "2020-01-01", freq="D")
        # A rate that steps up by 0.25 every ~90 days -- gives real, spaced-out events.
        steps = (np.arange(len(idx)) // 90) * 0.25
        return pd.Series(steps, index=idx, name=series_id)

    monkeypatch.setattr(nc, "_fetch_fred_series", fake_fetch)

    result = nc.run_news_correlation_study(close, cache_dir=None, cache_ttl_seconds=None)
    assert len(result.events) > 0
    assert result.gold_history_start == close.index.min()
    assert result.gold_history_end == close.index.max()
    # ALL/HIKES/CUTS x 3 windows = 9 directional checks (CUTS may be empty-sample but still present)
    assert len(result.checks) == 9
    assert len(result.volatility_checks) == 2
    summary = result.summary()
    assert "Fed rate-change event study" in summary
    assert "SIGNIFICANT" in summary or "not significant" in summary


def test_run_news_correlation_study_handles_tz_aware_gold_index(monkeypatch):
    """Regression test: yfinance returns a tz-aware DatetimeIndex (the
    exchange's local time), while FRED's dates are plain tz-naive calendar
    days. Comparing them directly raises TypeError -- caught by actually
    running this command against real yfinance data, not just the
    synthetic tz-naive fixtures the other tests use."""
    close = _gold_close(n=400, seed=2, start="2015-01-01")
    close.index = close.index.tz_localize("America/New_York")

    def fake_fetch(series_id, cache_dir, cache_ttl_seconds):
        idx = pd.date_range("2010-01-01", "2020-01-01", freq="D")  # tz-naive, like real FRED data
        steps = (np.arange(len(idx)) // 90) * 0.25
        return pd.Series(steps, index=idx, name=series_id)

    monkeypatch.setattr(nc, "_fetch_fred_series", fake_fetch)
    result = nc.run_news_correlation_study(close, cache_dir=None, cache_ttl_seconds=None)
    assert len(result.events) > 0
    assert result.gold_history_start.tz is None


def test_fetch_fed_funds_target_rate_stitches_at_the_switch_date(monkeypatch):
    def fake_fetch(series_id, cache_dir, cache_ttl_seconds):
        if series_id == "DFEDTAR":
            idx = pd.date_range("2008-01-01", "2008-12-20", freq="D")
            return pd.Series(1.0, index=idx, name=series_id)
        else:
            idx = pd.date_range("2008-12-01", "2009-01-01", freq="D")
            return pd.Series(0.25, index=idx, name=series_id)

    monkeypatch.setattr(nc, "_fetch_fred_series", fake_fetch)
    combined = nc.fetch_fed_funds_target_rate(cache_dir=None, cache_ttl_seconds=None)

    before_switch = combined[combined.index < nc.TARGET_RANGE_SWITCH_DATE]
    after_switch = combined[combined.index >= nc.TARGET_RANGE_SWITCH_DATE]
    assert (before_switch == 1.0).all()
    assert (after_switch == 0.25).all()
    assert combined.index.is_monotonic_increasing
