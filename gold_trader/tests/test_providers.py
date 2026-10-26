import pandas as pd
import pytest

from gold_trader.data.providers import YFinanceDataProvider


def _fake_raw_df(seed_value: float, n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": seed_value, "high": seed_value + 1, "low": seed_value - 1,
         "close": seed_value, "volume": 100},
        index=idx,
    )
    df.index.name = "time"  # matches what the real _fetch_raw sets before caching
    return df


def test_caching_disabled_by_default_fetches_every_call(tmp_path, monkeypatch):
    provider = YFinanceDataProvider(cache_dir=tmp_path, cache_ttl_seconds=None)
    calls = []

    def fake_fetch(interval, period):
        calls.append(1)
        return _fake_raw_df(len(calls))

    monkeypatch.setattr(provider, "_fetch_raw", fake_fetch)
    provider.get_historical_candles("H1", 10)
    provider.get_historical_candles("H1", 10)
    assert len(calls) == 2  # no caching: hits the fetch every time


def test_second_call_within_ttl_reuses_cached_data_not_a_fresh_fetch(tmp_path, monkeypatch):
    provider = YFinanceDataProvider(cache_dir=tmp_path, cache_ttl_seconds=3600)
    calls = []

    def fake_fetch(interval, period):
        calls.append(1)
        return _fake_raw_df(seed_value=42.0)

    monkeypatch.setattr(provider, "_fetch_raw", fake_fetch)
    first = provider.get_historical_candles("H1", 10)
    second = provider.get_historical_candles("H1", 10)

    assert len(calls) == 1  # second call served from cache, not re-fetched
    # check_freq=False: pd.date_range's synthetic index carries freq metadata that a
    # CSV round-trip legitimately drops; real yfinance data never has it either.
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_cache_expires_after_ttl_and_refetches(tmp_path, monkeypatch):
    provider = YFinanceDataProvider(cache_dir=tmp_path, cache_ttl_seconds=3600)
    calls = []

    def fake_fetch(interval, period):
        calls.append(1)
        return _fake_raw_df(seed_value=float(len(calls)))

    monkeypatch.setattr(provider, "_fetch_raw", fake_fetch)
    provider.get_historical_candles("H1", 10)

    cache_path = provider._cache_path(*provider._GRANULARITY_TO_YF["H1"])
    # Back-date the cache file's mtime past the TTL to simulate it having gone stale.
    import os
    import time
    stale_time = time.time() - 7200
    os.utime(cache_path, (stale_time, stale_time))

    provider.get_historical_candles("H1", 10)
    assert len(calls) == 2  # stale cache was not reused


def test_different_tickers_use_separate_cache_files(tmp_path, monkeypatch):
    gold = YFinanceDataProvider(ticker="GC=F", cache_dir=tmp_path, cache_ttl_seconds=3600)
    other = YFinanceDataProvider(ticker="SI=F", cache_dir=tmp_path, cache_ttl_seconds=3600)

    monkeypatch.setattr(gold, "_fetch_raw", lambda i, p: _fake_raw_df(1.0))
    monkeypatch.setattr(other, "_fetch_raw", lambda i, p: _fake_raw_df(2.0))

    gold.get_historical_candles("H1", 10)
    other.get_historical_candles("H1", 10)

    assert gold._cache_path("60m", "730d") != other._cache_path("60m", "730d")
    assert gold._cache_path("60m", "730d").exists()
    assert other._cache_path("60m", "730d").exists()


def test_cached_round_trip_preserves_tz_aware_datetime_index(tmp_path, monkeypatch):
    """Regression guard for the known pd.read_csv(parse_dates=True) pitfall where a
    mixed-offset datetime column silently stays dtype=object instead of becoming a
    real DatetimeIndex (hit once already in an ad-hoc scratch script this session)."""
    provider = YFinanceDataProvider(cache_dir=tmp_path, cache_ttl_seconds=3600)
    monkeypatch.setattr(provider, "_fetch_raw", lambda i, p: _fake_raw_df(1.0))

    provider.get_historical_candles("H1", 10)
    loaded = provider.get_historical_candles("H1", 10)  # served from cache
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.tz is not None
