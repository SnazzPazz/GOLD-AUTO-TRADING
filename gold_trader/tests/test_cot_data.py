import pandas as pd

from gold_trader.sentiment.cot_data import PointInTimeCotLookup, _trailing_percentile


def _history(rows):
    """rows: list of (as_of_str, net_pct) tuples, ascending by date."""
    df = pd.DataFrame(
        {"net_noncomm_pct_oi": [v for _, v in rows]},
        index=pd.to_datetime([d for d, _ in rows]),
    )
    df.index.name = "as_of"
    df["available_at"] = df.index + pd.Timedelta(days=3)
    df["percentile_in_history"] = _trailing_percentile(df["net_noncomm_pct_oi"], min_periods=2)
    return df


def test_publish_lag_means_as_of_date_alone_is_not_enough():
    history = _history([("2024-01-02", 10.0)])  # a Tuesday
    lookup = PointInTimeCotLookup(history)
    # The as-of date has passed, but the report isn't public for 3 more days.
    assert lookup.reading_at(pd.Timestamp("2024-01-02")) is None
    assert lookup.reading_at(pd.Timestamp("2024-01-04")) is None
    assert lookup.reading_at(pd.Timestamp("2024-01-05")) is not None  # available_at


def test_lookup_never_leaks_a_future_report():
    history = _history([("2024-01-02", 10.0), ("2024-01-09", 50.0), ("2024-01-16", -30.0)])
    lookup = PointInTimeCotLookup(history)
    # Querying right after the first report becomes available must not see the later ones.
    reading = lookup.reading_at(pd.Timestamp("2024-01-05"))
    assert reading is not None
    assert reading.net_noncomm_pct_oi == 10.0


def test_lookup_returns_most_recent_available_report():
    history = _history([("2024-01-02", 10.0), ("2024-01-09", 50.0), ("2024-01-16", -30.0)])
    lookup = PointInTimeCotLookup(history)
    reading = lookup.reading_at(pd.Timestamp("2024-01-20"))
    assert reading.net_noncomm_pct_oi == -30.0


def test_trailing_percentile_ignores_future_rows():
    """The whole point of using an expanding window: an early row's percentile
    must not be influenced by a later, more extreme row that hadn't happened yet."""
    series = pd.Series([10.0, 10.0, 10.0, 99.0])  # last value is an extreme outlier
    pctl = _trailing_percentile(series, min_periods=2)
    # The third row's percentile is computed only from rows 0-2 (all equal to 10),
    # so it must rank at the top (100), unaffected by the outlier that comes after it.
    assert pctl.iloc[2] == 100.0


def test_trailing_percentile_none_before_min_periods():
    history = _history([("2024-01-02", 10.0)])
    lookup = PointInTimeCotLookup(history)
    reading = lookup.reading_at(pd.Timestamp("2024-01-10"))
    assert reading is not None
    assert reading.percentile_in_history is None  # only one data point, below min_periods=2


def test_empty_history_returns_none():
    lookup = PointInTimeCotLookup(history=pd.DataFrame(columns=["net_noncomm_pct_oi", "available_at", "percentile_in_history"]))
    assert lookup.reading_at(pd.Timestamp("2024-01-01")) is None
