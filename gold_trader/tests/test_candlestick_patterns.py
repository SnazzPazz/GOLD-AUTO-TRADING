import pandas as pd

from gold_trader.analysis import candlestick_patterns as cp


def _row(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1000}


def test_hammer_detected():
    df = pd.DataFrame([
        _row(110, 112, 90, 109),     # prior bar, irrelevant
        _row(100, 100.3, 90, 99.5),  # small body near top, long lower wick, tiny upper wick -> hammer
    ])
    result = cp.hammer(df)
    assert bool(result.iloc[-1]) is True


def test_shooting_star_detected():
    df = pd.DataFrame([
        _row(90, 110, 89, 91),     # prior bar
        _row(100, 110, 99.5, 100.5),  # small body near bottom, long upper wick
    ])
    result = cp.shooting_star(df)
    assert bool(result.iloc[-1]) is True


def test_bullish_engulfing_detected():
    df = pd.DataFrame([
        _row(105, 106, 98, 99),    # prior bearish bar: open 105 -> close 99
        _row(98, 110, 97, 106),    # bullish bar engulfing prior body
    ])
    result = cp.bullish_engulfing(df)
    assert bool(result.iloc[-1]) is True


def test_bearish_engulfing_detected():
    df = pd.DataFrame([
        _row(99, 106, 98, 105),    # prior bullish bar: open 99 -> close 105
        _row(106, 107, 95, 98),    # bearish bar engulfing prior body
    ])
    result = cp.bearish_engulfing(df)
    assert bool(result.iloc[-1]) is True


def test_doji_detected_on_tiny_body():
    df = pd.DataFrame([_row(100, 105, 95, 100.1)])
    result = cp.doji(df)
    assert bool(result.iloc[-1]) is True


def test_detect_all_returns_expected_columns():
    df = pd.DataFrame([_row(100, 101, 99, 100.5)] * 5)
    out = cp.detect_all(df)
    for col in ["doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing",
                "morning_star", "evening_star", "piercing_line", "dark_cloud_cover"]:
        assert col in out.columns
        assert out[col].dtype == bool
