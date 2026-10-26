import numpy as np
import pandas as pd
import pytest

from gold_trader.config import RiskConfig
from gold_trader.research.stress_test import (
    block_bootstrap_resample,
    generate_shock_scenario,
    run_stress_test,
)


def _real_like_candles(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    returns = rng.normal(loc=0.0003, scale=0.004, size=n)
    close = 2000 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, close * 0.001))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, close * 0.001))
    volume = rng.integers(500, 5000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _assert_valid_ohlcv(df: pd.DataFrame):
    assert not df[["open", "high", "low", "close"]].isna().any().any()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["high"] >= df["low"]).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_block_bootstrap_produces_valid_ohlcv_same_length():
    candles = _real_like_candles(n=1000)
    rng = np.random.default_rng(42)
    resampled = block_bootstrap_resample(candles, block_size=50, rng=rng)
    assert len(resampled) == len(candles) - 1  # first row's prior-close is undefined, dropped
    _assert_valid_ohlcv(resampled)


def test_block_bootstrap_is_deterministic_given_same_seed():
    candles = _real_like_candles(n=1000)
    a = block_bootstrap_resample(candles, block_size=50, rng=np.random.default_rng(7))
    b = block_bootstrap_resample(candles, block_size=50, rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_block_bootstrap_differs_from_original_sequence():
    """The whole point: a shuffled path should not just reproduce the
    original bar sequence (astronomically unlikely by chance, so a
    collision here would indicate the shuffle isn't actually happening)."""
    candles = _real_like_candles(n=2000)
    resampled = block_bootstrap_resample(candles, block_size=50, rng=np.random.default_rng(1))
    assert not np.allclose(resampled["close"].values, candles["close"].iloc[1:].values)


def test_block_bootstrap_rejects_block_size_larger_than_history():
    candles = _real_like_candles(n=50)
    with pytest.raises(ValueError):
        block_bootstrap_resample(candles, block_size=100, rng=np.random.default_rng(0))


@pytest.mark.parametrize("kind", ["sustained_crash", "volatility_spike", "no_edge_chop"])
def test_shock_scenario_produces_valid_ohlcv(kind):
    scenario = generate_shock_scenario(kind, n_bars=500, start_price=2000.0, rng=np.random.default_rng(3))
    assert len(scenario) == 500
    _assert_valid_ohlcv(scenario)


def test_sustained_crash_scenario_actually_declines():
    scenario = generate_shock_scenario("sustained_crash", n_bars=1000, start_price=2000.0, rng=np.random.default_rng(5))
    assert scenario["close"].iloc[-1] < scenario["close"].iloc[0] * 0.85  # meaningfully down, not just noise


def test_no_edge_chop_has_no_strong_directional_drift():
    """A single zero-drift random walk can still wander 20-30%+ over 2000
    steps -- that's the nature of a random walk, not evidence of hidden
    bias. What actually proves "no deliberate drift" is that the move
    averages out to ~0 across many independent realizations (law of large
    numbers), not that any one path stays small."""
    moves = []
    for seed in range(30):
        scenario = generate_shock_scenario("no_edge_chop", n_bars=500, start_price=2000.0, rng=np.random.default_rng(seed))
        moves.append(scenario["close"].iloc[-1] / scenario["close"].iloc[0] - 1)
    assert abs(np.mean(moves)) < 0.05  # averaged over many paths, net drift should be near zero


def test_run_stress_test_returns_results_for_every_path():
    candles = _real_like_candles(n=2000)
    result = run_stress_test(
        candles, RiskConfig(), n_bootstrap_paths=3, block_size=100,
        review_interval_days=30, shock_bars=400, seed=0,
    )
    assert len(result.bootstrap_results) == 3
    assert len(result.shock_results) == 3
    assert {r.path_id for r in result.shock_results} == {"sustained_crash", "volatility_spike", "no_edge_chop"}
    summary = result.summary()
    assert "Paths tested: 3 bootstrap + 3 shock" in summary
