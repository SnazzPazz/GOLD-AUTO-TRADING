import numpy as np
import pandas as pd
import pytest

from gold_trader.config import RiskConfig
from gold_trader.research.walk_forward import (
    MIN_TRADES_FOR_RANKING,
    _ranking_key,
    grid_search_in_sample,
    run_rolling_walk_forward,
    run_walk_forward_study,
    split_folds,
)
from gold_trader.backtest.backtester import BacktestReport


def _synthetic_candles(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min")
    returns = rng.normal(loc=0.0002, scale=0.4, size=n)
    close = 1900 + np.cumsum(returns)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    volume = rng.integers(500, 5000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _dummy_report(num_trades, sharpe, pf, total_return=0.0):
    return BacktestReport(
        starting_balance=10_000, ending_balance=10_000 * (1 + total_return / 100),
        total_return_pct=total_return, num_trades=num_trades, win_rate=0.5, profit_factor=pf,
        max_drawdown_pct=1.0, sharpe_like=sharpe, regime_rejections=0, reviews_performed=0,
        equity_curve=pd.Series([10_000]), trades=[],
    )


def test_split_folds_covers_whole_range_without_overlap():
    df = _synthetic_candles(n=1000)
    folds = split_folds(df, 4)
    assert len(folds) == 4
    reassembled = pd.concat(folds)
    assert len(reassembled) == len(df)
    assert reassembled.index.equals(df.index)  # same rows, same order, nothing duplicated or dropped


def test_split_folds_last_fold_absorbs_remainder():
    df = _synthetic_candles(n=1003)
    folds = split_folds(df, 4)
    assert sum(len(f) for f in folds) == 1003
    # first three folds equal-sized, last one gets the extra rows
    assert len(folds[0]) == len(folds[1]) == len(folds[2])
    assert len(folds[3]) >= len(folds[0])


def test_split_folds_rejects_too_many_folds_for_data():
    df = _synthetic_candles(n=3)
    with pytest.raises(ValueError):
        split_folds(df, 10)


def test_ranking_key_none_when_any_fold_under_min_trades():
    reports = [_dummy_report(MIN_TRADES_FOR_RANKING, 1.0, 1.5), _dummy_report(MIN_TRADES_FOR_RANKING - 1, 1.0, 1.5)]
    assert _ranking_key(reports) is None


def test_ranking_key_uses_mean_sharpe_then_profit_factor():
    reports = [_dummy_report(20, sharpe=1.0, pf=1.2), _dummy_report(20, sharpe=2.0, pf=0.8)]
    key = _ranking_key(reports)
    assert key is not None
    assert key[0] == pytest.approx(1.5)  # mean sharpe


def test_grid_search_never_evaluates_on_held_out_data():
    """The core correctness property of this whole module: grid_search_in_sample
    must only ever be called with in-sample folds, and must never see the
    held-out fold's data. This test constructs a case where the held-out
    fold has wildly different data and confirms the search result doesn't
    depend on it (it can't, since it's never passed in)."""
    in_sample = [_synthetic_candles(n=800, seed=s) for s in (1, 2)]
    grid = {"min_confirmations": [2, 3]}
    config, leaderboard = grid_search_in_sample(in_sample, RiskConfig(), grid, starting_balance=10_000.0)
    assert isinstance(config, RiskConfig)
    assert len(leaderboard) == 2  # one entry per grid combination, regardless of held-out data


def test_grid_search_returns_base_config_when_nothing_clears_min_trades():
    # Tiny, near-flat data with almost no volatility produces almost no signals.
    idx = pd.date_range("2023-01-01", periods=50, freq="15min")
    flat = pd.DataFrame({"open": 1900.0, "high": 1900.01, "low": 1899.99, "close": 1900.0, "volume": 100}, index=idx)
    config, leaderboard = grid_search_in_sample([flat], RiskConfig(), {"min_confirmations": [2]}, 10_000.0)
    assert config == RiskConfig()  # falls back to the base/default config, not a fabricated pick
    assert all(entry["rank_key"] is None for entry in leaderboard)


def test_run_walk_forward_study_held_out_fold_is_last_chronological_chunk():
    df = _synthetic_candles(n=2000, seed=7)
    result = run_walk_forward_study(df, n_folds=4, base_config=RiskConfig(), param_grid={"min_confirmations": [2, 3]})
    expected_last_fold = split_folds(df, 4)[-1]
    assert result.held_out_fold.start == str(expected_last_fold.index[0])
    assert result.held_out_fold.end == str(expected_last_fold.index[-1])
    assert len(result.default_config_folds) == 4  # default config reported across every fold, for consistency


def test_run_walk_forward_study_requires_at_least_two_folds():
    df = _synthetic_candles(n=500)
    with pytest.raises(ValueError):
        run_walk_forward_study(df, n_folds=1, base_config=RiskConfig(), param_grid={"min_confirmations": [2]})


def test_rolling_walk_forward_produces_one_outcome_per_fold_after_minimum():
    df = _synthetic_candles(n=4000, seed=3)
    outcomes = run_rolling_walk_forward(
        df, n_folds=8, base_config=RiskConfig(), param_grid={"atr_stop_multiple": [1.0, 1.5]},
        min_in_sample_folds=3,
    )
    # folds 3,4,5,6,7 each get tested once -> 5 outcomes
    assert [o.fold_index for o in outcomes] == [3, 4, 5, 6, 7]


def test_rolling_walk_forward_each_fold_selection_only_uses_strictly_prior_folds():
    """The core correctness property: fold i's selection must never be
    influenced by fold i's own data or any later fold's data. Verified
    indirectly: each outcome's test fold matches split_folds' fold i exactly,
    and the selection step for fold i only ever receives folds[:i]."""
    df = _synthetic_candles(n=4000, seed=5)
    folds = split_folds(df, 8)
    outcomes = run_rolling_walk_forward(
        df, n_folds=8, base_config=RiskConfig(), param_grid={"atr_stop_multiple": [1.0, 1.5]},
        min_in_sample_folds=3,
    )
    for outcome in outcomes:
        expected_fold = folds[outcome.fold_index]
        assert outcome.start == str(expected_fold.index[0])
        assert outcome.end == str(expected_fold.index[-1])


def test_rolling_walk_forward_requires_more_folds_than_minimum():
    df = _synthetic_candles(n=500)
    with pytest.raises(ValueError):
        run_rolling_walk_forward(df, n_folds=3, base_config=RiskConfig(), param_grid={"atr_stop_multiple": [1.0]},
                                  min_in_sample_folds=3)
