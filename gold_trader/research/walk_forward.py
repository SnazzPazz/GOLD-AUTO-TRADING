"""Walk-forward parameter research: the honest way to answer "does this
strategy have real edge, or did one backtest just get lucky on one slice
of history."

The rule is simple and non-negotiable: parameters are ever only chosen
using in-sample folds, and the final reported number comes exclusively
from a held-out fold that selection never saw. Anything else -- picking a
config, then testing it on the same data used to pick it -- isn't research,
it's curve-fitting with extra steps, and this module is built specifically
to make that mistake impossible to make by accident.

Because the strategy here isn't ML-fitted (no weights learned from data),
"walk-forward" mostly buys two things:
  1. A distribution of outcomes across many non-overlapping historical
     periods for a FIXED config, which answers "is performance consistent,
     or did it only work in one lucky/unlucky stretch."
  2. A small, bounded grid search over the config knobs that plausibly
     affect edge, selected on everything except the final fold, then
     judged only on that final fold.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import pandas as pd

from gold_trader.backtest.backtester import BacktestReport, run_backtest
from gold_trader.config import RiskConfig

MIN_TRADES_FOR_RANKING = 10  # a config that fired fewer trades than this in a fold is too noisy to rank on


@dataclass
class FoldResult:
    fold_index: int
    start: str
    end: str
    report: BacktestReport


def split_folds(df: pd.DataFrame, n_folds: int) -> list[pd.DataFrame]:
    """Splits df into n_folds contiguous, non-overlapping chunks covering
    the whole range, in time order. Each fold overlaps no other."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    fold_size = len(df) // n_folds
    if fold_size < 1:
        raise ValueError(f"not enough rows ({len(df)}) to make {n_folds} folds")
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(df)  # last fold absorbs the remainder
        folds.append(df.iloc[start:end])
    return folds


def evaluate_fold(df: pd.DataFrame, risk_config: RiskConfig, starting_balance: float, fold_index: int = 0) -> FoldResult:
    report = run_backtest(df, risk_config, starting_balance=starting_balance, use_recorded_sentiment=False)
    start = str(df.index[0]) if len(df) else ""
    end = str(df.index[-1]) if len(df) else ""
    return FoldResult(fold_index, start, end, report)


def _ranking_key(reports: list[BacktestReport]) -> tuple[float, float] | None:
    """Mean Sharpe-like ratio across folds, tie-broken by mean profit
    factor. Sharpe-like is used as the primary ranking signal (not total
    return) because it penalizes inconsistency across folds, which raw
    return doesn't -- a config that's wildly profitable in one fold and
    disastrous in another should rank below one that's modestly, reliably
    OK, and raw return alone can't tell those apart.

    Returns None if any fold didn't clear MIN_TRADES_FOR_RANKING trades --
    too few trades makes the fold's number mostly noise, and a config
    shouldn't win by accident of barely trading.
    """
    if any(r.num_trades < MIN_TRADES_FOR_RANKING for r in reports):
        return None
    mean_sharpe = sum(r.sharpe_like for r in reports) / len(reports)
    finite_pf = [r.profit_factor for r in reports if r.profit_factor != float("inf")]
    mean_pf = sum(finite_pf) / len(finite_pf) if finite_pf else 0.0
    return (mean_sharpe, mean_pf)


def grid_search_in_sample(
    in_sample_folds: list[pd.DataFrame],
    base_config: RiskConfig,
    param_grid: dict[str, list],
    starting_balance: float,
) -> tuple[RiskConfig, list[dict]]:
    """Evaluates every combination in param_grid across ALL in_sample_folds
    (never the held-out fold) and returns the best-ranked config plus a
    full leaderboard for transparency -- so a losing search is visible,
    not just the winner.
    """
    keys = list(param_grid.keys())
    leaderboard = []

    for values in itertools.product(*param_grid.values()):
        overrides = dict(zip(keys, values))
        candidate = replace(base_config, **overrides)
        reports = [run_backtest(fold, candidate, starting_balance=starting_balance, use_recorded_sentiment=False)
                   for fold in in_sample_folds]
        rank_key = _ranking_key(reports)
        leaderboard.append({
            "params": overrides,
            "rank_key": rank_key,
            "mean_return_pct": sum(r.total_return_pct for r in reports) / len(reports),
            "mean_trades": sum(r.num_trades for r in reports) / len(reports),
        })

    rankable = [entry for entry in leaderboard if entry["rank_key"] is not None]
    if not rankable:
        # Nothing cleared the minimum-trades bar anywhere in the grid; the
        # honest answer is "no config here was testable", not a fabricated pick.
        return base_config, leaderboard

    best = max(rankable, key=lambda entry: entry["rank_key"])
    best_config = replace(base_config, **best["params"])
    leaderboard.sort(key=lambda e: e["rank_key"] or (float("-inf"), float("-inf")), reverse=True)
    return best_config, leaderboard


@dataclass
class WalkForwardResult:
    default_config_folds: list[FoldResult]        # the shipped default config, evaluated on every fold
    selected_config: RiskConfig                    # the config chosen using only in-sample folds
    selected_config_in_sample_leaderboard: list[dict]
    held_out_fold: FoldResult                      # the ONLY fold used to judge selected_config
    held_out_default_fold: FoldResult              # default config on that same held-out fold, for comparison


def run_walk_forward_study(
    df: pd.DataFrame,
    n_folds: int,
    base_config: RiskConfig,
    param_grid: dict[str, list],
    starting_balance: float = 10_000.0,
) -> WalkForwardResult:
    if n_folds < 2:
        raise ValueError("need at least 2 folds: in-sample selection plus one held-out fold")

    folds = split_folds(df, n_folds)
    in_sample_folds, held_out = folds[:-1], folds[-1]

    default_results = [
        evaluate_fold(fold, base_config, starting_balance, i) for i, fold in enumerate(folds)
    ]

    selected_config, leaderboard = grid_search_in_sample(in_sample_folds, base_config, param_grid, starting_balance)

    held_out_selected = evaluate_fold(held_out, selected_config, starting_balance, n_folds - 1)
    held_out_default = evaluate_fold(held_out, base_config, starting_balance, n_folds - 1)

    return WalkForwardResult(
        default_config_folds=default_results,
        selected_config=selected_config,
        selected_config_in_sample_leaderboard=leaderboard,
        held_out_fold=held_out_selected,
        held_out_default_fold=held_out_default,
    )
