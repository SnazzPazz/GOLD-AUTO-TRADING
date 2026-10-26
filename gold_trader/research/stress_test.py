"""Stress-testing against synthetic adverse paths.

The gap this fills: every backtest and walk-forward study in this project
so far has run against the one real price history that actually happened.
That's fine for asking "does the signal have edge" (walk_forward.py
answers that honestly), but it's the wrong tool for judging a circuit
breaker like MAX_DRAWDOWN_PCT -- a breaker's entire job is to limit the
damage on paths that *don't* recover, and a backtest over history that
already happened can only ever contain paths that, in hindsight, either
recovered or didn't. Loosening a risk control because it would have
helped on the one path we can see is survivorship bias, not evidence.

Two independent ways to generate paths that didn't actually happen:

1. Block-bootstrap resampling: cut the real historical bars into
   contiguous blocks (preserving each block's local volatility clustering
   and candle shapes), then reassemble the blocks in a random order. This
   draws new paths from the *same* statistical distribution as real
   history, just a different sequence -- it can't produce a worse tail
   event than what's already in the data, but it does break the specific
   "this drawdown happened to recover" narrative by shuffling when things
   happen relative to each other.

2. Injected shock scenarios: purpose-built adverse conditions (a sustained
   crash, a volatility spike, a pure no-edge random walk) that may not
   exist anywhere in the real sample at all -- gold's real 2.4-year window
   here was broadly bullish, so a genuine bear regime has to be
   constructed on purpose to be tested at all.

The right question to ask of the results is never "which setting gives
the best average return" -- that's the same curve-fitting mistake dressed
up as a risk decision. It's "which setting keeps the worst-case outcome
survivable," since that's what a circuit breaker is for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_trader.backtest.backtester import BacktestReport, run_backtest
from gold_trader.config import RiskConfig


def _relative_bar_returns(candles: pd.DataFrame) -> pd.DataFrame:
    """Expresses every bar's O/H/L/C as a return relative to the PRIOR bar's
    close -- this is what gets block-bootstrapped, not raw price levels.
    Shuffling price levels directly would create nonsense discontinuities
    (gold went from ~$2366 to ~$4030 over this sample; splicing a block from
    the $2400 era after one from the $3800 era mid-series would corrupt
    every indicator). Shuffling *returns* and recompounding from a fixed
    starting price keeps every synthetic path internally consistent.
    """
    prev_close = candles["close"].shift(1)
    out = pd.DataFrame({
        "open_ret": candles["open"] / prev_close - 1,
        "high_ret": candles["high"] / prev_close - 1,
        "low_ret": candles["low"] / prev_close - 1,
        "close_ret": candles["close"] / prev_close - 1,
        "volume": candles["volume"],
    })
    return out.iloc[1:]  # first row's prev_close is undefined


def _reconstruct_from_returns(returns: pd.DataFrame, start_price: float, start_time: pd.Timestamp, freq: str) -> pd.DataFrame:
    """Inverse of _relative_bar_returns: compounds a sequence of relative
    bar-returns forward from a fixed starting price into an absolute OHLCV
    series. The high/low/open ordering within each bar is preserved exactly
    because it's inherited unchanged from whichever real bar each return
    tuple came from -- only the base price being compounded against changes.
    """
    close = start_price
    rows = []
    for _, r in returns.iterrows():
        prev_close = close
        open_ = prev_close * (1 + r["open_ret"])
        high = prev_close * (1 + r["high_ret"])
        low = prev_close * (1 + r["low_ret"])
        close = prev_close * (1 + r["close_ret"])
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": r["volume"]})
    index = pd.date_range(start=start_time, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, index=index)


def block_bootstrap_resample(
    candles: pd.DataFrame, block_size: int, rng: np.random.Generator, freq: str = "h",
) -> pd.DataFrame:
    """One synthetic path: real historical bars cut into contiguous blocks
    of `block_size` bars, blocks drawn with replacement and concatenated
    (same total length as the input, same per-block internal structure,
    different order) into a new relative-return sequence, then recompounded
    into an absolute OHLCV series starting at the real data's first close.
    """
    returns = _relative_bar_returns(candles)
    n_blocks_needed = -(-len(returns) // block_size)  # ceil division
    max_start = len(returns) - block_size
    if max_start < 0:
        raise ValueError(f"block_size ({block_size}) is larger than available history ({len(returns)} bars)")

    starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
    blocks = [returns.iloc[s:s + block_size] for s in starts]
    shuffled = pd.concat(blocks, ignore_index=True).iloc[:len(returns)]

    return _reconstruct_from_returns(shuffled, start_price=float(candles["close"].iloc[0]), start_time=candles.index[0], freq=freq)


def generate_shock_scenario(
    kind: str, n_bars: int, start_price: float, rng: np.random.Generator,
    start_time: pd.Timestamp | None = None, freq: str = "h",
) -> pd.DataFrame:
    """Purpose-built adverse scenarios that may not exist in any real
    sample available to this project. `kind`:
      - "sustained_crash": a steady, elevated-volatility decline of roughly
        30% over the full window -- a genuine bear regime, which gold's
        real 2024-2026 bull run never gave us a chance to test against.
      - "volatility_spike": calm, then a short burst of 5x-normal-ATR
        whipsaw with no directional bias -- stress on the regime filter
        and stop placement, not on trend-following.
      - "no_edge_chop": pure random walk, zero drift -- the honest
        worst-case for a strategy whose edge, if any, is modest: does the
        risk management limit the damage when there's nothing to find.
    """
    start_time = start_time or pd.Timestamp("2000-01-01")
    idx = pd.date_range(start=start_time, periods=n_bars, freq=freq)

    if kind == "sustained_crash":
        drift = np.log(0.70) / n_bars  # ~30% cumulative decline
        vol = 0.012
    elif kind == "volatility_spike":
        drift = 0.0
        calm, spike = n_bars // 3, n_bars // 3
        vol = np.concatenate([
            np.full(calm, 0.004), np.full(spike, 0.004 * 5), np.full(n_bars - calm - spike, 0.004),
        ])
    elif kind == "no_edge_chop":
        drift = 0.0
        vol = 0.006
    else:
        raise ValueError(f"unknown shock scenario kind: {kind!r}")

    log_returns = rng.normal(loc=drift, scale=vol, size=n_bars)
    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    wick = np.abs(rng.normal(0, close * 0.002))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.integers(500, 5000, n_bars)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@dataclass
class PathResult:
    path_id: str
    report: BacktestReport
    worst_drawdown_pct: float  # max observed drawdown at any point, regardless of recovery


def _worst_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (running_max - equity_curve) / running_max.replace(0, np.nan)
    return float(drawdown.max() * 100) if len(drawdown) else 0.0


@dataclass
class StressTestResult:
    risk_config: RiskConfig
    bootstrap_results: list[PathResult]
    shock_results: list[PathResult]

    def all_results(self) -> list[PathResult]:
        return self.bootstrap_results + self.shock_results

    def summary(self) -> str:
        returns = [r.report.total_return_pct for r in self.all_results()]
        worst_dds = [r.worst_drawdown_pct for r in self.all_results()]
        returns_sorted = sorted(returns)
        n = len(returns_sorted)

        def pct(p: float) -> float:
            return returns_sorted[min(n - 1, int(p * n))]

        lines = [
            f"Paths tested: {len(self.bootstrap_results)} bootstrap + {len(self.shock_results)} shock scenarios",
            f"Return  -- mean: {np.mean(returns):+.2f}%  median: {np.median(returns):+.2f}%  "
            f"5th pct: {pct(0.05):+.2f}%  95th pct: {pct(0.95):+.2f}%  worst: {min(returns):+.2f}%",
            f"Worst realized drawdown across all paths -- mean: {np.mean(worst_dds):.2f}%  "
            f"max: {max(worst_dds):.2f}%",
        ]
        for r in self.shock_results:
            lines.append(
                f"  Shock [{r.path_id}]: return={r.report.total_return_pct:+.2f}% "
                f"worst_dd={r.worst_drawdown_pct:.2f}% trades={r.report.num_trades} "
                f"reviews={r.report.reviews_performed}"
            )
        return "\n".join(lines)


def run_stress_test(
    candles: pd.DataFrame,
    risk_config: RiskConfig,
    n_bootstrap_paths: int = 50,
    block_size: int = 120,
    review_interval_days: int | None = 30,
    starting_balance: float = 10_000.0,
    shock_bars: int = 1500,
    seed: int = 0,
) -> StressTestResult:
    """Runs the same risk_config across many bootstrap-resampled paths plus
    the three shock scenarios, with the exact same backtester (including
    review_interval_days, so this measures the circuit breaker's behavior
    under the realistic operating model, not the pessimistic or fold-reset
    extremes) used everywhere else in this project.
    """
    rng = np.random.default_rng(seed)
    bootstrap_results = []
    for i in range(n_bootstrap_paths):
        path = block_bootstrap_resample(candles, block_size, rng)
        report = run_backtest(
            path, risk_config, starting_balance=starting_balance,
            use_recorded_sentiment=False, review_interval_days=review_interval_days,
        )
        bootstrap_results.append(PathResult(f"bootstrap_{i}", report, _worst_drawdown(report.equity_curve)))

    shock_results = []
    start_price = float(candles["close"].iloc[-1])
    for kind in ("sustained_crash", "volatility_spike", "no_edge_chop"):
        path = generate_shock_scenario(kind, shock_bars, start_price, rng)
        report = run_backtest(
            path, risk_config, starting_balance=starting_balance,
            use_recorded_sentiment=False, review_interval_days=review_interval_days,
        )
        shock_results.append(PathResult(kind, report, _worst_drawdown(report.equity_curve)))

    return StressTestResult(risk_config=risk_config, bootstrap_results=bootstrap_results, shock_results=shock_results)
