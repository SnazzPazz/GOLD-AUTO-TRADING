"""Bar-by-bar backtest: walks forward through historical candles one bar at
a time so the signal engine only ever sees data up to and including the
current bar (no lookahead), exactly mirroring how the live/demo loop
consumes data.

This is the main way to "verify the trades are reliable" before trusting
the demo loop at all: it runs the exact same signal engine, risk manager,
and broker-sizing math against history and reports whether the strategy
would have made or lost money, and by how much, under realistic risk
constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_trader.config import RiskConfig
from gold_trader.execution.broker import SimulatedBroker
from gold_trader.risk.approval_gate import PortfolioApprovalGate
from gold_trader.risk.risk_manager import RiskManager
from gold_trader.sentiment.cot_data import PointInTimeCotLookup
from gold_trader.sentiment.history import PointInTimeSentimentLookup
from gold_trader.strategy import signal_engine as se


@dataclass
class BacktestReport:
    starting_balance: float
    ending_balance: float
    total_return_pct: float
    num_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_like: float
    regime_rejections: int
    reviews_performed: int
    equity_curve: pd.Series
    trades: list

    def summary(self) -> str:
        return (
            f"Starting balance:  {self.starting_balance:,.2f}\n"
            f"Ending balance:    {self.ending_balance:,.2f}\n"
            f"Total return:      {self.total_return_pct:+.2f}%\n"
            f"Trades:            {self.num_trades}\n"
            f"Win rate:          {self.win_rate:.1%}\n"
            f"Profit factor:     {self.profit_factor:.2f}\n"
            f"Max drawdown:      {self.max_drawdown_pct:.2f}%\n"
            f"Sharpe-like ratio: {self.sharpe_like:.2f}\n"
            f"Regime-rejected:   {self.regime_rejections} signals skipped (abnormal volatility)\n"
            f"Reviews performed: {self.reviews_performed} (simulated human check-ins that cleared a halt)\n"
        )


def run_backtest(
    candles: pd.DataFrame,
    risk_config: RiskConfig,
    starting_balance: float = 10_000.0,
    warmup_bars: int = 210,
    use_recorded_sentiment: bool = True,
    cot_lookup: PointInTimeCotLookup | None = None,
    review_interval_days: int | None = None,
) -> BacktestReport:
    """`use_recorded_sentiment=True` replays point-in-time sentiment recorded
    by the live demo loop (see sentiment/history.py) wherever it exists for
    this period; bars before any recording started simply get no sentiment
    input, same as if sentiment were disabled -- there is no way to
    backdate it honestly, so this never guesses.

    `cot_lookup`: pass a pre-fetched `PointInTimeCotLookup` when calling this
    many times (e.g. from walk-forward research) so the CFTC API is hit once,
    not once per call. If `risk_config.enable_cot_filter` is True and no
    lookup is supplied, one is fetched here for convenience on a single
    standalone backtest run.

    `review_interval_days`: if set, simulates a human checking in on the bot
    every N days -- if the max-drawdown circuit breaker has permanently
    halted trading, the check-in clears it (see RiskManager.review_and_reset)
    rather than leaving it halted for the rest of the backtest. This is the
    system's actual intended operating model (stop, wait for a human, don't
    keep trading blind); leave as None for the pessimistic "never looked at
    it again" reading of a long continuous backtest.
    """
    broker = SimulatedBroker(starting_balance, spread_cost_per_unit=risk_config.spread_cost_per_unit)
    risk_mgr = RiskManager(risk_config)
    risk_mgr.start_of_day(starting_balance)
    risk_mgr.update_equity(starting_balance)
    gate = PortfolioApprovalGate(
        regime_lookback=risk_config.volatility_regime_lookback,
        regime_low_percentile=risk_config.volatility_regime_low_percentile,
        regime_high_percentile=risk_config.volatility_regime_high_percentile,
    )
    sentiment_lookup = PointInTimeSentimentLookup() if use_recorded_sentiment else None
    if risk_config.enable_cot_filter and cot_lookup is None:
        cot_lookup = PointInTimeCotLookup()

    enriched = se.prepare(candles)
    atr_series = enriched["atr_14"]
    index = enriched.index
    regime_lookback = risk_config.volatility_regime_lookback
    equity_curve = []
    last_day = None
    last_review_day = None
    regime_rejections = 0
    reviews_performed = 0

    # Iterates with .iloc[i] / bounded Series slices rather than re-slicing the whole,
    # ever-growing enriched DataFrame every bar (enriched.iloc[:i+1]) -- that used to copy
    # all ~20 columns on every one of potentially thousands of iterations, most of whose
    # rows generate_signal and the regime check never looked at anyway.
    for i in range(warmup_bars, len(enriched)):
        row = enriched.iloc[i]
        current_time = index[i]

        bar_day = current_time.date() if hasattr(current_time, "date") else None
        if bar_day is not None and bar_day != last_day:
            risk_mgr.start_of_day(broker.get_account_balance())
            last_day = bar_day

            if review_interval_days is not None:
                if last_review_day is None:
                    last_review_day = bar_day
                elif (bar_day - last_review_day).days >= review_interval_days:
                    if risk_mgr.halted_permanently:
                        risk_mgr.review_and_reset(broker.get_account_balance())
                        reviews_performed += 1
                    last_review_day = bar_day

        broker.check_stops_bar(row["high"], row["low"])

        equity = broker.get_account_balance()
        risk_mgr.update_equity(equity)
        equity_curve.append((current_time, equity))

        open_positions = 1 if broker.get_open_position() else 0
        can_trade, _ = risk_mgr.can_trade(open_positions)
        if not can_trade or open_positions:
            continue

        atr_window = atr_series.iloc[max(0, i + 1 - regime_lookback):i + 1]
        regime = gate.check_volatility_regime(atr_window)
        if not regime.approved:
            regime_rejections += 1
            continue

        sentiment_score = sentiment_lookup.score_at(current_time.timestamp()) if sentiment_lookup else None
        cot_percentile = None
        if cot_lookup is not None and risk_config.enable_cot_filter:
            cot_reading = cot_lookup.reading_at(current_time)
            cot_percentile = cot_reading.percentile_in_history if cot_reading else None

        signal = se.generate_signal_from_row(
            row, i, sentiment_score=sentiment_score, min_confirmations=risk_config.min_confirmations,
            cot_percentile=cot_percentile,
            cot_extreme_low=risk_config.cot_extreme_low_percentile,
            cot_extreme_high=risk_config.cot_extreme_high_percentile,
        )
        if signal.direction is se.Direction.FLAT:
            continue

        plan = risk_mgr.size_trade(signal, broker.get_account_balance())
        if plan:
            broker.open_position(plan)
            risk_mgr.record_trade_opened()

    # Close any position still open at the end of the test on the last close price.
    if broker.get_open_position() is not None:
        broker.close_position(enriched.iloc[-1]["close"], reason="end_of_backtest")

    curve = pd.Series(dict(equity_curve))
    trades = broker.closed_trades

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    win_rate = len(wins) / len(trades) if trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    running_max = curve.cummax()
    drawdown = (running_max - curve) / running_max.replace(0, np.nan)
    max_dd = float(drawdown.max() * 100) if len(drawdown) else 0.0

    returns = curve.pct_change().dropna()
    sharpe_like = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    ending_balance = broker.get_account_balance()
    total_return_pct = (ending_balance - starting_balance) / starting_balance * 100

    return BacktestReport(
        starting_balance=starting_balance,
        ending_balance=ending_balance,
        total_return_pct=total_return_pct,
        num_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd,
        sharpe_like=sharpe_like,
        regime_rejections=regime_rejections,
        reviews_performed=reviews_performed,
        equity_curve=curve,
        trades=trades,
    )
