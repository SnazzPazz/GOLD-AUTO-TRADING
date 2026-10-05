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
from gold_trader.risk.risk_manager import RiskManager
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
        )


def run_backtest(
    candles: pd.DataFrame,
    risk_config: RiskConfig,
    starting_balance: float = 10_000.0,
    warmup_bars: int = 210,
) -> BacktestReport:
    broker = SimulatedBroker(starting_balance)
    risk_mgr = RiskManager(risk_config)
    risk_mgr.start_of_day(starting_balance)
    risk_mgr.update_equity(starting_balance)

    enriched = se.prepare(candles)
    equity_curve = []
    last_day = None

    for i in range(warmup_bars, len(enriched)):
        window = enriched.iloc[: i + 1]
        row = window.iloc[-1]

        bar_day = window.index[-1].date() if hasattr(window.index[-1], "date") else None
        if bar_day is not None and bar_day != last_day:
            risk_mgr.start_of_day(broker.get_account_balance())
            last_day = bar_day

        closed = broker.check_stops(row["low"])
        if closed is None:
            closed = broker.check_stops(row["high"])

        equity = broker.get_account_balance()
        risk_mgr.update_equity(equity)
        equity_curve.append((window.index[-1], equity))

        open_positions = 1 if broker.get_open_position() else 0
        can_trade, _ = risk_mgr.can_trade(open_positions)
        if not can_trade or open_positions:
            continue

        signal = se.generate_signal(window)
        if signal.direction is se.Direction.FLAT:
            continue

        plan = risk_mgr.size_trade(signal, broker.get_account_balance())
        if plan:
            broker.open_position(plan)

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
        equity_curve=curve,
        trades=trades,
    )
