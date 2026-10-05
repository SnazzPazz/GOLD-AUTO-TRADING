"""Position sizing and capital-protection circuit breakers.

This is where "don't lose your capital" is actually enforced — not by a
promise that trades always win (no algorithm can promise that on leveraged
gold), but by bounding how much any single trade, day, or drawdown can
cost before the system takes itself offline.
"""
from __future__ import annotations

from dataclasses import dataclass

from gold_trader.config import RiskConfig
from gold_trader.strategy.signal_engine import Direction, Signal


@dataclass
class TradePlan:
    direction: Direction
    units: int           # positive for long, negative for short (OANDA convention)
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float


class RiskManager:
    def __init__(self, risk_config: RiskConfig):
        self.cfg = risk_config
        self.day_start_balance: float | None = None
        self.peak_equity: float | None = None
        self.halted_for_day = False
        self.halted_permanently = False
        self.halt_reason: str | None = None

    def start_of_day(self, balance: float) -> None:
        self.day_start_balance = balance
        self.halted_for_day = False

    def update_equity(self, equity: float) -> None:
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity

        if self.day_start_balance is not None:
            daily_loss_pct = (self.day_start_balance - equity) / self.day_start_balance
            if daily_loss_pct >= self.cfg.max_daily_loss_pct:
                self.halted_for_day = True
                self.halt_reason = (
                    f"Daily loss circuit breaker: down {daily_loss_pct:.1%} "
                    f"(limit {self.cfg.max_daily_loss_pct:.1%}). No new trades until tomorrow."
                )

        if self.peak_equity:
            drawdown_pct = (self.peak_equity - equity) / self.peak_equity
            if drawdown_pct >= self.cfg.max_drawdown_pct:
                self.halted_permanently = True
                self.halt_reason = (
                    f"Max drawdown circuit breaker: down {drawdown_pct:.1%} from peak "
                    f"(limit {self.cfg.max_drawdown_pct:.1%}). Trading halted — manual review required."
                )

    def can_trade(self, open_position_count: int) -> tuple[bool, str | None]:
        if self.halted_permanently:
            return False, self.halt_reason
        if self.halted_for_day:
            return False, self.halt_reason
        if open_position_count >= self.cfg.max_concurrent_positions:
            return False, f"max concurrent positions ({self.cfg.max_concurrent_positions}) reached"
        return True, None

    def size_trade(self, signal: Signal, account_balance: float) -> TradePlan | None:
        if signal.direction is Direction.FLAT or signal.atr <= 0:
            return None

        stop_distance = self.cfg.atr_stop_multiple * signal.atr
        if stop_distance <= 0:
            return None

        risk_amount = account_balance * self.cfg.risk_per_trade_pct
        # For XAU_USD on OANDA, 1 unit = 1 troy ounce; price is quoted per ounce,
        # so $ risk / $ stop-distance-per-unit gives units directly.
        units = risk_amount / stop_distance

        notional = units * signal.price
        max_notional = account_balance * self.cfg.max_leverage
        if notional > max_notional:
            units = max_notional / signal.price

        units = int(units)
        if units <= 0:
            return None

        if signal.direction is Direction.LONG:
            stop_loss = signal.price - stop_distance
            take_profit = signal.price + stop_distance * self.cfg.reward_risk_ratio
        else:
            units = -units
            stop_loss = signal.price + stop_distance
            take_profit = signal.price - stop_distance * self.cfg.reward_risk_ratio

        return TradePlan(
            direction=signal.direction,
            units=units,
            entry_price=signal.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
        )
