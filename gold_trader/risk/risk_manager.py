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
        self.trades_today = 0
        self._last_day = None

        # Separate from peak_equity: this NEVER resets on an ordinary review, because
        # review_and_reset exists to clear a single acute drawdown, not to repeatedly
        # forgive a sustained bad patch. Stress-testing against synthetic reshuffled
        # paths (gold_trader/research/stress_test.py) found that review_and_reset alone
        # lets losses compound across cycles: three successive 10%-then-reviewed hits
        # compound to ~27% cumulative loss, not 10% -- the real historical path just
        # happened to need few enough review cycles, with good recovery between them,
        # that this never showed up. This tracks true loss from the account's all-time
        # high and halts for real (no routine-review path clears it) once it's severe.
        self.all_time_peak_equity: float | None = None
        self.halted_catastrophically = False

    def start_of_day(self, balance: float) -> None:
        self.day_start_balance = balance
        self.halted_for_day = False
        self.trades_today = 0

    def roll_day_if_needed(self, timestamp, balance: float) -> bool:
        """Calls start_of_day() exactly once per calendar day, the first
        time this is called for that day. The backtester and the live demo
        loop both need this same "is this a new day" check on every bar/poll;
        centralizing it here means they can't drift apart on how a day
        boundary is detected (e.g. one using midnight UTC, the other some
        other rule) by each reimplementing it separately. Returns True the
        one time a new day was actually detected, so a caller that needs to
        piggyback other once-per-day logic (e.g. a periodic review check)
        knows when that happened."""
        bar_day = timestamp.date() if hasattr(timestamp, "date") else None
        if bar_day is not None and bar_day != self._last_day:
            self.start_of_day(balance)
            self._last_day = bar_day
            return True
        return False

    def record_trade_opened(self) -> None:
        self.trades_today += 1

    def review_and_reset(self, current_equity: float) -> None:
        """A human reviewed a permanent halt and decided to resume trading --
        this is the only way `halted_permanently` ever clears, deliberately
        (see module docstring: the breaker exists to force that review, not
        to be worked around). The loss that triggered the halt is real and
        stays on the books; what resets is the drawdown baseline, now
        measured from this point forward rather than against the old peak
        that caused the halt. A further drop of the same magnitude from
        here will trigger it again.

        Deliberately does NOT clear `halted_catastrophically` -- that one
        tracks loss from the account's true all-time peak across every
        review cycle, precisely so routine review can't repeatedly forgive
        a sustained bad patch into an unbounded cumulative loss. Clearing
        that one goes through acknowledge_catastrophic_loss_and_reset
        instead, a separate and deliberately not-routine action.
        """
        self.halted_permanently = False
        self.peak_equity = current_equity
        if not self.halted_catastrophically:
            self.halt_reason = None

    def acknowledge_catastrophic_loss_and_reset(self, current_equity: float) -> None:
        """Clears `halted_catastrophically`. Separate from review_and_reset
        on purpose: this halt means a sustained bad patch has already cost
        far more than any single review cycle's limit, and clearing it
        should be a deliberate, rare action, not something that happens as
        a side effect of the same routine check-in that clears ordinary
        drawdown halts."""
        self.halted_catastrophically = False
        self.halted_permanently = False
        self.peak_equity = current_equity
        self.all_time_peak_equity = current_equity
        self.halt_reason = None

    def update_equity(self, equity: float) -> None:
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.all_time_peak_equity is None or equity > self.all_time_peak_equity:
            self.all_time_peak_equity = equity

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

        if self.all_time_peak_equity:
            cumulative_drawdown_pct = (self.all_time_peak_equity - equity) / self.all_time_peak_equity
            if cumulative_drawdown_pct >= self.cfg.cumulative_drawdown_halt_pct:
                self.halted_catastrophically = True
                self.halt_reason = (
                    f"Cumulative drawdown circuit breaker: down {cumulative_drawdown_pct:.1%} from the "
                    f"account's all-time peak (limit {self.cfg.cumulative_drawdown_halt_pct:.1%}). "
                    f"Trading halted -- this does not clear on routine review; the repeated ordinary "
                    f"drawdown halt has been forgiven too many times without real recovery."
                )

    def can_trade(self, open_position_count: int) -> tuple[bool, str | None]:
        if self.halted_catastrophically:
            return False, self.halt_reason
        if self.halted_permanently:
            return False, self.halt_reason
        if self.halted_for_day:
            return False, self.halt_reason
        if open_position_count >= self.cfg.max_concurrent_positions:
            return False, f"max concurrent positions ({self.cfg.max_concurrent_positions}) reached"
        if self.trades_today >= self.cfg.max_trades_per_day:
            return False, f"max trades per day ({self.cfg.max_trades_per_day}) reached"
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

        # Recomputed from the final (possibly leverage-capped, possibly truncated by
        # int()) unit count -- using the pre-cap risk_amount from above would report a
        # dollar-risk figure that no longer matches the position actually being sized
        # whenever the leverage cap or the int() truncation changed `units`.
        risk_amount = units * stop_distance

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
