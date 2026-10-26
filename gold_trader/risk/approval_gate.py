"""Final pre-execution approval gate.

Inspired by the Portfolio-Manager-approves/rejects-the-trade step in
multi-agent trading frameworks (e.g. TauricResearch/TradingAgents, where a
Risk Management team and a Portfolio Manager sign off on the Trader's
proposal before anything executes). This system has no LLM agents, but the
same idea holds without one: a well-sized, well-confirmed signal can still
be a bad trade to execute right now, for reasons the signal engine and risk
manager don't look at.

Two checks, both reasons to reject rather than execute:

  - Staleness / slippage: this system polls rather than streams, so the
    price used to size the trade can be seconds to a minute stale by the
    time an order would actually go in. If price has already moved too far
    from the signal price, the stop/target the risk manager computed no
    longer reflects the risk actually being taken.
  - Volatility regime: ATR-based sizing assumes "normal" volatility for this
    instrument right now. When ATR is abnormally low versus its own recent
    history, the market is too quiet for these technical patterns to mean
    much (noise dominates real signal); when it's abnormally high, something
    unusual is happening (a data release, a liquidity gap) and a standard
    stop multiple may not hold. Both get rejected rather than traded through.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gold_trader.risk.risk_manager import TradePlan


@dataclass
class ApprovalResult:
    approved: bool
    reason: str | None = None


class PortfolioApprovalGate:
    def __init__(
        self,
        max_slippage_atr_multiple: float = 0.5,
        regime_lookback: int = 100,
        regime_low_percentile: float = 10.0,
        regime_high_percentile: float = 95.0,
    ):
        self.max_slippage_atr_multiple = max_slippage_atr_multiple
        self.regime_lookback = regime_lookback
        self.regime_low_percentile = regime_low_percentile
        self.regime_high_percentile = regime_high_percentile

    def check_volatility_regime(self, atr_series: pd.Series) -> ApprovalResult:
        recent = atr_series.tail(self.regime_lookback).dropna()
        min_sample = max(20, self.regime_lookback // 2)
        if len(recent) < min_sample:
            return ApprovalResult(True)  # not enough history yet to judge a regime

        current_atr = recent.iloc[-1]
        low = recent.quantile(self.regime_low_percentile / 100)
        high = recent.quantile(self.regime_high_percentile / 100)

        if current_atr < low:
            return ApprovalResult(
                False,
                f"volatility regime too quiet: ATR {current_atr:.3f} is below the "
                f"{self.regime_low_percentile:.0f}th percentile ({low:.3f}) of its last "
                f"{len(recent)} bars",
            )
        if current_atr > high:
            return ApprovalResult(
                False,
                f"volatility regime abnormally high: ATR {current_atr:.3f} is above the "
                f"{self.regime_high_percentile:.0f}th percentile ({high:.3f}) of its last "
                f"{len(recent)} bars",
            )
        return ApprovalResult(True)

    def check_slippage(self, plan: TradePlan, current_price: float) -> ApprovalResult:
        stop_distance = abs(plan.entry_price - plan.stop_loss)
        if stop_distance <= 0:
            return ApprovalResult(True)

        drift = abs(current_price - plan.entry_price)
        max_drift = self.max_slippage_atr_multiple * stop_distance
        if drift > max_drift:
            return ApprovalResult(
                False,
                f"price drifted {drift:.3f} since the signal was generated "
                f"(signal price {plan.entry_price:.3f}, now {current_price:.3f}; "
                f"limit was {max_drift:.3f}) — stale signal, rejecting rather than "
                f"executing on outdated risk numbers",
            )
        return ApprovalResult(True)

    def approve(self, plan: TradePlan, current_price: float, atr_series: pd.Series) -> ApprovalResult:
        regime = self.check_volatility_regime(atr_series)
        if not regime.approved:
            return regime
        return self.check_slippage(plan, current_price)
