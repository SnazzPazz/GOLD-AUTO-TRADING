import pandas as pd
import pytest

from gold_trader.risk.approval_gate import PortfolioApprovalGate
from gold_trader.risk.risk_manager import TradePlan
from gold_trader.strategy.signal_engine import Direction


def make_plan(entry=2000.0, stop=1990.0):
    return TradePlan(Direction.LONG, units=10, entry_price=entry, stop_loss=stop,
                      take_profit=2015.0, risk_amount=100.0)


def test_slippage_within_limit_approved():
    gate = PortfolioApprovalGate(max_slippage_atr_multiple=0.5)
    plan = make_plan(entry=2000.0, stop=1990.0)  # stop distance = 10, limit = 5
    result = gate.check_slippage(plan, current_price=2003.0)  # drift = 3 <= 5
    assert result.approved is True


def test_slippage_beyond_limit_rejected():
    gate = PortfolioApprovalGate(max_slippage_atr_multiple=0.5)
    plan = make_plan(entry=2000.0, stop=1990.0)  # stop distance = 10, limit = 5
    result = gate.check_slippage(plan, current_price=2008.0)  # drift = 8 > 5
    assert result.approved is False
    assert "drifted" in result.reason


def test_volatility_regime_approves_with_insufficient_history():
    gate = PortfolioApprovalGate(regime_lookback=100)
    atr = pd.Series([1.0, 1.1, 0.9])  # far fewer than the lookback needs
    result = gate.check_volatility_regime(atr)
    assert result.approved is True


def test_volatility_regime_rejects_abnormally_quiet_market():
    gate = PortfolioApprovalGate(regime_lookback=100, regime_low_percentile=10.0, regime_high_percentile=95.0)
    # 99 normal bars around ATR=1.0, then one bar collapsing to near zero.
    normal = [1.0 + 0.01 * (i % 5) for i in range(99)]
    atr = pd.Series(normal + [0.01])
    result = gate.check_volatility_regime(atr)
    assert result.approved is False
    assert "too quiet" in result.reason


def test_volatility_regime_rejects_abnormal_spike():
    gate = PortfolioApprovalGate(regime_lookback=100, regime_low_percentile=10.0, regime_high_percentile=95.0)
    normal = [1.0 + 0.01 * (i % 5) for i in range(99)]
    atr = pd.Series(normal + [10.0])  # sudden spike
    result = gate.check_volatility_regime(atr)
    assert result.approved is False
    assert "abnormally high" in result.reason


def test_volatility_regime_approves_normal_conditions():
    gate = PortfolioApprovalGate(regime_lookback=100, regime_low_percentile=10.0, regime_high_percentile=95.0)
    normal = [1.0 + 0.01 * (i % 5) for i in range(100)]
    atr = pd.Series(normal)
    result = gate.check_volatility_regime(atr)
    assert result.approved is True


def test_approve_checks_regime_before_slippage():
    gate = PortfolioApprovalGate(regime_lookback=100)
    normal = [1.0 + 0.01 * (i % 5) for i in range(99)]
    atr = pd.Series(normal + [0.001])  # abnormally quiet
    plan = make_plan()
    result = gate.approve(plan, current_price=plan.entry_price, atr_series=atr)
    assert result.approved is False
    assert "too quiet" in result.reason
