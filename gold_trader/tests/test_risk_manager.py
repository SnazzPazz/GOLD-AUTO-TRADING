import pytest

from gold_trader.config import RiskConfig
from gold_trader.risk.risk_manager import RiskManager
from gold_trader.strategy.signal_engine import Direction, Signal


def make_risk_manager(**overrides):
    cfg = RiskConfig(
        risk_per_trade_pct=0.01,
        atr_stop_multiple=1.5,
        reward_risk_ratio=1.5,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.10,
        max_concurrent_positions=1,
        max_leverage=10.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return RiskManager(cfg)


def test_size_trade_risks_exactly_configured_pct():
    rm = make_risk_manager()
    signal = Signal(Direction.LONG, confidence=0.75, price=2000.0, atr=10.0)
    plan = rm.size_trade(signal, account_balance=10_000)
    assert plan is not None
    # risk_amount = 10_000 * 0.01 = 100; stop_distance = 1.5*10 = 15; units ~= 100/15 = 6
    assert plan.units == 6
    assert plan.stop_loss == pytest.approx(2000.0 - 15.0)
    assert plan.take_profit == pytest.approx(2000.0 + 15.0 * 1.5)


def test_size_trade_short_has_negative_units_and_correct_stops():
    rm = make_risk_manager()
    signal = Signal(Direction.SHORT, confidence=0.75, price=2000.0, atr=10.0)
    plan = rm.size_trade(signal, account_balance=10_000)
    assert plan.units < 0
    assert plan.stop_loss == pytest.approx(2000.0 + 15.0)
    assert plan.take_profit == pytest.approx(2000.0 - 15.0 * 1.5)


def test_size_trade_respects_leverage_cap():
    # risk_per_trade_pct=5% with a tight ATR stop would normally size ~333 units;
    # the 2x leverage cap on a $10,000 account must bring that down to 10.
    rm = make_risk_manager(max_leverage=2.0, risk_per_trade_pct=0.05)
    signal = Signal(Direction.LONG, confidence=0.9, price=2000.0, atr=1.0)
    plan = rm.size_trade(signal, account_balance=10_000)
    assert plan is not None
    assert plan.units == 10
    assert plan.units * plan.entry_price <= 10_000 * 2.0 + 1e-6


def test_daily_loss_circuit_breaker_halts_trading():
    rm = make_risk_manager(max_daily_loss_pct=0.03)
    rm.start_of_day(10_000)
    rm.update_equity(9_600)  # down 4% > 3% limit
    can_trade, reason = rm.can_trade(open_position_count=0)
    assert can_trade is False
    assert "Daily loss" in reason


def test_max_drawdown_circuit_breaker_halts_permanently():
    rm = make_risk_manager(max_drawdown_pct=0.10)
    rm.update_equity(10_000)
    rm.update_equity(8_900)  # down 11% from peak > 10% limit
    can_trade, reason = rm.can_trade(open_position_count=0)
    assert can_trade is False
    assert "drawdown" in reason.lower()
    # Recovering equity should not lift a permanent halt.
    rm.update_equity(10_500)
    can_trade_after_recovery, _ = rm.can_trade(open_position_count=0)
    assert can_trade_after_recovery is False


def test_max_concurrent_positions_blocks_new_trade():
    rm = make_risk_manager(max_concurrent_positions=1)
    can_trade, reason = rm.can_trade(open_position_count=1)
    assert can_trade is False
    assert "concurrent" in reason


def test_max_trades_per_day_blocks_further_trades():
    rm = make_risk_manager(max_trades_per_day=2)
    rm.start_of_day(10_000)
    rm.record_trade_opened()
    rm.record_trade_opened()
    can_trade, reason = rm.can_trade(open_position_count=0)
    assert can_trade is False
    assert "trades per day" in reason


def test_max_trades_per_day_resets_on_new_day():
    rm = make_risk_manager(max_trades_per_day=1)
    rm.start_of_day(10_000)
    rm.record_trade_opened()
    can_trade, _ = rm.can_trade(open_position_count=0)
    assert can_trade is False

    rm.start_of_day(10_000)  # new day
    can_trade_after_reset, _ = rm.can_trade(open_position_count=0)
    assert can_trade_after_reset is True


def test_flat_signal_sizes_to_nothing():
    rm = make_risk_manager()
    signal = Signal(Direction.FLAT, confidence=0.0, price=2000.0, atr=10.0)
    assert rm.size_trade(signal, account_balance=10_000) is None
