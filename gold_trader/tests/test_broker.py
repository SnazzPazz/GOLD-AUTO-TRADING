import pytest

from gold_trader.execution.broker import SimulatedBroker
from gold_trader.risk.risk_manager import TradePlan
from gold_trader.strategy.signal_engine import Direction


def test_long_stop_wins_over_target_in_same_bar():
    """If a single bar's range touches both the stop and the target, the
    worst case (stop-loss) must win regardless of which extreme (high/low)
    is checked first -- never assume the best case happened."""
    broker = SimulatedBroker(starting_balance=10_000)
    broker.open_position(TradePlan(Direction.LONG, units=10, entry_price=2000.0,
                                    stop_loss=1990.0, take_profit=2010.0, risk_amount=100.0))
    closed = broker.check_stops_bar(high=2015.0, low=1985.0)  # bar spans both levels
    assert closed is not None
    assert closed.reason == "stop_loss"
    assert closed.exit_price == 1990.0


def test_short_stop_wins_over_target_in_same_bar():
    """Same guarantee for shorts: this is the exact case the old
    check-low-then-high implementation got backwards, since for a short
    the stop sits on the HIGH side and the target on the LOW side."""
    broker = SimulatedBroker(starting_balance=10_000)
    broker.open_position(TradePlan(Direction.SHORT, units=-10, entry_price=2000.0,
                                    stop_loss=2010.0, take_profit=1990.0, risk_amount=100.0))
    closed = broker.check_stops_bar(high=2015.0, low=1985.0)  # bar spans both levels
    assert closed is not None
    assert closed.reason == "stop_loss"
    assert closed.exit_price == 2010.0


def test_short_target_only_hit_when_stop_not_touched():
    broker = SimulatedBroker(starting_balance=10_000)
    broker.open_position(TradePlan(Direction.SHORT, units=-10, entry_price=2000.0,
                                    stop_loss=2010.0, take_profit=1990.0, risk_amount=100.0))
    closed = broker.check_stops_bar(high=2005.0, low=1985.0)  # stop untouched, target touched
    assert closed is not None
    assert closed.reason == "take_profit"
    assert closed.exit_price == 1990.0


def test_no_stop_hit_leaves_position_open():
    broker = SimulatedBroker(starting_balance=10_000)
    broker.open_position(TradePlan(Direction.LONG, units=10, entry_price=2000.0,
                                    stop_loss=1990.0, take_profit=2010.0, risk_amount=100.0))
    closed = broker.check_stops_bar(high=2005.0, low=1995.0)
    assert closed is None
    assert broker.get_open_position() is not None


def test_spread_cost_reduces_pnl():
    broker_no_cost = SimulatedBroker(starting_balance=10_000, spread_cost_per_unit=0.0)
    broker_no_cost.open_position(TradePlan(Direction.LONG, units=10, entry_price=2000.0,
                                            stop_loss=1990.0, take_profit=2010.0, risk_amount=100.0))
    trade_no_cost = broker_no_cost.close_position(2010.0, reason="take_profit")

    broker_with_cost = SimulatedBroker(starting_balance=10_000, spread_cost_per_unit=0.30)
    broker_with_cost.open_position(TradePlan(Direction.LONG, units=10, entry_price=2000.0,
                                              stop_loss=1990.0, take_profit=2010.0, risk_amount=100.0))
    trade_with_cost = broker_with_cost.close_position(2010.0, reason="take_profit")

    assert trade_no_cost.pnl == pytest.approx(100.0)
    # 10 units * $0.30/unit round-trip cost = $3 taken off the same gross move.
    assert trade_with_cost.pnl == pytest.approx(97.0)
    assert broker_with_cost.balance == pytest.approx(10_097.0)
