from unittest.mock import MagicMock

import pytest

from gold_trader.config import Settings
from gold_trader.execution.broker import OandaPracticeBroker, SimulatedBroker
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


def _make_oanda_broker() -> OandaPracticeBroker:
    settings = Settings(oanda_api_key="fake-key", oanda_account_id="fake-account", oanda_environment="practice")
    return OandaPracticeBroker(settings)


def _fake_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}") if status_code >= 400 else None
    return resp


def test_get_open_position_parses_short_position_with_decimal_unit_string():
    """Regression test for a real bug found live against a real OANDA practice
    account this session: OANDA returns position units as a decimal string
    ("-40.0000", not "-40"), which int() can't parse at all (ValueError) --
    this crashed the entire demo loop on the very first poll cycle after a
    real position was opened. No prior test exercised this method against
    OANDA's actual response format."""
    broker = _make_oanda_broker()
    broker._session.get = MagicMock(return_value=_fake_response({
        "position": {
            "long": {"units": "0.0000", "averagePrice": "0.0"},
            "short": {"units": "-40.0000", "averagePrice": "4011.21"},
        }
    }))
    pos = broker.get_open_position()
    assert pos is not None
    assert pos.units == -40
    assert pos.entry_price == pytest.approx(4011.21)


def test_get_open_position_parses_long_position_with_decimal_unit_string():
    broker = _make_oanda_broker()
    broker._session.get = MagicMock(return_value=_fake_response({
        "position": {
            "long": {"units": "25.0000", "averagePrice": "4011.21"},
            "short": {"units": "0.0000", "averagePrice": "0.0"},
        }
    }))
    pos = broker.get_open_position()
    assert pos is not None
    assert pos.units == 25
    assert pos.entry_price == pytest.approx(4011.21)


def test_get_open_position_returns_none_when_flat():
    broker = _make_oanda_broker()
    broker._session.get = MagicMock(return_value=_fake_response({
        "position": {
            "long": {"units": "0.0000", "averagePrice": "0.0"},
            "short": {"units": "0.0000", "averagePrice": "0.0"},
        }
    }))
    assert broker.get_open_position() is None


def test_get_open_position_returns_none_on_404():
    broker = _make_oanda_broker()
    broker._session.get = MagicMock(return_value=_fake_response({}, status_code=404))
    assert broker.get_open_position() is None
