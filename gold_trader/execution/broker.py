"""Order execution.

SimulatedBroker: a pure local paper-trading ledger. Needs no credentials
at all — useful for instantly trying the system or for backtesting.

OandaPracticeBroker: places real orders on an OANDA *practice* account.
Because OANDA's practice environment streams the same live market prices
as the real one, this is the "demo that mirrors the exact movement of the
market" the system is built around — it's real price action, fake money.

There is deliberately no "OandaLiveBroker" wired into the CLI. Settings.
oanda_host already refuses to resolve to the live API host unless
LIVE_TRADING_CONFIRMED is explicitly set (see config.py) — that gate lives
in config, not here, so nothing in this file can accidentally bypass it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests

from gold_trader.config import Settings
from gold_trader.risk.risk_manager import TradePlan
from gold_trader.strategy.signal_engine import Direction


@dataclass
class Position:
    units: int
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class ClosedTrade:
    direction: Direction
    units: int
    entry_price: float
    exit_price: float
    pnl: float
    reason: str
    opened_at: str = ""
    closed_at: str = ""


class BrokerExecution(ABC):
    @abstractmethod
    def get_account_balance(self) -> float: ...

    @abstractmethod
    def get_open_position(self) -> Position | None: ...

    @abstractmethod
    def open_position(self, plan: TradePlan) -> None: ...

    @abstractmethod
    def close_position(self, exit_price: float, reason: str) -> ClosedTrade | None: ...

    @abstractmethod
    def check_stops(self, current_price: float) -> ClosedTrade | None:
        """Close the open position if price has crossed its stop/take-profit. Returns the closed trade, if any."""


class SimulatedBroker(BrokerExecution):
    def __init__(self, starting_balance: float):
        self.balance = starting_balance
        self._position: Position | None = None
        self.closed_trades: list[ClosedTrade] = []

    def get_account_balance(self) -> float:
        return self.balance

    def get_open_position(self) -> Position | None:
        return self._position

    def open_position(self, plan: TradePlan) -> None:
        if self._position is not None:
            raise RuntimeError("SimulatedBroker already has an open position")
        self._position = Position(plan.units, plan.entry_price, plan.stop_loss, plan.take_profit)

    def _close(self, exit_price: float, reason: str) -> ClosedTrade:
        pos = self._position
        assert pos is not None
        pnl = (exit_price - pos.entry_price) * pos.units
        self.balance += pnl
        direction = Direction.LONG if pos.units > 0 else Direction.SHORT
        trade = ClosedTrade(direction, pos.units, pos.entry_price, exit_price, pnl, reason)
        self.closed_trades.append(trade)
        self._position = None
        return trade

    def close_position(self, exit_price: float, reason: str = "manual") -> ClosedTrade | None:
        if self._position is None:
            return None
        return self._close(exit_price, reason)

    def check_stops(self, current_price: float) -> ClosedTrade | None:
        pos = self._position
        if pos is None:
            return None
        if pos.units > 0:  # long
            if current_price <= pos.stop_loss:
                return self._close(pos.stop_loss, "stop_loss")
            if current_price >= pos.take_profit:
                return self._close(pos.take_profit, "take_profit")
        else:  # short
            if current_price >= pos.stop_loss:
                return self._close(pos.stop_loss, "stop_loss")
            if current_price <= pos.take_profit:
                return self._close(pos.take_profit, "take_profit")
        return None


class OandaPracticeBroker(BrokerExecution):
    """Executes against the OANDA practice account configured in Settings.

    Refuses to initialize against anything other than the practice host,
    even if misconfigured upstream — this class is only ever meant to
    touch demo money.
    """

    def __init__(self, settings: Settings):
        if settings.oanda_environment == "live":
            raise RuntimeError(
                "OandaPracticeBroker refuses to run against a live-flagged environment. "
                "Live execution is intentionally not implemented in this codebase."
            )
        if not settings.has_oanda_credentials:
            raise RuntimeError("OANDA_API_KEY / OANDA_ACCOUNT_ID are not set.")
        self.settings = settings
        self.host = settings.oanda_host
        self.account_id = settings.oanda_account_id
        self.instrument = settings.instrument
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {settings.oanda_api_key}",
            "Content-Type": "application/json",
        })

    def get_account_balance(self) -> float:
        resp = self._session.get(f"{self.host}/v3/accounts/{self.account_id}/summary", timeout=10)
        resp.raise_for_status()
        return float(resp.json()["account"]["balance"])

    def get_open_position(self) -> Position | None:
        resp = self._session.get(
            f"{self.host}/v3/accounts/{self.account_id}/positions/{self.instrument}", timeout=10
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()["position"]
        long_units = int(data["long"]["units"])
        short_units = int(data["short"]["units"])
        if long_units == 0 and short_units == 0:
            return None
        units = long_units if long_units != 0 else short_units
        avg_price = float(data["long"]["averagePrice"] or data["short"]["averagePrice"])
        return Position(units=units, entry_price=avg_price, stop_loss=0.0, take_profit=0.0)

    def open_position(self, plan: TradePlan) -> None:
        order = {
            "order": {
                "type": "MARKET",
                "instrument": self.instrument,
                "units": str(plan.units),
                "stopLossOnFill": {"price": f"{plan.stop_loss:.2f}"},
                "takeProfitOnFill": {"price": f"{plan.take_profit:.2f}"},
            }
        }
        resp = self._session.post(f"{self.host}/v3/accounts/{self.account_id}/orders", json=order, timeout=15)
        resp.raise_for_status()

    def close_position(self, exit_price: float, reason: str = "manual") -> ClosedTrade | None:
        pos = self.get_open_position()
        if pos is None:
            return None
        side = "longUnits" if pos.units > 0 else "shortUnits"
        resp = self._session.put(
            f"{self.host}/v3/accounts/{self.account_id}/positions/{self.instrument}/close",
            json={side: "ALL"},
            timeout=15,
        )
        resp.raise_for_status()
        direction = Direction.LONG if pos.units > 0 else Direction.SHORT
        pnl = (exit_price - pos.entry_price) * pos.units
        return ClosedTrade(direction, pos.units, pos.entry_price, exit_price, pnl, reason)

    def check_stops(self, current_price: float) -> ClosedTrade | None:
        # OANDA enforces stop-loss/take-profit server-side (attached on fill above),
        # so there is nothing to poll for client-side here.
        return None
