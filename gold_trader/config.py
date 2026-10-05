"""Central configuration, loaded from environment variables / .env.

Safety note: the system defaults to OANDA's *practice* (demo) environment.
Switching to the live host requires a second, explicit confirmation flag so
that a stray config change can never silently start trading real money.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


OANDA_PRACTICE_HOST = "https://api-fxpractice.oanda.com"
OANDA_LIVE_HOST = "https://api-fxtrade.oanda.com"


@dataclass
class RiskConfig:
    # Fraction of account balance risked on a single trade's stop-loss distance.
    risk_per_trade_pct: float = field(default_factory=lambda: _get_float("RISK_PER_TRADE_PCT", 0.01))
    # ATR multiple used to place the stop-loss away from entry.
    atr_stop_multiple: float = field(default_factory=lambda: _get_float("ATR_STOP_MULTIPLE", 1.5))
    # Reward:risk ratio used to place the take-profit.
    reward_risk_ratio: float = field(default_factory=lambda: _get_float("REWARD_RISK_RATIO", 1.5))
    # Trading halts for the rest of the day once this fraction of starting-day balance is lost.
    max_daily_loss_pct: float = field(default_factory=lambda: _get_float("MAX_DAILY_LOSS_PCT", 0.03))
    # Trading halts entirely (manual reset required) once this fraction of peak equity is lost.
    max_drawdown_pct: float = field(default_factory=lambda: _get_float("MAX_DRAWDOWN_PCT", 0.10))
    # Gold-only system: at most this many concurrent open positions (1 = no pyramiding).
    max_concurrent_positions: int = field(default_factory=lambda: _get_int("MAX_CONCURRENT_POSITIONS", 1))
    # Refuse to size a trade beyond this leverage against account balance.
    max_leverage: float = field(default_factory=lambda: _get_float("MAX_LEVERAGE", 10.0))
    # Cap on new positions opened per calendar day, to stop whipsaw/overtrading in choppy conditions.
    max_trades_per_day: int = field(default_factory=lambda: _get_int("MAX_TRADES_PER_DAY", 5))
    # Approximate round-trip bid/ask spread cost per unit (price terms), applied in the
    # simulated broker so backtest/demo results aren't flattered by frictionless fills.
    # OANDA's typical XAU_USD spread is roughly $0.20-$0.40 under normal conditions and
    # widens around news/thin liquidity; this is a conservative flat approximation, not a feed.
    spread_cost_per_unit: float = field(default_factory=lambda: _get_float("SPREAD_COST_PER_UNIT", 0.30))


@dataclass
class Settings:
    instrument: str = field(default_factory=lambda: os.getenv("INSTRUMENT", "XAU_USD"))
    oanda_api_key: str = field(default_factory=lambda: os.getenv("OANDA_API_KEY", ""))
    oanda_account_id: str = field(default_factory=lambda: os.getenv("OANDA_ACCOUNT_ID", ""))
    # "practice" (default, safe) or "live". Live additionally requires LIVE_TRADING_CONFIRMED=yes.
    oanda_environment: str = field(default_factory=lambda: os.getenv("OANDA_ENVIRONMENT", "practice"))
    live_trading_confirmed: bool = field(default_factory=lambda: _get_bool("LIVE_TRADING_CONFIRMED", False))

    starting_demo_balance: float = field(default_factory=lambda: _get_float("STARTING_DEMO_BALANCE", 10_000.0))
    poll_interval_seconds: int = field(default_factory=lambda: _get_int("POLL_INTERVAL_SECONDS", 60))
    candle_granularity: str = field(default_factory=lambda: os.getenv("CANDLE_GRANULARITY", "M15"))

    enable_sentiment: bool = field(default_factory=lambda: _get_bool("ENABLE_SENTIMENT", True))
    # How many independent confirmations (trend/momentum/pattern/Bollinger) must agree before a
    # signal fires. Higher = fewer, more selective trades; lower = more trades, more false positives.
    min_confirmations: int = field(default_factory=lambda: _get_int("MIN_CONFIRMATIONS", 2))

    risk: RiskConfig = field(default_factory=RiskConfig)

    @property
    def oanda_host(self) -> str:
        if self.oanda_environment == "live":
            if not self.live_trading_confirmed:
                raise RuntimeError(
                    "OANDA_ENVIRONMENT=live but LIVE_TRADING_CONFIRMED is not set. "
                    "This system is intentionally shipped as demo/paper-trading only. "
                    "Live execution is not something this codebase will enable on its own."
                )
            return OANDA_LIVE_HOST
        return OANDA_PRACTICE_HOST

    @property
    def has_oanda_credentials(self) -> bool:
        return bool(self.oanda_api_key and self.oanda_account_id)


settings = Settings()
