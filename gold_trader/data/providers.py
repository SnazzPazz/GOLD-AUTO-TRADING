"""Market data providers.

Two implementations:
  - YFinanceDataProvider: free, no API key, used for backtesting on
    historical COMEX gold futures (GC=F) data.
  - OandaDataProvider: real OANDA candle/pricing data (practice or live
    host, decided entirely by config.Settings.oanda_host). Used for the
    live-mirroring demo loop.

Both expose the same interface so the rest of the system (indicators,
patterns, strategy, backtester) never has to know which one is in use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import requests

from gold_trader.config import Settings


class MarketDataProvider(ABC):
    @abstractmethod
    def get_historical_candles(self, granularity: str, count: int) -> pd.DataFrame:
        """Return a DataFrame indexed by time with columns open/high/low/close/volume."""

    @abstractmethod
    def get_latest_price(self) -> float:
        """Return the latest tradable mid price."""


class YFinanceDataProvider(MarketDataProvider):
    """No API key required. Used for backtesting and offline experimentation."""

    _GRANULARITY_TO_YF = {
        "M1": ("1m", "7d"),
        "M5": ("5m", "60d"),
        "M15": ("15m", "60d"),
        "M30": ("30m", "60d"),
        "H1": ("60m", "730d"),
        "H4": ("60m", "730d"),  # yfinance has no native 4h bar; caller may resample.
        "D": ("1d", "max"),
    }

    def __init__(self, ticker: str = "GC=F"):
        self.ticker = ticker

    def get_historical_candles(self, granularity: str, count: int) -> pd.DataFrame:
        import yfinance as yf

        interval, period = self._GRANULARITY_TO_YF.get(granularity, ("15m", "60d"))
        df = yf.Ticker(self.ticker).history(period=period, interval=interval)
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {self.ticker} ({interval}/{period})")
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index.name = "time"
        if granularity == "H4":
            df = (
                df.resample("4h")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
            )
        return df.tail(count)

    def get_latest_price(self) -> float:
        import yfinance as yf

        fast = yf.Ticker(self.ticker).fast_info
        price = fast.get("lastPrice") or fast.get("last_price")
        if price is None:
            raise RuntimeError(f"yfinance has no live price for {self.ticker}")
        return float(price)


class OandaDataProvider(MarketDataProvider):
    """Talks to the OANDA v20 REST API (practice or live, per settings.oanda_host)."""

    _GRANULARITY_MAP = {  # our names already match OANDA's, kept explicit for clarity
        "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4", "D": "D",
    }

    def __init__(self, settings: Settings):
        if not settings.has_oanda_credentials:
            raise RuntimeError("OANDA_API_KEY / OANDA_ACCOUNT_ID are not set.")
        self.settings = settings
        self.host = settings.oanda_host
        self.instrument = settings.instrument
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {settings.oanda_api_key}",
            "Content-Type": "application/json",
        })

    def get_historical_candles(self, granularity: str, count: int) -> pd.DataFrame:
        url = f"{self.host}/v3/instruments/{self.instrument}/candles"
        params = {
            "granularity": self._GRANULARITY_MAP.get(granularity, granularity),
            "count": min(count, 5000),
            "price": "M",  # midpoint candles
        }
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        rows = [
            {
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": int(c["volume"]),
            }
            for c in candles
            if c.get("complete", True)
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"OANDA returned no candles for {self.instrument}")
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time")

    def get_latest_price(self) -> float:
        url = f"{self.host}/v3/accounts/{self.settings.oanda_account_id}/pricing"
        resp = self._session.get(url, params={"instruments": self.instrument}, timeout=10)
        resp.raise_for_status()
        prices = resp.json()["prices"][0]
        bid = float(prices["bids"][0]["price"])
        ask = float(prices["asks"][0]["price"])
        return (bid + ask) / 2

    def get_account_balance(self) -> float:
        url = f"{self.host}/v3/accounts/{self.settings.oanda_account_id}/summary"
        resp = self._session.get(url, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["account"]["balance"])
