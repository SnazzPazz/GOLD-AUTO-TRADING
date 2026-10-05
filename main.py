#!/usr/bin/env python3
"""Gold (XAU/USD) automated trading system — demo/paper-trading only.

Usage:
    python main.py backtest [--bars N] [--granularity M15]
    python main.py demo     [--broker simulated|oanda]

`backtest` replays historical candles bar-by-bar through the exact same
signal engine, risk manager, and sizing math the demo loop uses, and
prints a performance report. Run this first and look hard at the numbers
before trusting anything else here.

`demo` runs continuously against either:
  - simulated: a local paper-trading ledger (no credentials needed).
  - oanda:     your OANDA *practice* account, which streams real live
               market prices — real market movement, fake money. This is
               the literal "demo that mirrors the market" feature.

Live trading is intentionally not a mode this CLI offers. See README.md
and gold_trader/config.py for why, and what a deliberate opt-in would
require.
"""
from __future__ import annotations

import argparse
import logging
import time

from gold_trader.backtest.backtester import run_backtest
from gold_trader.config import settings
from gold_trader.data.providers import OandaDataProvider, YFinanceDataProvider
from gold_trader.execution.broker import OandaPracticeBroker, SimulatedBroker
from gold_trader.logging_config import setup_logging
from gold_trader.risk.risk_manager import RiskManager
from gold_trader.sentiment.news_feed import NewsSentimentProvider
from gold_trader.strategy import signal_engine as se

logger = setup_logging()


def cmd_backtest(args: argparse.Namespace) -> None:
    logger.info("Fetching historical %s candles for backtest (source: yfinance GC=F)...", args.granularity)
    provider = YFinanceDataProvider()
    candles = provider.get_historical_candles(args.granularity, args.bars)
    logger.info("Loaded %d candles (%s -> %s)", len(candles), candles.index[0], candles.index[-1])

    report = run_backtest(candles, settings.risk, starting_balance=settings.starting_demo_balance)
    logger.info("Backtest complete.\n%s", report.summary())

    report_path = "reports/latest_backtest.csv"
    report.equity_curve.to_csv(report_path, header=["equity"])
    logger.info("Equity curve written to %s", report_path)

    print("\n" + "=" * 60)
    print(" BACKTEST REPORT (historical data, no real/demo money moved)")
    print("=" * 60)
    print(report.summary())
    print(
        "Past performance on historical data does not guarantee future\n"
        "results. Use this to sanity-check the strategy, not to predict\n"
        "live outcomes."
    )


def cmd_demo(args: argparse.Namespace) -> None:
    if args.broker == "oanda" and not settings.has_oanda_credentials:
        logger.error(
            "No OANDA credentials found. Set OANDA_API_KEY and OANDA_ACCOUNT_ID "
            "(see .env.example) for a free practice account, or run with --broker simulated."
        )
        return

    data_provider = OandaDataProvider(settings) if args.broker == "oanda" else YFinanceDataProvider()
    broker = (
        OandaPracticeBroker(settings)
        if args.broker == "oanda"
        else SimulatedBroker(settings.starting_demo_balance)
    )
    sentiment_provider = NewsSentimentProvider() if settings.enable_sentiment else None

    risk_mgr = RiskManager(settings.risk)
    risk_mgr.start_of_day(broker.get_account_balance())
    risk_mgr.update_equity(broker.get_account_balance())

    logger.info(
        "Starting DEMO loop | broker=%s | instrument=%s | granularity=%s | poll=%ss",
        args.broker, settings.instrument, settings.candle_granularity, settings.poll_interval_seconds,
    )
    logger.info("This is demo/paper trading only. No live orders will ever be placed by this CLI.")

    last_day = None
    try:
        while True:
            candles = data_provider.get_historical_candles(settings.candle_granularity, 400)
            enriched = se.prepare(candles)
            latest_price = data_provider.get_latest_price()

            bar_day = enriched.index[-1].date() if hasattr(enriched.index[-1], "date") else None
            if bar_day is not None and bar_day != last_day:
                risk_mgr.start_of_day(broker.get_account_balance())
                last_day = bar_day

            closed = broker.check_stops(latest_price)
            if closed:
                logger.info(
                    "Position closed [%s]: %s %d units, entry=%.2f exit=%.2f pnl=%+.2f",
                    closed.reason, closed.direction.value, abs(closed.units),
                    closed.entry_price, closed.exit_price, closed.pnl,
                )

            equity = broker.get_account_balance()
            risk_mgr.update_equity(equity)

            open_positions = 1 if broker.get_open_position() else 0
            can_trade, halt_reason = risk_mgr.can_trade(open_positions)

            if not can_trade:
                if halt_reason:
                    logger.warning("Trading halted: %s", halt_reason)
            else:
                sentiment_score = sentiment_provider.get_sentiment().score if sentiment_provider else None
                signal = se.generate_signal(enriched, sentiment_score=sentiment_score)

                if signal.direction is not se.Direction.FLAT:
                    logger.info(
                        "Signal: %s (confidence=%.2f) price=%.2f | %s",
                        signal.direction.value, signal.confidence, signal.price, "; ".join(signal.reasons),
                    )
                    plan = risk_mgr.size_trade(signal, equity)
                    if plan:
                        broker.open_position(plan)
                        logger.info(
                            "Opened %s %d units @ %.2f | stop=%.2f target=%.2f | risking $%.2f",
                            plan.direction.value, abs(plan.units), plan.entry_price,
                            plan.stop_loss, plan.take_profit, plan.risk_amount,
                        )

            logger.info("Equity: %.2f | open_position=%s", equity, bool(open_positions))
            time.sleep(settings.poll_interval_seconds)
    except KeyboardInterrupt:
        logger.info("Demo loop stopped by user. Final equity: %.2f", broker.get_account_balance())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Replay historical candles and report performance.")
    bt.add_argument("--bars", type=int, default=2000)
    bt.add_argument("--granularity", default=settings.candle_granularity)
    bt.set_defaults(func=cmd_backtest)

    demo = sub.add_parser("demo", help="Run the continuous demo/paper-trading loop.")
    demo.add_argument("--broker", choices=["simulated", "oanda"], default="simulated")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
