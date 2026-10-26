#!/usr/bin/env python3
"""Gold (XAU/USD) automated trading system — demo/paper-trading only.

Usage:
    python main.py backtest     [--bars N] [--granularity M15]
    python main.py walk-forward [--bars N] [--granularity M15] [--folds 6]
    python main.py demo         [--broker simulated|oanda]

`backtest` replays historical candles bar-by-bar through the exact same
signal engine, risk manager, and sizing math the demo loop uses, and
prints a performance report. Run this first and look hard at the numbers
before trusting anything else here.

`walk-forward` is the honest way to ask "does this strategy have real
edge, or did one backtest just get lucky": it splits history into folds,
grid-searches a few config knobs on all but the last fold, then reports
performance ONLY on that final held-out fold, which the search never saw.
See gold_trader/research/walk_forward.py for why this matters.

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

from gold_trader import decision_log
from gold_trader.backtest.backtester import run_backtest
from gold_trader.config import settings
from gold_trader.data.providers import OandaDataProvider, YFinanceDataProvider
from gold_trader.execution.broker import OandaPracticeBroker, SimulatedBroker
from gold_trader.logging_config import setup_logging
from gold_trader.research.walk_forward import run_walk_forward_study
from gold_trader.risk.approval_gate import PortfolioApprovalGate
from gold_trader.risk.risk_manager import RiskManager
from gold_trader.sentiment.cot_data import PointInTimeCotLookup
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


# A small, fixed grid: broad enough to matter, bounded enough that searching it
# isn't itself a curve-fitting exercise. Extending this is fine; extending it
# until some combination "wins" on the held-out fold by chance is exactly the
# mistake this whole module exists to prevent.
WALK_FORWARD_PARAM_GRID = {
    "min_confirmations": [2, 3],
    "atr_stop_multiple": [1.0, 1.5, 2.0],
    "reward_risk_ratio": [1.0, 1.5, 2.0],
    "enable_cot_filter": [False, True],
}


def cmd_walk_forward(args: argparse.Namespace) -> None:
    logger.info("Fetching historical %s candles for walk-forward study...", args.granularity)
    provider = YFinanceDataProvider()
    candles = provider.get_historical_candles(args.granularity, args.bars)
    logger.info("Loaded %d candles (%s -> %s)", len(candles), candles.index[0], candles.index[-1])

    result = run_walk_forward_study(
        candles, n_folds=args.folds, base_config=settings.risk,
        param_grid=WALK_FORWARD_PARAM_GRID, starting_balance=settings.starting_demo_balance,
    )

    print("\n" + "=" * 70)
    print(" DEFAULT CONFIG, EVALUATED ON EVERY FOLD (consistency check)")
    print("=" * 70)
    for fold in result.default_config_folds:
        r = fold.report
        print(f"Fold {fold.fold_index} [{fold.start} -> {fold.end}]: "
              f"return={r.total_return_pct:+.2f}% trades={r.num_trades} "
              f"win_rate={r.win_rate:.1%} profit_factor={r.profit_factor:.2f} sharpe={r.sharpe_like:.2f}")

    print("\n" + "=" * 70)
    print(" IN-SAMPLE GRID SEARCH LEADERBOARD (selection folds only, never the held-out fold)")
    print("=" * 70)
    for entry in result.selected_config_in_sample_leaderboard[:10]:
        rk = entry["rank_key"]
        rk_str = f"sharpe={rk[0]:.2f} pf={rk[1]:.2f}" if rk else "not ranked (too few trades in some fold)"
        print(f"{entry['params']} -> {rk_str} | mean_return={entry['mean_return_pct']:+.2f}% "
              f"mean_trades={entry['mean_trades']:.1f}")

    print("\n" + "=" * 70)
    print(" HELD-OUT FOLD RESULT (the only honest number here)")
    print("=" * 70)
    changed = {
        k: getattr(result.selected_config, k)
        for k in WALK_FORWARD_PARAM_GRID
        if getattr(result.selected_config, k) != getattr(settings.risk, k)
    }
    print(f"Selected config (chosen using in-sample folds only): {changed or 'same as default'}")
    print(f"\nDefault config on held-out fold:\n{result.held_out_default_fold.report.summary()}")
    print(f"Selected config on held-out fold:\n{result.held_out_fold.report.summary()}")
    print(
        "This held-out number is the only one of the three sections above\n"
        "that wasn't influenced by picking a config to look good on this\n"
        "data. If it's not meaningfully better than the default, the grid\n"
        "search found no real edge here -- report that honestly rather than\n"
        "widening the grid until something sticks."
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
        else SimulatedBroker(settings.starting_demo_balance, spread_cost_per_unit=settings.risk.spread_cost_per_unit)
    )
    sentiment_provider = NewsSentimentProvider() if settings.enable_sentiment else None
    cot_lookup = PointInTimeCotLookup() if settings.risk.enable_cot_filter else None
    cot_last_refreshed = time.time()
    gate = PortfolioApprovalGate(
        max_slippage_atr_multiple=settings.risk.max_slippage_atr_multiple,
        regime_lookback=settings.risk.volatility_regime_lookback,
        regime_low_percentile=settings.risk.volatility_regime_low_percentile,
        regime_high_percentile=settings.risk.volatility_regime_high_percentile,
    )

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

            closed = broker.check_stops_bar(latest_price, latest_price)
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
            now_iso = enriched.index[-1].isoformat() if hasattr(enriched.index[-1], "isoformat") else str(enriched.index[-1])

            if not can_trade:
                if halt_reason:
                    logger.warning("Trading halted: %s", halt_reason)
                decision_log.record_decision(
                    timestamp=now_iso, price=latest_price, direction="FLAT", confidence=0.0,
                    reasons=[halt_reason or ""], approved=False, rejection_reason=halt_reason, action="halted",
                )
                logger.info("Equity: %.2f | open_position=%s", equity, bool(open_positions))
                time.sleep(settings.poll_interval_seconds)
                continue

            regime = gate.check_volatility_regime(enriched["atr_14"])
            if not regime.approved:
                logger.info("Skipping this cycle: %s", regime.reason)
                decision_log.record_decision(
                    timestamp=now_iso, price=latest_price, direction="FLAT", confidence=0.0,
                    reasons=[regime.reason or ""], approved=False, rejection_reason=regime.reason, action="rejected",
                )
                logger.info("Equity: %.2f | open_position=%s", equity, bool(open_positions))
                time.sleep(settings.poll_interval_seconds)
                continue

            sentiment_score = sentiment_provider.get_sentiment().score if sentiment_provider else None

            cot_percentile = None
            if cot_lookup is not None:
                if time.time() - cot_last_refreshed > 86_400:  # COT reports update weekly; daily refresh is plenty
                    cot_lookup = PointInTimeCotLookup()
                    cot_last_refreshed = time.time()
                cot_reading = cot_lookup.reading_at(enriched.index[-1])
                cot_percentile = cot_reading.percentile_in_history if cot_reading else None

            signal = se.generate_signal(
                enriched, sentiment_score=sentiment_score, min_confirmations=settings.risk.min_confirmations,
                cot_percentile=cot_percentile,
                cot_extreme_low=settings.risk.cot_extreme_low_percentile,
                cot_extreme_high=settings.risk.cot_extreme_high_percentile,
            )

            if signal.direction is se.Direction.FLAT:
                decision_log.record_decision(
                    timestamp=now_iso, price=signal.price, direction="FLAT", confidence=0.0,
                    reasons=signal.reasons, approved=True, rejection_reason=None, action="flat",
                )
            else:
                logger.info(
                    "Signal: %s (confidence=%.2f) price=%.2f | %s",
                    signal.direction.value, signal.confidence, signal.price, "; ".join(signal.reasons),
                )
                plan = risk_mgr.size_trade(signal, equity)
                approval = gate.check_slippage(plan, latest_price) if plan else None

                if plan is None:
                    decision_log.record_decision(
                        timestamp=now_iso, price=signal.price, direction=signal.direction.value,
                        confidence=signal.confidence, reasons=signal.reasons, approved=False,
                        rejection_reason="risk manager could not size a valid trade", action="rejected",
                    )
                elif not approval.approved:
                    logger.info("Trade rejected by approval gate: %s", approval.reason)
                    decision_log.record_decision(
                        timestamp=now_iso, price=signal.price, direction=signal.direction.value,
                        confidence=signal.confidence, reasons=signal.reasons, approved=False,
                        rejection_reason=approval.reason, action="rejected",
                    )
                else:
                    broker.open_position(plan)
                    risk_mgr.record_trade_opened()
                    logger.info(
                        "Opened %s %d units @ %.2f | stop=%.2f target=%.2f | risking $%.2f",
                        plan.direction.value, abs(plan.units), plan.entry_price,
                        plan.stop_loss, plan.take_profit, plan.risk_amount,
                    )
                    decision_log.record_decision(
                        timestamp=now_iso, price=signal.price, direction=signal.direction.value,
                        confidence=signal.confidence, reasons=signal.reasons, approved=True,
                        rejection_reason=None, action="opened",
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

    wf = sub.add_parser("walk-forward", help="Out-of-sample walk-forward research (see module docstring).")
    wf.add_argument("--bars", type=int, default=5000)
    wf.add_argument("--granularity", default=settings.candle_granularity)
    wf.add_argument("--folds", type=int, default=6)
    wf.set_defaults(func=cmd_walk_forward)

    demo = sub.add_parser("demo", help="Run the continuous demo/paper-trading loop.")
    demo.add_argument("--broker", choices=["simulated", "oanda"], default="simulated")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
