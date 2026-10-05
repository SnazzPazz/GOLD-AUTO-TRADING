# GOLD-AUTO-TRADING

An automated, demo/paper-trading system for Gold (XAU/USD): candlestick
pattern recognition, technical indicators, a free news-sentiment filter,
strict risk management, a bar-by-bar backtester, and a continuous demo
loop that trades a simulated or real OANDA **practice** account.

## Read this first — what this is and isn't

- **This is demo/paper-trading only.** No code path in this repository
  places a live order with real money. `gold_trader/config.py` refuses to
  resolve the OANDA "live" API host unless `LIVE_TRADING_CONFIRMED=yes` is
  set, and even then no live-order execution class exists — only
  `OandaPracticeBroker` and `SimulatedBroker`, both demo-only.
- **No algorithm can guarantee you won't lose money on margin.** Gold is
  volatile and leverage magnifies both gains and losses. What this system
  actually does to protect capital is bound the damage: a fixed % risked
  per trade, ATR-based stops, a daily loss circuit breaker, and a
  max-drawdown circuit breaker that halts trading entirely until you
  review it (see **Risk management** below). That reduces risk; it does
  not eliminate it.
- **Backtests are not proof of future performance.** They tell you
  whether the strategy would have worked on the history you tested, under
  realistic risk constraints. Markets change. Use the backtester and the
  demo loop together, for a meaningful stretch of time, before trusting
  this with real capital.
- **Same-bar stop/target ambiguity always resolves to the worst case.**
  Candle data doesn't tell you whether price touched the stop or the
  target *first* within a bar. `SimulatedBroker.check_stops_bar` always
  resolves the stop-loss side first when both are inside the same bar's
  range, for both longs and shorts — it never assumes the best case
  happened. Earlier versions of this got that wrong specifically for
  shorts (checked low before high regardless of position direction),
  which silently inflated short-trade win rate in backtests; that's fixed.
- **Backtests now charge a spread cost** (`SPREAD_COST_PER_UNIT`) on every
  simulated close, since a strategy that trades often can see its edge
  erased by real bid/ask spread alone. Treat the backtest number as the
  honest floor, not the live result — real spreads vary and widen around
  news.

## Why gold, why OANDA

- OANDA's free **practice** account streams the exact same live market
  prices as a real account — real price action, fake money. That's the
  "demo that mirrors the market exactly" feature: run `demo --broker
  oanda` and you're trading real XAU/USD moves on a sandboxed balance.
- It's a pure REST API over HTTPS, so it runs anywhere Python runs — no
  MetaTrader terminal, no Windows/Wine requirement.
- The system is intentionally single-instrument (`XAU_USD`). There's no
  multi-asset scanning, correlation logic, or portfolio code — everything
  here assumes gold and only gold.

## Architecture

```
gold_trader/
  config.py            Settings from .env (risk params, OANDA creds, safety gates)
  data/providers.py     YFinanceDataProvider (free, no key, backtesting)
                         OandaDataProvider (live candles/pricing)
  analysis/
    indicators.py        EMA/RSI/MACD/Bollinger/ATR/ADX, pure pandas+numpy
    candlestick_patterns.py  Hammer, engulfing, star, piercing/dark-cloud, doji
  sentiment/news_feed.py  Free gold-news RSS + VADER scoring (secondary filter only)
  strategy/signal_engine.py  Combines trend + momentum + pattern + sentiment,
                              requires >=2 independent confirmations before
                              signaling LONG/SHORT
  risk/risk_manager.py   Position sizing (risk % / ATR stop), daily-loss and
                          max-drawdown circuit breakers, leverage cap
  execution/broker.py    SimulatedBroker (local ledger) and
                          OandaPracticeBroker (real demo-account orders)
  backtest/backtester.py  Bar-by-bar walk-forward replay, no lookahead
  tests/                 pytest unit tests for all of the above
main.py                 CLI: `backtest` and `demo` subcommands
```

Price action is the primary signal. Candlestick patterns, trend (EMA50 vs
EMA200), and momentum (RSI/MACD) each cast an independent "vote"; a trade
only fires when at least two agree. News/social sentiment, when enabled,
can only *dampen* a signal that strongly contradicts it — it never
originates a trade by itself, because free sentiment data is noisy and
often lags price.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # fill in OANDA creds only if you want --broker oanda
```

OANDA practice account (optional, only for `demo --broker oanda`): sign up
free at https://www.oanda.com/demo-account/tpa/personal_info, then put
your API token and account ID into `.env`.

### A note on this specific environment's network access

If you're running this inside a Claude Code cloud session and a command
fails with a proxy/connection error reaching Yahoo Finance, OANDA, or the
news RSS feeds, it's because that session's network policy doesn't allow
the host yet, not a bug in the code. Open the environment's **Network
access** settings (cloud environment menu → Edit) and add the host under
Allowed domains, or run this on your own machine where normal internet
access applies.

## Running it

**Backtest first — always.** This replays historical candles bar-by-bar
through the exact same signal engine, risk manager, and position-sizing
math the live demo loop uses:

```bash
python main.py backtest --bars 2000 --granularity M15
```

Prints a report (win rate, profit factor, max drawdown, Sharpe-like
ratio) and writes the equity curve to `reports/latest_backtest.csv`. Look
hard at these numbers — a strategy with a poor profit factor or deep
drawdown on history is not one to trust live, demo or otherwise.

**Then the continuous demo loop**, either fully local:

```bash
python main.py demo --broker simulated
```

or against your OANDA practice account (real live prices, fake money):

```bash
python main.py demo --broker oanda
```

Every signal, trade, and circuit-breaker event is logged to the console
and to `logs/gold_trader.log` so you can audit exactly why each decision
was made.

## Risk management defaults (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `RISK_PER_TRADE_PCT` | 1% | Fraction of account balance risked on one trade's stop distance |
| `ATR_STOP_MULTIPLE` | 1.5 | Stop-loss placed this many ATRs from entry |
| `REWARD_RISK_RATIO` | 1.5 | Take-profit distance relative to stop distance |
| `MAX_DAILY_LOSS_PCT` | 3% | Halts new trades for the rest of the day past this loss |
| `MAX_DRAWDOWN_PCT` | 10% | Halts trading entirely (manual restart) past this drawdown from peak equity |
| `MAX_CONCURRENT_POSITIONS` | 1 | Gold-only, so no pyramiding by default |
| `MAX_LEVERAGE` | 10x | Hard cap on notional exposure vs. account balance |
| `MAX_TRADES_PER_DAY` | 5 | Caps new positions per day to limit whipsaw in choppy conditions |
| `SPREAD_COST_PER_UNIT` | 0.30 | Approximate round-trip bid/ask cost charged on every simulated close, so results aren't flattered by frictionless fills |
| `MIN_CONFIRMATIONS` | 2 | Independent signals (trend/momentum/pattern/Bollinger) required to agree before a trade fires |

These are conservative starting points, not tuned recommendations — adjust
them deliberately, and re-run the backtester after any change.

## Tests

```bash
python -m pytest gold_trader/tests/ -v
```

## Extending this

- **Live trading**: not implemented, by design (see above). If you decide
  to add it yourself later, treat it as a new, carefully reviewed broker
  class behind its own explicit confirmation — don't repurpose
  `OandaPracticeBroker`.
- **Better sentiment**: swap `NewsSentimentProvider` for a paid feed
  (NewsAPI, X/Twitter API, Reddit API) by implementing the same
  `get_sentiment()` interface; nothing else needs to change.
- **TradingView as a signal source**: TradingView itself isn't a broker —
  a common pattern is a Pine Script alert → webhook → a small Flask
  endpoint that feeds the payload into `strategy/signal_engine.py`. Not
  built here, but the signal engine's interface is the natural seam.
