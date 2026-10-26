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
- **No LLM agents here, by design.** Frameworks like
  [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
  use a swarm of LLM-powered agents (analysts, bull/bear researchers, a
  trader, a risk-management team, a portfolio manager) that debate every
  decision. That's a different kind of system — it means an LLM API call,
  real cost, and added latency on every signal. This project borrows the
  *structural* ideas below without taking on that cost; if you want actual
  LLM-driven reasoning layered on top later, that's a separate, deliberate
  decision to make, not something to add quietly.

## What this borrowed from TradingAgents, and why

A few ideas from that framework's architecture and changelog turned out to
apply directly here, without needing any LLM:

- **Point-in-time data integrity.** TradingAgents' changelog lists look-ahead
  fixes across nearly every data type it touches (news, social, fundamentals,
  macro) — the discipline that a backtest must only ever see what was
  actually knowable at that point in time. This project had exactly that bug:
  the backtester never passed a sentiment score into the signal engine at
  all, while the live demo loop always did, so the backtest never validated
  the sentiment-dampening path the demo loop actually runs. Fixed by
  `sentiment/history.py`: every live sentiment reading gets recorded with its
  timestamp, and a backtest run afterward can replay the system's own
  point-in-time sentiment history instead of silently skipping it. There's no
  honest way to backdate sentiment before this file started recording — free
  RSS feeds have no historical archive — so a backtest over an earlier period
  still runs with no sentiment input, which is correct, not a bug.
- **A final approval gate before execution**, mirroring the
  Risk-Management-team-then-Portfolio-Manager sign-off that sits between
  TradingAgents' Trader proposal and actual execution. `risk/approval_gate.py`
  adds two checks neither the signal engine nor the risk manager made before:
  a **volatility-regime filter** (skip trading when ATR is abnormally low —
  too quiet for these patterns to mean anything — or abnormally high — an
  unusual event where a standard stop multiple may not hold) and a
  **slippage/staleness guard** (reject a trade if the live price has already
  drifted too far from the signal's price by the time an order would go in,
  since this system polls rather than streams).
- **A persistent decision log** (`logs/decision_log.jsonl`), one line per
  evaluation cycle in the live demo loop — not just the cycles that became
  trades. Mirrors TradingAgents' own decision log: an auditable trail of
  every signal, rejection, and halt with its reasoning, so "is this
  reliable" can be checked against the record instead of a summary.

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
  risk/
    risk_manager.py       Position sizing (risk % / ATR stop), daily-loss and
                           max-drawdown circuit breakers, leverage cap
    approval_gate.py       Final pre-execution checks: volatility regime,
                            slippage/staleness (see "What this borrowed...")
  sentiment/history.py    Point-in-time sentiment log + lookup, for backtesting
                           the sentiment path honestly (no retroactive guessing)
  decision_log.py         Append-only JSONL record of every live-loop evaluation
  execution/broker.py    SimulatedBroker (local ledger) and
                          OandaPracticeBroker (real demo-account orders)
  backtest/backtester.py  Bar-by-bar replay, no lookahead
  research/walk_forward.py  Out-of-sample parameter research: grid search
                              on in-sample folds, judged only on a held-out
                              fold the search never saw
  tests/                 pytest unit tests for all of the above
main.py                 CLI: `backtest`, `walk-forward`, and `demo` subcommands
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

**Then walk-forward research — before trusting any single backtest number.**
One backtest on one window tells you almost nothing: it's one sample, and
tuning parameters against that same window to make the number look better
is curve-fitting, not research. `walk-forward` does it the honest way: it
splits history into folds, grid-searches a small set of config knobs using
every fold *except the last*, then reports performance only on that final
held-out fold — the only fold the search never touched.

```bash
python main.py walk-forward --bars 5000 --granularity H1 --folds 6
```

Three sections print: the shipped default config's performance on every
fold (a consistency check — wildly different results fold-to-fold means
the strategy is fragile, not just unlucky once), the in-sample leaderboard
(for transparency — a losing search should be visible, not hidden), and
the held-out fold result, which is the only number in the whole report
that wasn't influenced by picking a config to look good on this data. If
the held-out result isn't meaningfully better than the default, that's the
honest answer — report it, don't widen the grid until something sticks.

**Then the continuous demo loop**, either fully local:

```bash
python main.py demo --broker simulated
```

or against your OANDA practice account (real live prices, fake money):

```bash
python main.py demo --broker oanda
```

Every signal, trade, and circuit-breaker event is logged to the console
and to `logs/gold_trader.log` in human-readable form, and every evaluation
cycle (including the ones that didn't trade — flat signals, regime
rejections, slippage rejections, halts) is also appended to
`logs/decision_log.jsonl` as structured records, so you can audit exactly
why each decision was or wasn't made.

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
| `MAX_SLIPPAGE_ATR_MULTIPLE` | 0.5 | Rejects a trade if live price has drifted more than this many stop-distances from the signal price (stale-signal guard) |
| `VOLATILITY_REGIME_LOOKBACK` | 100 | Bars of ATR history used to judge "normal" volatility |
| `VOLATILITY_REGIME_LOW/HIGH_PERCENTILE` | 10 / 95 | Rejects trades when current ATR falls outside this percentile band of its own recent history |

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
