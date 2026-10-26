# Running unattended

This is scaffolding for actually leaving `python main.py demo` running for a
long stretch unattended, rather than in a terminal you're watching. It does
not change what the bot does -- it's still demo/paper-trading only, with the
exact same risk/circuit-breaker behavior documented in the main README. All
it adds is: restart it if it crashes, and don't let one bug restart-loop
forever.

Pick one of the two options below. Either way, check on it afterwards with
`python main.py status` (see main README) -- that's the thing that doesn't
need you to read raw logs to know whether it's still running and whether a
circuit breaker has tripped.

## Option A: systemd (Linux, if your machine runs it)

1. Create your real `.env` from `.env.example` if you haven't already, and
   make sure none of its values are wrapped in quotes -- `EnvironmentFile`
   (unlike python-dotenv) doesn't strip them.
2. Edit `deploy/systemd/gold-trader.service`: replace every `YOUR_USER` and
   the two `/home/YOUR_USER/GOLD-AUTO-TRADING` paths with your actual
   username and checkout path. Change `--broker simulated` to
   `--broker oanda` if you want it polling your OANDA practice account
   instead of yfinance.
3. Install and start it:
   ```bash
   sudo cp deploy/systemd/gold-trader.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now gold-trader
   ```
4. Check on it:
   ```bash
   systemctl status gold-trader       # is it running, how many times has it restarted
   journalctl -u gold-trader -f       # live log tail
   python main.py status              # the bot's own view of itself
   ```
5. Stop it: `sudo systemctl stop gold-trader` (and `disable` to stop it
   starting on boot).

If it keeps failing to start, `systemctl status` will show
`Start request repeated too quickly` once it hits the 5-failures-in-10-minutes
limit in the unit file -- that's deliberate (see the comments in the unit
file), not a bug in the service file. Fix the underlying problem, then
`systemctl reset-failed gold-trader` before starting it again.

## Option B: plain restart-loop script (any OS with bash)

No systemd, no sudo. Good for macOS, or a quick run on a Linux box you
don't want to configure further. If you've set up a project-local `.venv`
(as the systemd option above assumes), this script activates it
automatically; otherwise it falls back to whatever `python3` is on your
PATH, so make sure that one has `requirements.txt` installed.

```bash
./deploy/run_demo_supervised.sh simulated   # or: oanda
```

Leave that running (in `screen`, `tmux`, `nohup ... &`, or just a terminal
you leave open). It restarts the demo loop on crash with exponential backoff
(5s, 10s, 20s, ... capped at 5 minutes), and gives up entirely -- rather than
restart-looping forever -- if it crashes 5+ times within 10 minutes, since
that means a real bug, not a transient blip like a dropped connection. If it
runs cleanly for at least that same 10-minute window before a later crash,
the backoff and failure count reset, so one old bad patch doesn't leave a
long memory that punishes an unrelated crash months later.

Stop it with Ctrl-C. It won't restart after that (that's the "exited cleanly"
path, the same one a normal Ctrl-C to `main.py demo` itself takes).

## What this does NOT do

- **Does not clear a halted circuit breaker.** If `max_drawdown_pct` or
  `cumulative_drawdown_halt_pct` trips, the process keeps running (it's not
  crashing) but stops opening new trades, by design -- restarting it doesn't
  bypass that review requirement, and nothing here tries to.
- **Does not touch OANDA credentials, live trading, or anything requiring
  you to be at the machine.** Both options run the exact same demo-only
  `main.py demo` command you'd otherwise type yourself.
