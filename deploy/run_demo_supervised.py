#!/usr/bin/env python3
"""Cross-platform restart-loop wrapper around `python main.py demo`.

Exists mainly for Windows (via Task Scheduler), where the bash version
(run_demo_supervised.sh) doesn't apply. Mirrors its restart/backoff/give-up
logic exactly, so there's one behavior to reason about across platforms, not
two subtly different reimplementations. See deploy/README.md for setup.

Restarts on crash with exponential backoff. Gives up -- rather than
restart-looping forever -- if it's crashing repeatedly in a short window,
since that means a real bug, not a transient blip (a dropped connection,
yfinance hiccup, etc).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_FAILURES_IN_WINDOW = 5
FAILURE_WINDOW_SECONDS = 600
INITIAL_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 300


def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[supervisor] {stamp} {msg}", flush=True)


def _python_executable() -> str:
    # If a project-local venv exists (deploy/systemd/gold-trader.service assumes
    # one), use it directly -- otherwise fall back to whatever Python is running
    # this script. Without this, pointing Task Scheduler at this script instead of
    # the venv's own python.exe would fail with ModuleNotFoundError on every
    # restart attempt rather than actually running the bot.
    for candidate in (REPO_ROOT / ".venv" / "Scripts" / "python.exe", REPO_ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main() -> None:
    broker = sys.argv[1] if len(sys.argv) > 1 else "simulated"
    python_exe = _python_executable()
    backoff_seconds = INITIAL_BACKOFF_SECONDS
    failure_times: list[float] = []

    _log(f"Starting supervised demo loop (broker={broker}, python={python_exe}). Ctrl-C to stop for good.")

    while True:
        start_time = time.time()
        _log(f"Launching: {python_exe} main.py demo --broker {broker}")
        try:
            result = subprocess.run([python_exe, "main.py", "demo", "--broker", broker], cwd=str(REPO_ROOT))
        except KeyboardInterrupt:
            _log("Supervisor interrupted (Ctrl-C). Not restarting.")
            return
        exit_code = result.returncode
        now = time.time()
        run_duration = now - start_time

        if exit_code == 0:
            _log("Demo loop exited cleanly (exit 0, e.g. Ctrl-C). Not restarting.")
            return

        _log(f"Demo loop exited with code {exit_code} after running for {run_duration:.0f}s.")

        # Ran long enough to count as recovered: don't let a stale backoff/failure
        # count from a past bad patch linger forever and punish an unrelated,
        # much-later crash with a huge initial delay.
        if run_duration >= FAILURE_WINDOW_SECONDS:
            backoff_seconds = INITIAL_BACKOFF_SECONDS
            failure_times = []

        failure_times.append(now)
        cutoff = now - FAILURE_WINDOW_SECONDS
        failure_times = [t for t in failure_times if t >= cutoff]

        if len(failure_times) >= MAX_FAILURES_IN_WINDOW:
            _log(
                f"Failed {len(failure_times)} times in the last {FAILURE_WINDOW_SECONDS}s. Giving up -- "
                "this looks like a real bug, not a transient blip. Check logs/gold_trader.log, fix it, "
                "then re-run this script."
            )
            sys.exit(1)

        _log(f"Restarting in {backoff_seconds}s...")
        time.sleep(backoff_seconds)
        backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)


if __name__ == "__main__":
    main()
