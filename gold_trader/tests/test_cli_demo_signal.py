import signal
import subprocess
import sys
import time
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Runs as a standalone script in a subprocess rather than calling cmd_demo() in-process:
# SIGTERM semantics (does an unhandled signal kill the process? does the handler installed
# in *this* process actually fire?) aren't meaningfully testable without a real separate OS
# process to send a real signal to. Isolated from the real checkout in two ways a first
# version of this test got wrong (caught by review): it monkeypatches LOG_DIR/STATUS_PATH/
# LOG_PATH to a tmp directory before importing main (those modules resolve paths from
# __file__ at import time, not cwd, so without this the subprocess would overwrite the
# developer's real logs/status.json, logs/decision_log.jsonl and logs/gold_trader.log), and
# it stubs out YFinanceDataProvider's network calls so this doesn't depend on Yahoo Finance
# being reachable, unlike the real demo loop.
_SANDBOX_SCRIPT = Template(
    """
import sys
sys.path.insert(0, $repo_root)
from pathlib import Path

import gold_trader.logging_config as logging_config_mod
logging_config_mod.LOG_DIR = Path($log_dir)

from gold_trader import status as status_mod
from gold_trader import decision_log as decision_log_mod
status_mod.STATUS_PATH = Path($log_dir) / "status.json"
decision_log_mod.LOG_PATH = Path($log_dir) / "decision_log.jsonl"

import pandas as pd
from gold_trader.data import providers as providers_mod

def _fake_get_historical_candles(self, granularity, count):
    idx = pd.date_range("2024-01-01", periods=count, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.0, "volume": 100},
        index=idx,
    )

def _fake_get_latest_price(self):
    return 2000.0

providers_mod.YFinanceDataProvider.get_historical_candles = _fake_get_historical_candles
providers_mod.YFinanceDataProvider.get_latest_price = _fake_get_latest_price

import argparse
import main as main_mod
main_mod.cmd_demo(argparse.Namespace(broker="simulated"))
"""
)


def test_sigterm_shuts_down_cleanly_like_ctrl_c(tmp_path, monkeypatch):
    """Regression test for the `signal` (stdlib) vs. `signal` (the local trading
    Signal object in cmd_demo's loop) name collision this fix had to work around:
    the first version of this handler raised UnboundLocalError on the very first
    poll cycle because cmd_demo already binds a local named `signal`. Asserts the
    behavior that matters operationally: `systemctl stop` sends SIGTERM, and this
    needs to exit 0 with a clean log line, not vanish or exit non-zero (which would
    make systemd/the bash supervisor treat a deliberate stop as a crash)."""
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "2")
    script = _SANDBOX_SCRIPT.substitute(repo_root=repr(str(REPO_ROOT)), log_dir=repr(str(tmp_path)))
    script_path = tmp_path / "run_demo.py"
    script_path.write_text(script)

    proc = subprocess.Popen(
        [sys.executable, str(script_path)], cwd=str(tmp_path),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    status_path = tmp_path / "status.json"
    try:
        # Poll for status.json rather than a fixed sleep or a specific log line: this
        # is the one piece of evidence that a full poll cycle actually completed
        # (_write_status is the last thing each cycle does before sleeping), so
        # sending SIGTERM only once it exists means the signal always lands during
        # the sleep/next-cycle window rather than racing the first cycle's setup.
        deadline = time.time() + 10
        while time.time() < deadline and proc.poll() is None and not status_path.exists():
            time.sleep(0.05)
        assert status_path.exists(), "demo loop never completed a first cycle"

        proc.send_signal(signal.SIGTERM)
        stdout, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, f"expected a clean exit, got {proc.returncode}:\n{stdout}"
    assert "stopped by user" in stdout.lower()
    assert "Traceback" not in stdout
    # decision_log.jsonl in the sandbox dir, not status.json alone, is the proof that
    # isolation covered both modules -- status.json's existence was already required
    # above just to know when to send the signal.
    assert (tmp_path / "decision_log.jsonl").exists()
