import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_sigterm_shuts_down_cleanly_like_ctrl_c():
    """Regression test for the `signal` (stdlib) vs. `signal` (the local trading
    Signal object in cmd_demo's loop) name collision this fix had to work around:
    the first version of this handler raised UnboundLocalError on the very first
    poll cycle because cmd_demo already binds a local named `signal`. Runs the real
    CLI as a subprocess (SIGTERM semantics -- "the OS kills this unless the process
    explicitly handles it" -- aren't meaningfully testable by calling a function
    in-process) and asserts the specific behavior that matters operationally:
    `systemctl stop` sends SIGTERM, and this needs to exit 0 with a clean log line,
    not vanish or exit non-zero (which would make systemd/the bash supervisor treat
    a deliberate stop as a crash)."""
    env = {**os.environ, "POLL_INTERVAL_SECONDS": "2"}
    proc = subprocess.Popen(
        [sys.executable, "main.py", "demo", "--broker", "simulated"],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        time.sleep(3)  # let it start up and complete at least one poll cycle
        proc.send_signal(signal.SIGTERM)
        stdout, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, f"expected a clean exit, got {proc.returncode}:\n{stdout}"
    assert "stopped by user" in stdout.lower()
    assert "Traceback" not in stdout
