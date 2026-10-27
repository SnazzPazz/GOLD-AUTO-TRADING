import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "deploy" / "run_demo_supervised.py"


class FakeClock:
    """Standalone fake for the `time` module calls run_demo_supervised.py makes.
    advance() simulates a subprocess run taking some wall-clock duration (not
    recorded); sleep() is the supervisor's own backoff wait (recorded in
    sleep_calls, so tests can assert on the exact backoff sequence)."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds

    def strftime(self, fmt, t) -> str:
        return "2024-01-01T00:00:00Z"

    def gmtime(self):
        return None


def _load_module(run_sequence: list[tuple[float, int]]) -> tuple[object, FakeClock]:
    """run_sequence: list of (simulated_run_duration, exit_code), consumed in
    order by each call to subprocess.run. Raises IndexError (failing the test
    loudly) if main() tries to launch more times than scripted."""
    spec = importlib.util.spec_from_file_location("run_demo_supervised_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    clock = FakeClock()
    module.time = clock
    remaining = list(run_sequence)

    def fake_run(cmd, cwd=None):
        duration, exit_code = remaining.pop(0)
        clock.advance(duration)
        return SimpleNamespace(returncode=exit_code)

    module.subprocess.run = fake_run
    return module, clock


def test_clean_exit_does_not_restart():
    module, clock = _load_module([(0, 0)])
    module.main()
    assert clock.sleep_calls == []


def test_restarts_with_exponential_backoff_then_gives_up_after_five_failures():
    module, clock = _load_module([(0, 1)] * 5)
    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 1
    # 4 backoff waits between the 5 failures -- doubling from the 5s default,
    # no wait after the 5th since it gives up immediately instead.
    assert clock.sleep_calls == [5, 10, 20, 40]


def test_backoff_caps_at_max_backoff_seconds():
    module, clock = _load_module([(0, 1)] * 5)
    module.INITIAL_BACKOFF_SECONDS = 100
    module.MAX_BACKOFF_SECONDS = 150
    with pytest.raises(SystemExit):
        module.main()
    # 100 -> 200 would exceed the 150 cap, so it clamps instead of doubling freely.
    assert clock.sleep_calls == [100, 150, 150, 150]


def test_backoff_and_failure_count_reset_after_a_long_successful_run():
    """The core property the whole reset exists for: a run that lasted longer
    than the failure window counts as recovered, so the next failure starts
    the backoff over from the initial value instead of continuing to escalate
    from whatever it reached before."""
    module, clock = _load_module([
        (0, 1),    # fails fast -> sleep 5, backoff -> 10
        (0, 1),    # fails fast -> sleep 10, backoff -> 20
        (700, 1),  # ran long enough (>= 600s window) to reset -> sleep 5 again
        (0, 0),    # clean exit, stop
    ])
    module.main()
    assert clock.sleep_calls == [5, 10, 5]


def test_python_executable_prefers_project_venv(tmp_path, monkeypatch):
    module, _ = _load_module([(0, 0)])
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("#!/bin/sh\n")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    assert module._python_executable() == str(venv_python)


def test_python_executable_falls_back_to_sys_executable_when_no_venv(tmp_path, monkeypatch):
    module, _ = _load_module([(0, 0)])
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)  # empty dir, no .venv
    assert module._python_executable() == module.sys.executable
