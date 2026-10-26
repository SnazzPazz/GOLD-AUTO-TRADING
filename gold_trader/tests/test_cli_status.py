import argparse
import time

import main
from gold_trader import status


def make_snapshot(**overrides) -> status.StatusSnapshot:
    defaults = dict(
        updated_at=time.time(),
        broker="simulated",
        instrument="XAU_USD",
        granularity="H1",
        poll_interval_seconds=60,
        cycle_count=5,
        equity=10_000.0,
        open_position=False,
        can_trade=True,
        halt_reason=None,
        halted_for_day=False,
        halted_permanently=False,
        halted_catastrophically=False,
        last_action="flat",
        last_signal_direction=None,
        last_signal_confidence=None,
        trades_today=0,
    )
    defaults.update(overrides)
    return status.StatusSnapshot(**defaults)


def test_no_status_file_prints_guidance(monkeypatch, capsys):
    monkeypatch.setattr(status, "read_status", lambda: None)
    main.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "No status file found" in out
    assert "python main.py demo" in out


def test_healthy_running_shows_core_fields_and_no_halt_section(monkeypatch, capsys):
    snap = make_snapshot(equity=10_500.25, cycle_count=42, can_trade=True)
    monkeypatch.setattr(status, "read_status", lambda: snap)
    main.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out

    assert "(healthy)" in out
    assert "10500.25" in out
    assert "Cycle count:     42" in out
    assert "Halt reason:" not in out  # only shown when can_trade is False


def test_stale_snapshot_shows_warning(monkeypatch, capsys):
    snap = make_snapshot(updated_at=time.time() - 10_000, poll_interval_seconds=60)
    monkeypatch.setattr(status, "read_status", lambda: snap)
    main.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "STALE" in out


def test_signal_direction_and_confidence_shown_when_present(monkeypatch, capsys):
    snap = make_snapshot(last_action="opened", last_signal_direction="LONG", last_signal_confidence=0.73)
    monkeypatch.setattr(status, "read_status", lambda: snap)
    main.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "LONG" in out
    assert "0.73" in out


def test_daily_halt_shows_reason_and_flags_but_no_special_messages(monkeypatch, capsys):
    snap = make_snapshot(can_trade=False, halted_for_day=True, halt_reason="Daily loss circuit breaker: down 4.0%")
    monkeypatch.setattr(status, "read_status", lambda: snap)
    main.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "Daily loss circuit breaker" in out
    assert "halted_for_day=True" in out
    assert "clears automatically at the next calendar day" in out.lower()
    assert "acknowledge_catastrophic_loss_and_reset" not in out


def test_permanent_halt_shows_review_and_reset_guidance(monkeypatch, capsys):
    snap = make_snapshot(can_trade=False, halted_permanently=True, halt_reason="Max drawdown circuit breaker")
    monkeypatch.setattr(status, "read_status", lambda: snap)
    main.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "halted_permanently=True" in out
    assert "review_and_reset" in out
    assert "acknowledge_catastrophic_loss_and_reset" not in out


def test_catastrophic_halt_shows_the_hard_halt_guidance(monkeypatch, capsys):
    snap = make_snapshot(
        can_trade=False, halted_permanently=True, halted_catastrophically=True,
        halt_reason="Cumulative drawdown circuit breaker",
    )
    monkeypatch.setattr(status, "read_status", lambda: snap)
    main.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "halted_catastrophically=True" in out
    assert "acknowledge_catastrophic_loss_and_reset" in out
