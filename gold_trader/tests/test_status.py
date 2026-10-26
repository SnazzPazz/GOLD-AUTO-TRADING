import time

from gold_trader import status


def make_snapshot(**overrides) -> status.StatusSnapshot:
    defaults = dict(
        updated_at=time.time(),
        broker="simulated",
        instrument="XAU_USD",
        granularity="H1",
        poll_interval_seconds=60,
        cycle_count=1,
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


def test_write_then_read_round_trips_all_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "STATUS_PATH", tmp_path / "status.json")
    snap = make_snapshot(equity=9_500.5, cycle_count=42, last_action="opened",
                          last_signal_direction="LONG", last_signal_confidence=0.8)
    status.write_status(snap)

    loaded = status.read_status()
    assert loaded == snap


def test_read_status_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "STATUS_PATH", tmp_path / "does_not_exist.json")
    assert status.read_status() is None


def test_write_status_overwrites_previous_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "STATUS_PATH", tmp_path / "status.json")
    status.write_status(make_snapshot(cycle_count=1))
    status.write_status(make_snapshot(cycle_count=2))

    loaded = status.read_status()
    assert loaded.cycle_count == 2


def test_is_stale_false_when_recently_updated():
    snap = make_snapshot(updated_at=time.time(), poll_interval_seconds=60)
    assert status.is_stale(snap, now=time.time() + 30) is False


def test_is_stale_true_well_past_poll_multiple():
    snap = make_snapshot(updated_at=1000.0, poll_interval_seconds=60)
    # 3x poll interval is the threshold; 10 minutes later is well past it.
    assert status.is_stale(snap, now=1000.0 + 600) is True


def test_is_stale_boundary_uses_poll_multiple_of_three():
    snap = make_snapshot(updated_at=1000.0, poll_interval_seconds=60)
    assert status.is_stale(snap, now=1000.0 + 179) is False  # just under 3x60=180
    assert status.is_stale(snap, now=1000.0 + 181) is True   # just over
