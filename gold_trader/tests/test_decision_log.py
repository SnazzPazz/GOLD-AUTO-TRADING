import json

from gold_trader import decision_log


def _record(path, **overrides):
    decision_log.LOG_PATH = path
    defaults = dict(
        timestamp="2024-01-01T00:00:00Z", price=2000.0, direction="LONG", confidence=0.8,
        reasons=["trend", "momentum"], approved=True, rejection_reason=None, action="opened",
    )
    defaults.update(overrides)
    decision_log.record_decision(**defaults)


def test_record_decision_writes_one_valid_json_line(tmp_path, monkeypatch):
    path = tmp_path / "decision_log.jsonl"
    _record(path)

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["direction"] == "LONG"
    assert entry["price"] == 2000.0
    assert entry["approved"] is True
    assert "logged_at" in entry


def test_multiple_calls_append_rather_than_overwrite(tmp_path):
    path = tmp_path / "decision_log.jsonl"
    _record(path, action="flat", direction="FLAT")
    _record(path, action="opened", direction="SHORT")

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "flat"
    assert json.loads(lines[1])["action"] == "opened"


def test_switching_log_path_repoints_the_handler(tmp_path):
    """Regression guard for the indirection in _get_logger(): without it, the
    handler created for the first LOG_PATH would silently keep writing there
    even after LOG_PATH changes (exactly the failure mode that made this
    module hard to test before _get_logger existed)."""
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    _record(first_path, action="flat")
    _record(second_path, action="opened")

    assert len(first_path.read_text().splitlines()) == 1
    assert len(second_path.read_text().splitlines()) == 1
    assert json.loads(second_path.read_text().splitlines()[0])["action"] == "opened"


def test_log_directory_created_if_missing(tmp_path):
    path = tmp_path / "nested" / "dir" / "decision_log.jsonl"
    _record(path)
    assert path.exists()


def test_rotation_caps_active_file_size_and_keeps_a_backup(tmp_path, monkeypatch):
    path = tmp_path / "decision_log.jsonl"
    monkeypatch.setattr(decision_log, "_MAX_BYTES", 500)
    monkeypatch.setattr(decision_log, "_BACKUP_COUNT", 2)
    decision_log.LOG_PATH = path  # a path not used by any other test -> forces a fresh handler

    for i in range(200):
        decision_log.record_decision(
            timestamp=f"2024-01-01T00:00:{i:02d}Z", price=2000.0 + i, direction="LONG",
            confidence=0.8, reasons=["trend"] * 5, approved=True, rejection_reason=None,
            action="opened",
        )

    assert path.exists()
    assert path.stat().st_size <= 500 + 300  # the one line that tipped it over rotation
    assert (tmp_path / "decision_log.jsonl.1").exists()  # rotation actually produced a backup
