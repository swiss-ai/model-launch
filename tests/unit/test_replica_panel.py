import time

import pytest
from rich.console import Console

from swiss_ai_model_launch.cli.display.live import _render_replica_panel
from swiss_ai_model_launch.cli.display.state import DisplayState
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealth, ReplicaHealthReport


def _render(state: DisplayState) -> str:
    console = Console(width=120, record=True)
    console.print(_render_replica_panel(state))
    return console.export_text()


def test_panel_waiting_state() -> None:
    assert "Waiting for the model" in _render(DisplayState())


def test_panel_in_progress() -> None:
    state = DisplayState()
    state.set_replica_check_in_progress()
    assert "Checking replicas" in _render(state)


def test_panel_table_error() -> None:
    state = DisplayState()
    state.set_replica_report(ReplicaHealthReport("m", 2, (), table_error="connection refused"))
    assert "connection refused" in _render(state)


def test_panel_no_replicas() -> None:
    state = DisplayState()
    state.set_replica_report(ReplicaHealthReport("m", 2, ()))
    assert "No replicas registered" in _render(state)


def test_panel_lists_each_replica_with_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 2000.0)  # freeze the wall clock
    state = DisplayState()
    state.set_replica_report(
        ReplicaHealthReport(
            "m",
            3,
            (
                ReplicaHealth("QmReplicaAAA", ModelHealth.HEALTHY, last_seen=1995),
                ReplicaHealth("QmReplicaBBB", ModelHealth.HEALTHY, last_seen=1990),
                ReplicaHealth("QmReplicaCCC", ModelHealth.NOT_RESPONDING, last_seen=1980),
            ),
            checked_at=2000,
        )
    )
    out = _render(state)
    assert "2/3 healthy" in out  # summary reflects only HEALTHY replicas
    for peer_id in ("QmReplicaAAA", "QmReplicaBBB", "QmReplicaCCC"):
        assert peer_id in out
    assert "NOT RESPONDING" in out
    assert "Last Heartbeat" in out
    # heartbeat age is rendered relative to the live clock (frozen here at 2000)
    for ago in ("5s ago", "10s ago", "20s ago"):
        assert ago in out
