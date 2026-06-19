import time

import pytest
from rich.console import Console

from swiss_ai_model_launch.cli.display.live import _render_replica_panel
from swiss_ai_model_launch.cli.display.state import DisplayState
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealth, ReplicaHealthReport


def _render(state: DisplayState, blink_on: bool = False) -> str:
    console = Console(width=140, record=True)
    console.print(_render_replica_panel(state, blink_on))
    return console.export_text()


def test_panel_waiting_state() -> None:
    assert "Waiting" in _render(DisplayState())


def test_panel_report_error() -> None:
    state = DisplayState()
    state.set_replica_report(ReplicaHealthReport("m", 2, (), error="report unreadable"))
    assert "report unreadable" in _render(state)


def test_panel_no_replicas() -> None:
    state = DisplayState()
    state.set_replica_report(ReplicaHealthReport("m", 2, ()))
    assert "No replicas reported" in _render(state)


def test_panel_lists_each_replica_with_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 2000.0)  # freeze the wall clock
    state = DisplayState()
    state.set_replica_report(
        ReplicaHealthReport(
            "m",
            3,
            (
                ReplicaHealth(ModelHealth.HEALTHY, "QmReplicaAAA", 1995, 0, "10.0.0.1"),
                ReplicaHealth(ModelHealth.HEALTHY, "QmReplicaBBB", 1990, 1, "10.0.0.2"),
                ReplicaHealth(ModelHealth.NOT_RESPONDING, None, 1980, 2, "10.0.0.3"),
            ),
            checked_at=2000,
        )
    )
    out = _render(state)
    assert "2/3 healthy" in out  # summary reflects only HEALTHY replicas
    assert "Node IP" in out
    for token in ("10.0.0.1", "10.0.0.3", "NOT RESPONDING"):
        assert token in out
    # heartbeat age is rendered relative to the live clock (frozen here at 2000)
    for ago in ("5s ago", "10s ago", "20s ago"):
        assert ago in out


def test_panel_shows_all_running_jobs_during_handover() -> None:
    state = DisplayState()
    # Overlapping handover: old job fully healthy, new job still spinning up.
    state.set_replica_reports(
        [
            (
                101,
                ReplicaHealthReport(
                    "m",
                    2,
                    (
                        ReplicaHealth(ModelHealth.HEALTHY, "QmA", 1995, 0, "10.0.0.1"),
                        ReplicaHealth(ModelHealth.HEALTHY, "QmB", 1990, 1, "10.0.0.2"),
                    ),
                ),
            ),
            (
                102,
                ReplicaHealthReport(
                    "m",
                    2,
                    (
                        ReplicaHealth(ModelHealth.HEALTHY, "QmC", 1996, 0, "10.0.0.3"),
                        ReplicaHealth(ModelHealth.NOT_DEPLOYED, None, None, 1, "10.0.0.4"),
                    ),
                ),
            ),
        ]
    )
    out = _render(state)
    # Both jobs are labelled with their id and per-job healthy summary.
    assert "Job 101 — " in out
    assert "Job 102 — " in out
    assert "2/2 healthy" in out  # old job
    assert "1/2 healthy" in out  # new job still coming up
    # Replicas from both jobs are shown.
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"):
        assert ip in out


def test_reports_take_precedence_over_single_report() -> None:
    state = DisplayState()
    state.set_replica_report(ReplicaHealthReport("m", 1, (ReplicaHealth(ModelHealth.HEALTHY, "QmX", 1, 0, "9.9.9.9"),)))
    state.set_replica_reports(
        [(101, ReplicaHealthReport("m", 1, (ReplicaHealth(ModelHealth.HEALTHY, "QmY", 1, 0, "10.0.0.1"),)))]
    )
    out = _render(state)
    assert "Job 101 — " in out
    assert "10.0.0.1" in out
    assert "9.9.9.9" not in out


def test_healthy_replicas_blink_a_green_heart() -> None:
    state = DisplayState()
    state.set_replica_report(
        ReplicaHealthReport(
            "m",
            2,
            (
                ReplicaHealth(ModelHealth.HEALTHY, "QmReplicaAAA", 1995, 0, "10.0.0.1"),
                ReplicaHealth(ModelHealth.NOT_RESPONDING, None, 1980, 1, "10.0.0.2"),
            ),
        )
    )
    # The heart shows beside HEALTHY only on the "blink on" tick, never beside
    # non-healthy replicas.
    on = _render(state, blink_on=True)
    off = _render(state, blink_on=False)
    assert "💚" in on
    assert on.count("💚") == 1  # only the HEALTHY replica gets one
    assert "💚" not in off
