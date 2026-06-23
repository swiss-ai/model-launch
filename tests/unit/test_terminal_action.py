"""The TUI replica-panel Terminal button: rendering + click → open-terminal action."""

import asyncio

from rich.text import Text

from swiss_ai_model_launch.cli.display.live import _SMLApp, _terminal_cell
from swiss_ai_model_launch.cli.display.state import DisplayState
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealth, ReplicaHealthReport
from swiss_ai_model_launch.launchers.launcher import TerminalCommand


def test_terminal_cell_clickable_when_host_known() -> None:
    cell = _terminal_cell(123, "nid001234", stale=False)
    assert isinstance(cell, Text)
    assert "open" in cell.plain
    assert cell.style.meta["@click"] == "app.open_terminal('123', 'nid001234')"


def test_terminal_cell_is_inert_without_host_or_when_stale() -> None:
    for cell in (
        _terminal_cell(None, "nid1", stale=False),  # no job id
        _terminal_cell(1, None, stale=False),  # no host yet
        _terminal_cell(1, "nid1", stale=True),  # job gone
        _terminal_cell(1, "bad host!", stale=False),  # unsafe host token
    ):
        assert isinstance(cell, Text)
        assert cell.plain == "—"  # a dim placeholder, with no clickable @click meta
        assert cell.style == "dim"


def _report_with_host(host: str) -> ReplicaHealthReport:
    return ReplicaHealthReport("m", 1, (ReplicaHealth(ModelHealth.HEALTHY, "Q", 1, 0, "10.0.0.1", host),))


def _find_clickable(app: _SMLApp) -> tuple[int, int]:
    screen = app.screen
    for y in range(screen.size.height):
        for x in range(screen.size.width):
            if "@click" in screen.get_style_at(x, y).meta:
                return x, y
    raise AssertionError("no clickable terminal cell rendered")


async def test_clicking_terminal_runs_launcher_command() -> None:
    state = DisplayState()
    state.open_terminal = lambda job_id, host: TerminalCommand(
        argv=["echo", f"{job_id}:{host}"], display=f"echo {job_id}:{host}"
    )
    state.update(job_id=99)
    state.set_replica_report(_report_with_host("nid7"))

    opened: list[TerminalCommand] = []
    app = _SMLApp(state, asyncio.sleep(3600))
    app._open_terminal = opened.append  # type: ignore[method-assign]  # capture instead of suspending

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(offset=_find_clickable(app))
        await pilot.pause()

    assert len(opened) == 1
    assert opened[0].argv == ["echo", "99:nid7"]  # job id + host plumbed through


async def test_clicking_unavailable_terminal_falls_back_to_copy() -> None:
    state = DisplayState()
    state.open_terminal = lambda job_id, host: TerminalCommand(
        argv=[], display="srun ...", available=False, reason="No cluster SSH host is configured."
    )
    state.update(job_id=42)
    state.set_replica_report(_report_with_host("nid7"))

    offered: list[TerminalCommand] = []
    app = _SMLApp(state, asyncio.sleep(3600))
    app._offer_terminal_command = offered.append  # type: ignore[method-assign]

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(offset=_find_clickable(app))
        await pilot.pause()

    assert len(offered) == 1
    assert not offered[0].available


async def test_open_terminal_action_without_factory_is_a_noop() -> None:
    # state.open_terminal stays None (no monitor wired it) -> the action must warn,
    # not raise, even if a click somehow reaches it.
    state = DisplayState()
    app = _SMLApp(state, asyncio.sleep(3600))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_open_terminal("1", "nid1")  # no factory set
        await pilot.pause()
