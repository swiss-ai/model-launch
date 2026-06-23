"""The keyboard chord for opening a node terminal: press "t", then a node number."""

import asyncio

from swiss_ai_model_launch.cli.display.live import _TERM_PROMPT_ID, _SMLApp, _terminal_targets
from swiss_ai_model_launch.cli.display.state import DisplayState
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealth, ReplicaHealthReport
from swiss_ai_model_launch.launchers.launcher import TerminalCommand


def _report(*hosts: str) -> ReplicaHealthReport:
    replicas = tuple(
        ReplicaHealth(ModelHealth.HEALTHY, "peer", 1, rank, f"10.0.0.{rank}", host) for rank, host in enumerate(hosts)
    )
    return ReplicaHealthReport("m", len(hosts), replicas)


def _wire(state: DisplayState) -> list[TerminalCommand]:
    opened: list[TerminalCommand] = []
    state.open_terminal = lambda job_id, host: TerminalCommand(
        argv=["echo", f"{job_id}:{host}"], display=f"echo {job_id}:{host}"
    )
    return opened


def test_terminal_targets_maps_rank_to_job_and_host() -> None:
    state = DisplayState()
    state.update(job_id=99)
    state.set_replica_report(_report("nid0", "nid1", "nid2"))
    assert _terminal_targets(state) == {0: (99, "nid0"), 1: (99, "nid1"), 2: (99, "nid2")}


def test_terminal_targets_newest_job_wins_in_a_chain() -> None:
    state = DisplayState()
    state.set_replica_reports([(10, _report("old0", "old1")), (11, _report("new0", "new1"))])
    # Same ranks under both jobs; the newer job (11) is the one to attach to.
    assert _terminal_targets(state) == {0: (11, "new0"), 1: (11, "new1")}


async def test_chord_opens_terminal_for_typed_node() -> None:
    state = DisplayState()
    opened = _wire(state)
    state.update(job_id=99)
    state.set_replica_report(_report("nid0", "nid1", "nid2"))

    app = _SMLApp(state, asyncio.sleep(3600))
    app._open_terminal = opened.append  # type: ignore[method-assign]  # capture instead of suspending

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert app.query_one(f"#{_TERM_PROMPT_ID}").display is True  # prompt is shown
        # With <10 nodes a single digit is unambiguous and opens immediately.
        await pilot.press("1")
        await pilot.pause()

    assert len(opened) == 1
    assert opened[0].argv == ["echo", "99:nid1"]
    assert app._term_buffer is None  # prompt closed again


async def test_chord_two_digit_node_needs_enter() -> None:
    state = DisplayState()
    opened = _wire(state)
    state.update(job_id=7)
    state.set_replica_report(_report(*(f"nid{i}" for i in range(12))))  # ranks 0..11

    app = _SMLApp(state, asyncio.sleep(3600))
    app._open_terminal = opened.append  # type: ignore[method-assign]

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        # "1" is ambiguous (1, 10, 11) so nothing opens yet.
        await pilot.press("1")
        await pilot.pause()
        assert not opened
        assert app._term_buffer == "1"
        # "11" is a complete number with no longer extension -> opens at once.
        await pilot.press("1")
        await pilot.pause()

    assert len(opened) == 1
    assert opened[0].argv == ["echo", "7:nid11"]


async def test_chord_enter_commits_ambiguous_prefix() -> None:
    state = DisplayState()
    opened = _wire(state)
    state.update(job_id=7)
    state.set_replica_report(_report(*(f"nid{i}" for i in range(12))))

    app = _SMLApp(state, asyncio.sleep(3600))
    app._open_terminal = opened.append  # type: ignore[method-assign]

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t", "1", "enter")  # "1" alone means node 1, confirmed by Enter
        await pilot.pause()

    assert len(opened) == 1
    assert opened[0].argv == ["echo", "7:nid1"]


async def test_chord_escape_cancels_without_opening() -> None:
    state = DisplayState()
    opened = _wire(state)
    state.update(job_id=99)
    state.set_replica_report(_report("nid0", "nid1"))

    app = _SMLApp(state, asyncio.sleep(3600))
    app._open_terminal = opened.append  # type: ignore[method-assign]

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert app.query_one(f"#{_TERM_PROMPT_ID}").display is True
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one(f"#{_TERM_PROMPT_ID}").display is False
        assert app._term_buffer is None

    assert not opened


async def test_t_is_inert_when_no_targets() -> None:
    # No replica report at all -> the leader key must not open an (empty) prompt.
    state = DisplayState()
    _wire(state)
    app = _SMLApp(state, asyncio.sleep(3600))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()
        assert app.query_one(f"#{_TERM_PROMPT_ID}").display is False
        assert app._term_buffer is None
