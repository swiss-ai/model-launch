import asyncio

from textual.widgets import TabbedContent, TextArea

from swiss_ai_model_launch.cli.display.live import _slug, _SMLApp
from swiss_ai_model_launch.cli.display.state import DisplayState


async def test_tabs_per_source_and_active_switch() -> None:
    state = DisplayState(["Master", "Replica 0", "Replica 1", "Router"])
    app = _SMLApp(state, asyncio.sleep(3600))
    async with app.run_test() as pilot:
        await pilot.pause()

        # Every source has its own stdout/stderr text areas.
        for source in state.sources:
            slug = _slug(source)
            app.query_one(f"#log-{slug}-out", TextArea)
            app.query_one(f"#log-{slug}-err", TextArea)

        # Master is active by default; its logs render into its panes.
        state.set_source_log("Master", "master stdout", "master stderr")
        await pilot.pause()
        assert app.query_one("#log-master-out", TextArea).text == "master stdout"
        assert app.query_one("#log-master-err", TextArea).text == "master stderr"

        # Switching the outer (source) tab updates which source the monitor fetches.
        app.query_one("#source-tabs", TabbedContent).active = "src-replica-1"
        await pilot.pause()
        assert state.active_source == "Replica 1"

        state.set_source_log("Replica 1", "replica1 stdout", "replica1 stderr")
        await pilot.pause()
        assert app.query_one("#log-replica-1-out", TextArea).text == "replica1 stdout"
        assert app.query_one("#log-replica-1-err", TextArea).text == "replica1 stderr"


def test_compact_time_shortens_iso_and_passes_through_other() -> None:
    from swiss_ai_model_launch.cli.display.live import _compact_time

    # All the ISO shapes the SLURM/FirecREST backends might emit.
    assert _compact_time("2026-06-20T06:04:00") == "06-20 06:04"
    assert _compact_time("2026-06-20T06:04:00+00:00") == "06-20 06:04"
    assert _compact_time("2026-06-20T06:04:00Z") == "06-20 06:04"
    assert _compact_time("2026-06-20 06:04:00") == "06-20 06:04"
    # Non-timestamp values (dependency hint, placeholders) pass through.
    assert _compact_time("5m after 2569108 starts") == "5m after 2569108 starts"
    assert _compact_time("N/A") == "N/A"


def test_chain_panel_shows_compact_times_and_full_job_id() -> None:
    from rich.console import Console

    from swiss_ai_model_launch.cli.display.live import _render_chain_panel
    from swiss_ai_model_launch.cli.display.state import ChainJobView
    from swiss_ai_model_launch.launchers.job_status import JobStatus

    state = DisplayState()
    state.chain = [
        ChainJobView(2569107, begin="2026-06-19T18:04:00", end="2026-06-20T06:04:00", status=JobStatus.RUNNING),
        ChainJobView(2569108, after="11h45m after 2569107 starts", status=JobStatus.PENDING),
    ]
    state.job_id = 2569107
    console = Console(width=100, record=True)
    console.print(_render_chain_panel(state))
    out = console.export_text()
    assert "06-20 06:04" in out  # compact end time, not truncated
    assert "2026-06-20" not in out  # full ISO is gone
    assert "2569107" in out and "2569108" in out  # job ids shown in full


async def test_log_pane_preserves_scroll_when_not_at_bottom() -> None:
    state = DisplayState(["Master"])
    app = _SMLApp(state, asyncio.sleep(3600))
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#log-master-out", TextArea)

        # A long log so the pane is scrollable.
        state.set_source_log("Master", "\n".join(f"line {i}" for i in range(200)), "")
        await pilot.pause()
        assert out.max_scroll_y > 0  # there is something to scroll

        # User scrolls up, away from the tail.
        out.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert out.scroll_offset.y == 0

        # New log lines arrive; the view must NOT jump back to the bottom.
        state.set_source_log("Master", "\n".join(f"line {i}" for i in range(220)), "")
        await pilot.pause()
        assert out.scroll_offset.y == 0
        assert "line 219" in out.text


async def test_log_pane_unchanged_text_is_not_reloaded() -> None:
    state = DisplayState(["Master"])
    app = _SMLApp(state, asyncio.sleep(3600))
    async with app.run_test() as pilot:
        await pilot.pause()
        out = app.query_one("#log-master-out", TextArea)

        state.set_source_log("Master", "\n".join(f"line {i}" for i in range(200)), "")
        await pilot.pause()
        out.scroll_to(y=0, animate=False)
        await pilot.pause()

        # Re-pushing identical content is a no-op: scroll position is left alone.
        state.set_source_log("Master", "\n".join(f"line {i}" for i in range(200)), "")
        await pilot.pause()
        assert out.scroll_offset.y == 0
