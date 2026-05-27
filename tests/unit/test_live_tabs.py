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
