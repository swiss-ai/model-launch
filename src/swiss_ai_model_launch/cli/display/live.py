from collections.abc import Coroutine
from typing import Any

from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, RichLog, TabbedContent, TabPane
from textual.worker import Worker, WorkerState

from swiss_ai_model_launch.cli.display.state import DisplayState
from swiss_ai_model_launch.cli.healthcheck import ModelHealth
from swiss_ai_model_launch.launchers.launcher import JobStatus

_JOB_STATUS_STYLE: dict[JobStatus, str] = {
    JobStatus.PENDING: "[yellow]PENDING[/yellow]",
    JobStatus.RUNNING: "[green]RUNNING[/green]",
    JobStatus.TIMEOUT: "[red]TIMEOUT[/red]",
    JobStatus.UNKNOWN: "[dim]UNKNOWN[/dim]",
}

_MODEL_HEALTH_STYLE: dict[ModelHealth, str] = {
    ModelHealth.WAITING: "[yellow]WAITING[/yellow]",
    ModelHealth.HEALTHY: "[green]HEALTHY[/green]",
    ModelHealth.ERROR: "[orange]ERROR[/orange]",
    ModelHealth.NOT_RESPONDING: "[red]NOT RESPONDING[/red]",
}

_STATUS_LABEL_ID = "status-label"
_OUT_LOG_ID = "out-log"
_ERR_LOG_ID = "err-log"


class _SMLApp(App[bool]):
    TITLE = "SwissAI Model Launch"
    BINDINGS = [
        Binding("ctrl+x", "quit_resume", "Quit and Resume"),
        Binding("ctrl+k", "quit_kill", "Quit and Kill"),
    ]

    CSS = """
    #status-label {
        height: auto;
        width: 1fr;
        padding: 1 2;
        border: solid $primary;
    }
    TabbedContent {
        height: 1fr;
    }
    RichLog {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, state: DisplayState, work: Coroutine[Any, Any, None]) -> None:
        super().__init__()
        self._state = state
        self._work = work
        self._out_written = 0
        self._err_written = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(self._render_status(), id=_STATUS_LABEL_ID, markup=True)
        with TabbedContent("stdout", "stderr"):
            with TabPane("stdout"):
                yield RichLog(id=_OUT_LOG_ID, highlight=False, markup=False)
            with TabPane("stderr"):
                yield RichLog(id=_ERR_LOG_ID, highlight=False, markup=False)
        yield Footer()

    async def on_mount(self) -> None:
        self._state._on_change = self._refresh_all
        self.run_worker(self._work, exclusive=True)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            self.exit(False)

    def action_quit_resume(self) -> None:
        self.exit(False)

    def action_quit_kill(self) -> None:
        self.exit(True)

    def _render_status(self) -> Table:
        s = self._state
        job_status = (
            _JOB_STATUS_STYLE[s.job_status]
            if s.job_status is not None
            else "[dim]—[/dim]"
        )
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(
            f"[bold]Cluster:[/bold] {s.cluster or '[dim]—[/dim]'}",
            f"[bold]Partition:[/bold] {s.partition or '[dim]—[/dim]'}",
        )
        table.add_row(
            f"[bold]Job ID:[/bold] {s.job_id if s.job_id else '[dim]—[/dim]'}",
            f"[bold]Served Model:[/bold] {s.served_model_name or '[dim]—[/dim]'}",
        )
        table.add_row(
            f"[bold]Job Status:[/bold] {job_status}",
            f"[bold]Model Health:[/bold] {_MODEL_HEALTH_STYLE[s.model_health]}",
        )
        return table

    def _refresh_all(self) -> None:
        self.query_one(f"#{_STATUS_LABEL_ID}", Label).update(self._render_status())

        out_lines = list(self._state.out_logs)
        out_log = self.query_one(f"#{_OUT_LOG_ID}", RichLog)
        for line in out_lines[self._out_written :]:
            out_log.write(line)
        self._out_written = len(out_lines)

        err_lines = list(self._state.err_logs)
        err_log = self.query_one(f"#{_ERR_LOG_ID}", RichLog)
        for line in err_lines[self._err_written :]:
            err_log.write(line)
        self._err_written = len(err_lines)


class LiveDisplay:
    def __init__(self, state: DisplayState) -> None:
        self._state = state

    async def run(self, work: Coroutine[Any, Any, None]) -> bool:
        app = _SMLApp(self._state, work)
        return await app.run_async() or False
