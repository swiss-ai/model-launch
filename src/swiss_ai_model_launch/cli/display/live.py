import os
import time
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

import rich
from rich import box
from rich.console import RenderableType
from rich.segment import Segments
from rich.table import Table
from rich.traceback import Traceback
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, TabbedContent, TabPane, TextArea
from textual.worker import Worker, WorkerState

from swiss_ai_model_launch.cli.display.state import DisplayState
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealthReport
from swiss_ai_model_launch.launchers.job_status import JobStatus

_JOB_STATUS_STYLE: dict[JobStatus, str] = {
    JobStatus.PENDING: "[yellow]PENDING[/yellow]",
    JobStatus.RUNNING: "[green]RUNNING[/green]",
    JobStatus.TIMEOUT: "[red]TIMEOUT[/red]",
    JobStatus.UNKNOWN: "[dim]UNKNOWN[/dim]",
}

_MODEL_HEALTH_STYLE: dict[ModelHealth, str] = {
    ModelHealth.HEALTHY: "[green]HEALTHY[/green]",
    ModelHealth.ERROR: "[orange]ERROR[/orange]",
    ModelHealth.NOT_DEPLOYED: "[dim]NOT DEPLOYED[/dim]",
    ModelHealth.NOT_RESPONDING: "[red]NOT RESPONDING[/red]",
}

_STATUS_LABEL_ID = "status-label"
_REPLICA_LABEL_ID = "replica-label"
_OUT_LOG_ID = "out-log"
_ERR_LOG_ID = "err-log"


def _replica_summary(report: ReplicaHealthReport) -> str:
    expected = report.expected_replicas if report.expected_replicas > 0 else report.found
    healthy = sum(r.health == ModelHealth.HEALTHY for r in report.replicas)
    color = "green" if report.complete else "yellow"
    return f"Replica Health — [{color}]{healthy}/{expected} healthy[/{color}]"


def _format_heartbeat(last_seen: int | None) -> str:
    if last_seen is None:
        return "[dim]—[/dim]"
    when = datetime.fromtimestamp(last_seen).strftime("%H:%M:%S")
    # Relative to the live wall clock (both epochs are UTC) so the age ticks up
    # smoothly on each panel re-render between data refreshes.
    ago = max(0, int(time.time()) - last_seen)
    return f"{when} [dim]({ago}s ago)[/dim]"


def _render_replica_panel(state: DisplayState) -> RenderableType:
    report: ReplicaHealthReport | None = state.replica_report
    if report is None:
        return "[bold]Replica Health[/bold]\n[dim]Waiting for the job's first health report…[/dim]"
    if report.error is not None:
        return f"[bold]Replica Health[/bold]\n[orange]Report unavailable: {report.error}[/orange]"
    if not report.replicas:
        return "[bold]Replica Health[/bold]\n[red]No replicas reported.[/red]"
    table = Table(box=box.SIMPLE_HEAD, expand=True, title=_replica_summary(report), title_justify="left")
    table.add_column("Node", justify="right", style="dim", width=4)
    table.add_column("Node IP")
    table.add_column("Peer ID", overflow="fold")
    table.add_column("Health", justify="right", width=16)
    table.add_column("Last Heartbeat", justify="right")
    for replica in report.replicas:
        table.add_row(
            str(replica.node_rank) if replica.node_rank is not None else "[dim]—[/dim]",
            replica.node_ip or "[dim]—[/dim]",
            replica.peer_id or "[dim]—[/dim]",
            _MODEL_HEALTH_STYLE[replica.health],
            _format_heartbeat(replica.last_seen),
        )
    if report.checked_at is not None:
        table.caption = f"checked at {datetime.fromtimestamp(report.checked_at).strftime('%H:%M:%S')}"
        table.caption_justify = "right"
    return table


class _SMLApp(App[bool]):
    TITLE = "SwissAI Model Launch"
    ALLOW_SELECT = True
    BINDINGS = [
        Binding("ctrl+x", "quit_resume", "Quit and Resume", priority=True),
        Binding("ctrl+k", "quit_kill", "Quit and Kill", priority=True),
    ]

    CSS = """
    #status-label {
        height: auto;
        width: 1fr;
        padding: 1 2;
        border: solid $primary;
    }
    #replica-label {
        height: auto;
        max-height: 14;
        width: 1fr;
        padding: 0 2;
        border: solid $secondary;
        overflow-y: auto;
    }
    TabbedContent {
        height: 1fr;
    }
    TextArea {
        height: 1fr;
    }
    """

    def __init__(self, state: DisplayState, work: Coroutine[Any, Any, None]) -> None:
        super().__init__()
        self._state = state
        self._work = work

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(self._render_status(), id=_STATUS_LABEL_ID, markup=True)
        yield Label(_render_replica_panel(self._state), id=_REPLICA_LABEL_ID, markup=True)
        with TabbedContent("stdout", "stderr"):
            with TabPane("stdout"):
                yield TextArea("", id=_OUT_LOG_ID, read_only=True)
            with TabPane("stderr"):
                yield TextArea("", id=_ERR_LOG_ID, read_only=True)
        yield Footer()

    async def on_mount(self) -> None:
        self._state._on_change = self._refresh_all
        # Re-render the replica panel twice a second so the per-replica heartbeat
        # "ago" counter ticks smoothly, independent of the slower data refresh.
        self.set_interval(0.5, self._refresh_replicas)
        self.run_worker(self._work, exclusive=True)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            self.exit(False)

    def action_quit_resume(self) -> None:
        self.exit(False)

    def action_quit_kill(self) -> None:
        self.exit(True)

    def _fatal_error(self) -> None:
        show_locals = os.environ.get("SML_DEBUG", "").lower() in ("1", "true", "yes")
        self.bell()
        self._exit_renderables.append(
            Segments(
                self.console.render(
                    Traceback(show_locals=show_locals, width=None, suppress=[rich]),
                    self.console.options,
                )
            )
        )
        self._close_messages_no_wait()

    def _render_status(self) -> Table:
        s = self._state
        job_status = _JOB_STATUS_STYLE[s.job_status] if s.job_status is not None else "[dim]—[/dim]"
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

    def _refresh_replicas(self) -> None:
        self.query_one(f"#{_REPLICA_LABEL_ID}", Label).update(_render_replica_panel(self._state))

    def _refresh_all(self) -> None:
        self.query_one(f"#{_STATUS_LABEL_ID}", Label).update(self._render_status())
        self._refresh_replicas()

        out_lines = list(self._state.out_logs)
        out_log = self.query_one(f"#{_OUT_LOG_ID}", TextArea)
        out_log.load_text("\n".join(out_lines))
        out_log.scroll_end(animate=False)

        err_lines = list(self._state.err_logs)
        err_log = self.query_one(f"#{_ERR_LOG_ID}", TextArea)
        err_log.load_text("\n".join(err_lines))
        err_log.scroll_end(animate=False)


class LiveDisplay:
    def __init__(self, state: DisplayState) -> None:
        self._state = state

    async def run(self, work: Coroutine[Any, Any, None]) -> bool:
        app = _SMLApp(self._state, work)
        return await app.run_async() or False
