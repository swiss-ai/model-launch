import os
import re
import time
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

import rich
from rich import box
from rich.console import Group, RenderableType
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
    JobStatus.COMPLETED: "[blue]COMPLETED[/blue]",
    JobStatus.CANCELLED: "[dim]CANCELLED[/dim]",
    JobStatus.FAILED: "[red]FAILED[/red]",
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
_CHAIN_LABEL_ID = "chain-label"
_REPLICA_LABEL_ID = "replica-label"
_SOURCE_TABS_ID = "source-tabs"


def _slug(source: str) -> str:
    """A widget-id-safe slug for a log source label (unique per distinct label)."""
    return re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")


def _model_health_label(health: ModelHealth, blink_on: bool) -> str:
    """The styled health text, with a green heart beside HEALTHY that blinks.

    The heart is driven by the panel's 0.5s refresh tick (``blink_on`` flips each
    tick). When off, a 2-cell placeholder keeps its slot reserved so the label
    doesn't jitter as the heart toggles.
    """
    label = _MODEL_HEALTH_STYLE[health]
    if health != ModelHealth.HEALTHY:
        return label
    heart = "[green]💚[/green]" if blink_on else "  "
    return f"{heart} {label}"


def _replica_summary(report: ReplicaHealthReport, job_id: int | None = None) -> str:
    expected = report.expected_replicas if report.expected_replicas > 0 else report.found
    healthy = sum(r.health == ModelHealth.HEALTHY for r in report.replicas)
    color = "green" if report.complete else "yellow"
    label = f"Job {job_id}" if job_id is not None else "Replica Health"
    return f"{label} — [{color}]{healthy}/{expected} healthy[/{color}]"


def _format_heartbeat(last_seen: int | None) -> str:
    if last_seen is None:
        return "[dim]—[/dim]"
    when = datetime.fromtimestamp(last_seen).strftime("%H:%M:%S")
    # Relative to the live wall clock (both epochs are UTC) so the age ticks up
    # smoothly on each panel re-render between data refreshes.
    ago = max(0, int(time.time()) - last_seen)
    return f"{when} [dim]({ago}s ago)[/dim]"


def _render_chain_panel(state: DisplayState) -> RenderableType:
    """Table of every job in a consecutive chain: id, begin time, status, focus.

    The ▶ marker is the job currently driving the replica/logs panels (the newest
    live one after a handover). Only meaningful for chains, so callers gate on
    ``len(state.chain) > 1``.
    """
    table = Table(box=box.SIMPLE_HEAD, expand=True, title="Consecutive Job Chain", title_justify="left")
    table.add_column("", width=2)
    table.add_column("Job ID")
    table.add_column("Begins")
    table.add_column("Ends")
    table.add_column("Status", justify="right", width=10)
    for job in state.chain:
        status = _JOB_STATUS_STYLE[job.status] if job.status is not None else "[dim]—[/dim]"
        marker = "[bold green]▶[/bold green]" if job.job_id == state.job_id else " "
        table.add_row(
            marker,
            str(job.job_id),
            job.begin or "[dim]now[/dim]",
            job.end or "[dim]—[/dim]",
            status,
        )
    return table


def _render_one_replica_report(
    report: ReplicaHealthReport, blink_on: bool, job_id: int | None = None
) -> RenderableType:
    label = f"Job {job_id} — " if job_id is not None else ""
    if report.error is not None:
        return f"[bold]{label}Replica Health[/bold]\n[orange]Report unavailable: {report.error}[/orange]"
    if not report.replicas:
        return f"[bold]{label}Replica Health[/bold]\n[red]No replicas reported.[/red]"
    table = Table(box=box.SIMPLE_HEAD, expand=True, title=_replica_summary(report, job_id), title_justify="left")
    table.add_column("Node", justify="right", style="dim", width=4)
    table.add_column("Node IP")
    table.add_column("Health", justify="right", width=16)
    table.add_column("Last Heartbeat", justify="right")
    for replica in report.replicas:
        table.add_row(
            str(replica.node_rank) if replica.node_rank is not None else "[dim]—[/dim]",
            replica.node_ip or "[dim]—[/dim]",
            _model_health_label(replica.health, blink_on),
            _format_heartbeat(replica.last_seen),
        )
    if report.checked_at is not None:
        table.caption = f"checked at {datetime.fromtimestamp(report.checked_at).strftime('%H:%M:%S')}"
        table.caption_justify = "right"
    return table


def _render_replica_panel(state: DisplayState, blink_on: bool = False) -> RenderableType:
    # Chains report per-job so an overlapping handover shows every running job's
    # replicas, stacked oldest→newest. A single launch uses the unlabelled report.
    if state.replica_reports:
        return Group(
            *(_render_one_replica_report(report, blink_on, job_id) for job_id, report in state.replica_reports)
        )
    if state.replica_report is None:
        return "[bold]Replica Health[/bold]\n[dim]Waiting for the job's first health report…[/dim]"
    return _render_one_replica_report(state.replica_report, blink_on)


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
    #chain-label {
        height: auto;
        max-height: 12;
        width: 1fr;
        padding: 0 2;
        border: solid $accent;
        overflow-y: auto;
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
        self._blink_on = False
        self._slug_to_source = {_slug(source): source for source in state.sources}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(self._render_status(), id=_STATUS_LABEL_ID, markup=True)
        # The chain panel starts hidden; _refresh_chain reveals it once a
        # consecutive chain (more than one job) is registered.
        chain_label = Label(_render_chain_panel(self._state), id=_CHAIN_LABEL_ID, markup=True)
        chain_label.display = False
        yield chain_label
        yield Label(_render_replica_panel(self._state, self._blink_on), id=_REPLICA_LABEL_ID, markup=True)
        # Outer tabs: one per log source (Master, Replica 0, …, Router). Inside
        # each, stdout/stderr sub-tabs. Only the active source's logs are fetched.
        with TabbedContent(id=_SOURCE_TABS_ID):
            for source in self._state.sources:
                slug = _slug(source)
                with TabPane(source, id=f"src-{slug}"):
                    with TabbedContent():
                        with TabPane("stdout"):
                            yield TextArea("", id=f"log-{slug}-out", read_only=True)
                        with TabPane("stderr"):
                            yield TextArea("", id=f"log-{slug}-err", read_only=True)
        yield Footer()

    async def on_mount(self) -> None:
        self._state._on_change = self._refresh_all
        # Re-render twice a second so the per-replica heartbeat "ago" counter ticks
        # smoothly and the HEALTHY heart blinks, independent of the slower data refresh.
        self.set_interval(0.5, self._tick)
        self.run_worker(self._work, exclusive=True)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        # Only the outer (source) tabs change which source the monitor fetches;
        # ignore the inner stdout/stderr switches.
        if event.tabbed_content.id != _SOURCE_TABS_ID:
            return
        source = self._slug_to_source.get((event.tabbed_content.active or "").removeprefix("src-"))
        if source is not None:
            self._state.set_active_source(source)
            self._load_source(source)  # show cached content immediately

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
            f"[bold]Model Health:[/bold] {_model_health_label(s.model_health, self._blink_on)}",
        )
        return table

    def _tick(self) -> None:
        # Drive the blinking HEALTHY heart and refresh both panels that show it.
        self._blink_on = not self._blink_on
        self.query_one(f"#{_STATUS_LABEL_ID}", Label).update(self._render_status())
        self._refresh_replicas()

    def _refresh_replicas(self) -> None:
        self.query_one(f"#{_REPLICA_LABEL_ID}", Label).update(_render_replica_panel(self._state, self._blink_on))

    def _refresh_chain(self) -> None:
        label = self.query_one(f"#{_CHAIN_LABEL_ID}", Label)
        show = len(self._state.chain) > 1
        label.display = show
        if show:
            label.update(_render_chain_panel(self._state))

    def _load_source(self, source: str) -> None:
        slug = _slug(source)
        out, err = self._state.source_logs.get(source, ("", ""))
        self._update_log_area(f"log-{slug}-out", out)
        self._update_log_area(f"log-{slug}-err", err)

    def _update_log_area(self, area_id: str, text: str) -> None:
        """Refresh a log pane without the full-reload flicker or scroll yank.

        The monitor re-pushes the whole log every cycle, so we skip the reload
        entirely when the text is unchanged. When it did change, we only jump to
        the tail if the user was already pinned to the bottom (follow mode);
        otherwise we keep their scroll position, since logs only grow at the end.
        """
        area = self.query_one(f"#{area_id}", TextArea)
        if area.text == text:
            return
        was_at_bottom = area.scroll_offset.y >= area.max_scroll_y
        previous_offset = area.scroll_offset
        area.load_text(text)
        if was_at_bottom:
            area.scroll_end(animate=False)
        else:
            area.scroll_to(x=previous_offset.x, y=previous_offset.y, animate=False)

    def _refresh_all(self) -> None:
        self.query_one(f"#{_STATUS_LABEL_ID}", Label).update(self._render_status())
        self._refresh_chain()
        self._refresh_replicas()
        # Only the active source's logs are fetched/updated each cycle.
        self._load_source(self._state.active_source)


class LiveDisplay:
    def __init__(self, state: DisplayState) -> None:
        self._state = state

    async def run(self, work: Coroutine[Any, Any, None]) -> bool:
        app = _SMLApp(self._state, work)
        return await app.run_async() or False
