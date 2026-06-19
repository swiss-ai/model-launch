import contextlib
import os
import re
import subprocess
import time
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

import rich
from rich import box
from rich.console import Group, RenderableType
from rich.segment import Segments
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.traceback import Traceback
from textual.app import App, ComposeResult, SuspendNotSupported
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, TabbedContent, TabPane, TextArea
from textual.worker import Worker, WorkerState

from swiss_ai_model_launch.cli.display.state import DisplayState
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealthReport
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launcher import TerminalCommand

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

# The in-job checker rewrites the report every 30s; if its timestamp is older
# than this, the job (or just the checker) has stopped — so its last "HEALTHY"
# is stale and must not be trusted (e.g. after the job ends or is cancelled).
_REPLICA_REPORT_STALE_AFTER_SECONDS = 90


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


def _compact_time(value: str) -> str:
    """Shorten an ISO timestamp to ``MM-DD HH:MM`` so it fits the chain columns.

    A chain spans at most a couple of days, so the year and seconds are noise and
    just get truncated in the narrow table. Accepts the ISO variants the backends
    emit — with/without a ``T`` separator, a trailing ``Z``, or a timezone offset
    (SLURM uses bare local time; FirecREST may include an offset). Non-ISO values
    (e.g. the dependency hint) are returned unchanged.
    """
    if not isinstance(value, str):
        return value
    raw = value.strip().replace(" ", "T")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).strftime("%m-%d %H:%M")
    except ValueError:
        return value


def _render_chain_panel(state: DisplayState) -> RenderableType:
    """Table of every job in a consecutive chain: id, begin time, status, focus.

    The ▶ marker is the job currently driving the replica/logs panels (the newest
    live one after a handover). Only meaningful for chains, so callers gate on
    ``len(state.chain) > 1``.
    """
    table = Table(box=box.SIMPLE_HEAD, expand=True, title="Consecutive Job Chain", title_justify="left")
    table.add_column("", width=2)
    table.add_column("Job ID", no_wrap=True)
    table.add_column("Begins", no_wrap=True)
    table.add_column("Ends", no_wrap=True)
    table.add_column("Status", justify="right", width=10)
    for job in state.chain:
        status = _JOB_STATUS_STYLE[job.status] if job.status is not None else "[dim]—[/dim]"
        marker = "[bold green]▶[/bold green]" if job.job_id == state.job_id else " "
        # The head job has an absolute begin/end; successors are dependency-scheduled
        # off the previous job's real start, so their wall-clock times are unknown
        # until they run — show the dependency descriptor instead.
        if job.begin:
            begins = _compact_time(job.begin)
        elif job.after:
            begins = f"[dim]{job.after}[/dim]"
        else:
            begins = "[dim]now[/dim]"
        ends = _compact_time(job.end) if job.end else "[dim]—[/dim]"
        table.add_row(
            marker,
            str(job.job_id),
            begins,
            ends,
            status,
        )
    return table


# Node names are simple tokens (e.g. "nid001234", "clariden-ln001"); restricting
# to them keeps the host safe to embed verbatim in the click-action string below.
_NODE_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _terminal_cell(job_id: int | None, node_host: str | None, stale: bool) -> RenderableType:
    """A clickable "open a shell on this node" affordance, or a dim placeholder.

    Offered only when the node's host name and the job id are known and the report
    is fresh — a stale job is gone, so ``srun`` could not attach anyway. The click
    runs the app's ``open_terminal`` action (see ``_SMLApp.action_open_terminal``),
    which suspends the TUI and drops into a shell on the node.
    """
    if job_id is None or not node_host or stale or not _NODE_HOST_RE.match(node_host):
        return Text("—", style="dim", justify="center")
    action = f"app.open_terminal('{job_id}', '{node_host}')"
    style = Style.from_meta({"@click": action}) + Style(color="cyan", underline=True)
    return Text("⏵ open", style=style, justify="center")


def _render_one_replica_report(
    report: ReplicaHealthReport,
    blink_on: bool,
    job_id: int | None = None,
    terminal_job_id: int | None = None,
) -> RenderableType:
    label = f"Job {job_id} — " if job_id is not None else ""
    if report.error is not None:
        return f"[bold]{label}Replica Health[/bold]\n[orange]Report unavailable: {report.error}[/orange]"
    if not report.replicas:
        return f"[bold]{label}Replica Health[/bold]\n[red]No replicas reported.[/red]"
    # Once the report stops being refreshed, its stored health is frozen — a job
    # that ended still reads HEALTHY. Detect that by age and show STALE instead.
    stale = report.checked_at is not None and int(time.time()) - report.checked_at > _REPLICA_REPORT_STALE_AFTER_SECONDS
    name = f"Job {job_id}" if job_id is not None else "Replica Health"
    title = f"{name} — [red]stale (job no longer reporting)[/red]" if stale else _replica_summary(report, job_id)
    # The terminal button targets a specific job id; for a chain that's this
    # report's own job, for a single launch it's the (separately passed) live job.
    tjob = job_id if job_id is not None else terminal_job_id
    table = Table(box=box.SIMPLE_HEAD, expand=True, title=title, title_justify="left")
    table.add_column("Node", justify="right", style="dim", width=4)
    table.add_column("Node Name")
    table.add_column("Node IP")
    table.add_column("Health", justify="right", width=16)
    table.add_column("Last Heartbeat", justify="right")
    table.add_column("Terminal", justify="center", width=8)
    for replica in report.replicas:
        health = "[dim]STALE[/dim]" if stale else _model_health_label(replica.health, blink_on)
        # node_host / node_ip come from the job's health report (untrusted, e.g. a
        # FirecREST download). Render them as literal Text so a crafted value can't
        # inject Rich console markup — notably a `[@click=…]` action span that would
        # otherwise become a clickable shortcut to arbitrary app actions.
        node_name = Text(replica.node_host) if replica.node_host else "[dim]—[/dim]"
        node_ip = Text(replica.node_ip) if replica.node_ip else "[dim]—[/dim]"
        table.add_row(
            str(replica.node_rank) if replica.node_rank is not None else "[dim]—[/dim]",
            node_name,
            node_ip,
            health,
            _format_heartbeat(replica.last_seen),
            _terminal_cell(tjob, replica.node_host, stale),
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
    # Single launch: the report is unlabelled (no "Job N" title), but the terminal
    # button still needs the live job id to target.
    return _render_one_replica_report(state.replica_report, blink_on, terminal_job_id=state.job_id)


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

    def action_open_terminal(self, job_id: str, node_host: str) -> None:
        """Open an interactive shell on a replica's node (the Terminal column click).

        Builds the launcher-specific command, then suspends the TUI and runs it so
        the user lands in a shell on the node; the app resumes when the shell exits.
        If the launcher can't open one (e.g. FirecREST with no SSH host), the
        command is copied to the clipboard with an explanation instead.
        """
        factory = self._state.open_terminal
        if factory is None:
            self.notify("Opening a node terminal isn't available here.", severity="warning")
            return
        try:
            command = factory(int(job_id), node_host)
        except Exception as exc:  # a malformed action must never crash the app
            self.notify(f"Couldn't build the terminal command: {exc}", severity="error")
            return
        if not command.available or not command.argv:
            self._offer_terminal_command(command)
            return
        self._open_terminal(command)

    def _offer_terminal_command(self, command: TerminalCommand) -> None:
        """Fallback when we can't launch directly: copy the command and explain."""
        if command.display:
            # Clipboard is best-effort (e.g. a terminal with no OSC-52 support).
            with contextlib.suppress(Exception):
                self.copy_to_clipboard(command.display)
        reason = command.reason or "Can't open a terminal here."
        suffix = f"\nCommand copied to clipboard:\n{command.display}" if command.display else ""
        self.notify(reason + suffix, severity="warning", timeout=10)

    def _open_terminal(self, command: TerminalCommand) -> None:
        # Deliberately synchronous: the child shell owns the real TTY while it runs,
        # so we must NOT let Textual's render loop keep painting underneath it. A
        # blocking subprocess.run on the message-pump (the documented `with
        # self.suspend(): …` pattern) guarantees that — the monitor worker simply
        # pauses (the display is suspended anyway) and resumes when the shell exits.
        try:
            with self.suspend():
                print(f"\n$ {command.display}\n", flush=True)
                try:
                    subprocess.run(command.argv)  # noqa: S603 - argv built from launcher + validated host
                except FileNotFoundError:
                    print(f"Command not found: {command.argv[0]!r}", flush=True)
                    input("Press Enter to return to SML… ")
        except SuspendNotSupported:
            self._offer_terminal_command(
                TerminalCommand(
                    argv=command.argv,
                    display=command.display,
                    available=False,
                    reason="This terminal can't be suspended to open a shell.",
                )
            )
        except Exception as exc:  # keep the TUI alive if the session blows up
            self.notify(f"Terminal session error: {exc}", severity="error")

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
