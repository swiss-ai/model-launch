from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import (
    LaunchArgs,
    plan_consecutive_offsets,
    seconds_to_time_str,
    time_str_to_seconds,
)
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry

if TYPE_CHECKING:
    from swiss_ai_model_launch.cli.healthcheck import ReplicaHealthReport

# The in-job checker writes its report next to the job logs.
REPLICA_HEALTH_FILENAME = "replica_health.json"

# SLURM --begin wants a wall-clock timestamp; ISO 8601 without timezone is
# interpreted in the cluster's local time (matching `date`-style begin specs).
_BEGIN_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _format_duration(seconds: int) -> str:
    """Compact h/m label for a chain interval, e.g. 39600 -> '11h', 5400 -> '1h30m'."""
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours and minutes:
        return f"{hours}h{minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


@dataclass(frozen=True)
class ScheduledJob:
    """A submitted job in a consecutive chain."""

    job_id: int
    served_model_name: str
    # The head job has an absolute begin/end (its anchor). Successors are
    # scheduled by dependency on the previous job's actual start, so their
    # wall-clock times aren't known at submission — begin/end are None and
    # ``after`` carries a human descriptor (e.g. "after 123 (+11h)") instead.
    begin: str | None = None
    end: str | None = None
    after: str | None = None


class Launcher(ABC):
    def __init__(
        self,
        system_name: str,
        username: str,
        account: str,
        partition: str,
        reservation: str | None = None,
        telemetry_endpoint: str | None = None,
    ):
        self.system_name = system_name
        self.username = username
        self.account = account
        self.partition = partition
        self.reservation = reservation
        self.telemetry_endpoint = telemetry_endpoint

    @abstractmethod
    async def get_preconfigured_models(self) -> list[ModelCatalogEntry]: ...

    @abstractmethod
    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]: ...

    @abstractmethod
    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]: ...

    async def _prepare_launch_args(self, launch_args: LaunchArgs) -> LaunchArgs:
        """Resolve/stage everything shared by every job in a chain, exactly once.

        Returns a LaunchArgs whose ``environment`` points at the launcher-ready
        location (an absolute local path for SLURM, an uploaded remote path for
        FirecREST). The returned value is the base each chained job is copied from.
        Overridden by launchers that support ``launch_consecutive_with_args``.
        """
        raise NotImplementedError

    async def _submit_one(self, launch_args: LaunchArgs) -> int:
        """Submit a single, fully-prepared job and return its SLURM job id.

        Overridden by launchers that support ``launch_consecutive_with_args``.
        """
        raise NotImplementedError

    async def launch_consecutive_with_args(
        self,
        launch_args: LaunchArgs,
        *,
        total_time: str,
        handover_time: str,
        now: datetime | None = None,
    ) -> list[ScheduledJob]:
        """Pre-schedule and submit a chain of consecutive jobs serving one model.

        ``launch_args.time`` is the per-job SLURM cap; ``total_time`` is the total
        uptime requested. The chain is submitted in order: the **head** job is
        anchored with an absolute ``--begin`` (≈ now), and every **successor** is
        scheduled with a SLURM ``--dependency=after:<prev>+<interval-in-minutes>``
        — so it starts a fixed interval after its predecessor *actually begins
        running*, not after a wall-clock time guessed at submission. This keeps
        the handover overlap correct even if the head (or any job) is delayed in
        the queue. Each job also carries its predecessor's id and cancels it from
        inside once healthy (see the in-job replica health checker).

        Returns a ``ScheduledJob`` for every job, in chain order; the served model
        name is shared so the endpoint stays continuous across the handover.
        """
        total_seconds = time_str_to_seconds(total_time)
        job_seconds = time_str_to_seconds(launch_args.time)
        offsets = plan_consecutive_offsets(total_seconds, job_seconds, time_str_to_seconds(handover_time))
        interval_seconds = job_seconds - time_str_to_seconds(handover_time)
        # SLURM dependency delays are expressed in whole minutes; floor keeps the
        # overlap at least the requested handover (a slightly earlier successor).
        delay_minutes = max(1, interval_seconds // 60)
        base = now or datetime.now()
        prepared = await self._prepare_launch_args(launch_args)

        results: list[ScheduledJob] = []
        previous_job_id: int | None = None
        last_index = len(offsets) - 1
        for index, offset in enumerate(offsets):
            # Earlier jobs run the full cap (they hand over to the next). The last
            # job has no successor, so trim it to land exactly on the requested
            # total instead of overshooting by up to a whole job. By construction
            # (see plan_consecutive_offsets) this remainder is always ≤ the cap.
            this_job_seconds = total_seconds - offset if index == last_index else job_seconds
            job_time = seconds_to_time_str(this_job_seconds)

            begin: str | None
            end: str | None
            after: str | None
            if previous_job_id is None:
                # Head: a single absolute anchor (≈ now). SLURM starts a begin time
                # in the recent past immediately, so it still launches right away.
                begin = (base + timedelta(seconds=offset)).strftime(_BEGIN_TIME_FORMAT)
                end = (base + timedelta(seconds=offset + this_job_seconds)).strftime(_BEGIN_TIME_FORMAT)
                job_args = prepared.model_copy(update={"time": job_time, "begin": begin, "previous_job_id": None})
                after = None
            else:
                # Successor: anchored to the predecessor's real start via dependency.
                begin = end = None
                after = f"{_format_duration(interval_seconds)} after {previous_job_id} starts"
                dependency = f"after:{previous_job_id}+{delay_minutes}"
                job_args = prepared.model_copy(
                    update={"time": job_time, "dependency": dependency, "previous_job_id": previous_job_id}
                )
            job_id = await self._submit_one(job_args)
            results.append(
                ScheduledJob(
                    job_id=job_id,
                    served_model_name=prepared.served_model_name,
                    begin=begin,
                    end=end,
                    after=after,
                )
            )
            previous_job_id = job_id
        return results

    @abstractmethod
    async def get_job_status(self, job_id: int) -> JobStatus: ...

    async def get_job_times(self, job_id: int) -> tuple[str | None, str | None]:
        """The job's ``(start, end)`` wall-clock times as display strings.

        ``start`` is the actual start once running, or the scheduler's estimate
        while pending; ``end`` is when it will (or did) hit its time limit. Either
        is ``None`` when the backend can't report it yet. Used to show real times
        in the consecutive-chain panel instead of submission-time guesses;
        launchers without time reporting can leave the default.
        """
        return None, None

    @abstractmethod
    async def cancel_job(self, job_id: int) -> None: ...

    def get_log_dir(self, job_id: int) -> str:
        return f"~/.sml/logs/{job_id}"

    @abstractmethod
    def get_tail_hint(self, job_id: int) -> str: ...

    @abstractmethod
    async def read_job_file(self, job_id: int, filename: str) -> str | None:
        """Return the decoded contents of ``logs/<job_id>/<filename>``.

        Reads from the job's log directory (local file for SLURM, FirecREST
        download otherwise). Returns ``None`` if the file doesn't exist yet — the
        per-replica logs and the checker's ``replica_health.json`` only appear
        once those processes start.
        """
        ...

    async def get_job_logs(self, job_id: int) -> tuple[str, str]:
        """The master's own ``log.out`` / ``log.err`` (orchestration output).

        Per-replica framework output goes to ``replica_<r>.out`` / ``.err`` — read
        those via ``read_job_file``.
        """
        out = await self.read_job_file(job_id, "log.out")
        err = await self.read_job_file(job_id, "log.err")
        return out or "", err or ""

    async def get_replica_health(
        self,
        job_id: int,
        served_model_name: str,
        expected_replicas: int,
    ) -> ReplicaHealthReport | None:
        """Read the in-job checker's latest report, or ``None`` if not yet written.

        The model's own job runs the checker (see ``framework._render_health_checker``)
        and writes an atomically-replaced JSON report, so this only ever reads a
        complete file. No helper job, no bootstrap, no API key.
        """
        from swiss_ai_model_launch.cli.healthcheck import parse_health_report

        report_json = await self.read_job_file(job_id, REPLICA_HEALTH_FILENAME)
        if report_json is None:
            return None
        return parse_health_report(report_json, served_model_name, expected_replicas)
