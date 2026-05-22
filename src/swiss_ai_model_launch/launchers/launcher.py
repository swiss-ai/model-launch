from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from swiss_ai_model_launch.launchers.framework import OCF_BOOTSTRAP_ADDR
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.utils import create_salt

if TYPE_CHECKING:
    from swiss_ai_model_launch.cli.healthcheck import ReplicaHealthReport

# The replica health-check runs as a short helper job; allow plenty of headroom
# for it to schedule on a busy cluster before giving up.
_REPLICA_JOB_TIME = "00:10:00"
_REPLICA_TERMINAL_STATUSES = {JobStatus.TIMEOUT, JobStatus.UNKNOWN}


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

    @abstractmethod
    async def get_job_status(self, job_id: int) -> JobStatus: ...

    @abstractmethod
    async def get_job_logs(self, job_id: int) -> tuple[str, str]: ...

    @abstractmethod
    async def cancel_job(self, job_id: int) -> None: ...

    def get_log_dir(self, job_id: int) -> str:
        return f"~/.sml/logs/{job_id}"

    @abstractmethod
    def get_log_dir(self, job_id: int) -> str: ...

    @abstractmethod
    async def _submit_helper_script(self, script_body: str, *, job_name: str, time: str) -> int:
        """Submit a single-node bash helper job and return its job ID.

        ``script_body`` is the script without shebang/``#SBATCH`` header; the
        implementation prepends whatever its submission path requires and writes
        stdout to the same ``logs/<job_id>/log.out`` location ``get_job_logs``
        reads from.
        """
        ...

    async def check_replicas_health(
        self,
        served_model_name: str,
        api_key: str,
        *,
        expected_replicas: int = 0,
        bootstrap_addr: str | None = None,
        probe_timeout_seconds: int = 10,
        poll_interval_seconds: int = 15,
        timeout_seconds: float = 1200.0,
    ) -> ReplicaHealthReport:
        """Probe every replica of ``served_model_name`` and report each separately.

        The OpenTela DNT API is only reachable from inside the cluster network, so
        this submits a short helper job that runs the probe on a compute node and
        prints a JSON report, then waits for and parses that report. Works
        uniformly for every launcher because it only uses the abstract job
        primitives.
        """
        # Imported lazily: `cli.healthcheck` pulls in `cli/__init__` -> `main` ->
        # `launchers`, which would be a cycle at module import time.
        from swiss_ai_model_launch.assets.replica_probe import REPORT_BEGIN, REPORT_END
        from swiss_ai_model_launch.cli.healthcheck import (
            ReplicaHealthReport,
            dnt_base_url_from_bootstrap,
            parse_report,
            render_probe_script,
        )

        dnt_base_url = dnt_base_url_from_bootstrap(bootstrap_addr or OCF_BOOTSTRAP_ADDR)
        script_body = render_probe_script(served_model_name, api_key, dnt_base_url, probe_timeout_seconds)
        job_name = f"sml_replica_health_{create_salt(8)}"
        job_id = await self._submit_helper_script(script_body, job_name=job_name, time=_REPLICA_JOB_TIME)

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        seen_active = False
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(poll_interval_seconds)
            status = await self.get_job_status(job_id)
            if status in (JobStatus.PENDING, JobStatus.RUNNING):
                seen_active = True
            stdout, _ = await self.get_job_logs(job_id)
            if REPORT_BEGIN in stdout and REPORT_END in stdout:
                return parse_report(stdout, served_model_name, expected_replicas)
            if seen_active and status in _REPLICA_TERMINAL_STATUSES:
                # Job finished (COMPLETED/FAILED collapse to UNKNOWN) without a
                # report — parse anyway to surface the failure to the caller.
                return parse_report(stdout, served_model_name, expected_replicas)

        with contextlib.suppress(Exception):
            await self.cancel_job(job_id)
        return ReplicaHealthReport(
            served_model_name,
            expected_replicas,
            table_error="replica health-check job did not report within the timeout",
        )

    async def start_replica_probe(
        self,
        served_model_name: str,
        api_key: str,
        *,
        bootstrap_addr: str | None = None,
        probe_timeout_seconds: int = 10,
        refresh_interval_seconds: int = 5,
        time_limit: str = "04:00:00",
    ) -> int:
        """Submit a long-running probe job that re-reports replica health on a loop.

        Returns the helper job ID. The job keeps emitting a fresh report every
        ``refresh_interval_seconds`` until cancelled or its ``time_limit`` expires;
        callers poll ``get_job_logs`` + ``parse_report`` to read the latest one.
        Used for live TUI monitoring (the one-shot ``check_replicas_health`` is for
        integration tests). Remember to ``cancel_job`` it when done.
        """
        from swiss_ai_model_launch.cli.healthcheck import dnt_base_url_from_bootstrap, render_probe_script

        dnt_base_url = dnt_base_url_from_bootstrap(bootstrap_addr or OCF_BOOTSTRAP_ADDR)
        script_body = render_probe_script(
            served_model_name,
            api_key,
            dnt_base_url,
            probe_timeout_seconds,
            refresh_interval_seconds,
        )
        job_name = f"sml_replica_health_{create_salt(8)}"
        return await self._submit_helper_script(script_body, job_name=job_name, time=time_limit)
