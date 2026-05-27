from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry

if TYPE_CHECKING:
    from swiss_ai_model_launch.cli.healthcheck import ReplicaHealthReport

# The in-job checker writes its report next to the job logs.
REPLICA_HEALTH_FILENAME = "replica_health.json"


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
    async def cancel_job(self, job_id: int) -> None: ...

    def get_log_dir(self, job_id: int) -> str:
        return f"~/.sml/logs/{job_id}"

    @abstractmethod
    def get_log_dir(self, job_id: int) -> str: ...

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
