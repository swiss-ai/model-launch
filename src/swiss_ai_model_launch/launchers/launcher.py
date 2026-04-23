from abc import ABC, abstractmethod
from enum import Enum

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest, ModelCatalogEntry


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, state: str) -> "JobStatus":
        try:
            return cls(state)
        except ValueError:
            return cls.UNKNOWN


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

    @abstractmethod
    def get_log_dir(self, job_id: int) -> str: ...
