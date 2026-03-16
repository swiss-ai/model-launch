from abc import ABC, abstractmethod
from enum import Enum

from .launch_request import LaunchRequest


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class Launcher(ABC):
    @abstractmethod
    async def get_preconfigured_models(self) -> list[LaunchRequest]: ...

    @abstractmethod
    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]: ...

    @abstractmethod
    async def get_job_status(self, job_id: int) -> JobStatus: ...

    @abstractmethod
    async def get_job_logs(self, job_id: int) -> tuple[str, str]: ...
