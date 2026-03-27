from collections import deque
from collections.abc import Callable

from swiss_ai_model_launch.cli.healthcheck import ModelHealth
from swiss_ai_model_launch.launchers.launcher import JobStatus


class DisplayState:
    def __init__(self) -> None:
        self.cluster: str | None = None
        self.partition: str | None = None
        self.job_id: int | None = None
        self.job_status: JobStatus | None = None
        self.model_health: ModelHealth = ModelHealth.NOT_DEPLOYED
        self.served_model_name: str | None = None
        self.out_logs: deque[str] = deque()
        self.err_logs: deque[str] = deque()
        self._on_change: Callable[[], None] = lambda: None

    def _notify(self) -> None:
        self._on_change()

    def update(
        self,
        cluster: str | None = None,
        partition: str | None = None,
        job_id: int | None = None,
        job_status: JobStatus | None = None,
        model_health: ModelHealth | None = None,
        served_model_name: str | None = None,
    ) -> None:
        if cluster is not None:
            self.cluster = cluster
        if partition is not None:
            self.partition = partition
        if job_id is not None:
            self.job_id = job_id
        if job_status is not None:
            self.job_status = job_status
        if model_health is not None:
            self.model_health = model_health
        if served_model_name is not None:
            self.served_model_name = served_model_name
        self._notify()

    def set_out_log(self, text: str) -> None:
        self.out_logs = deque(text.splitlines())
        self._notify()

    def set_err_log(self, text: str) -> None:
        self.err_logs = deque(text.splitlines())
        self._notify()
