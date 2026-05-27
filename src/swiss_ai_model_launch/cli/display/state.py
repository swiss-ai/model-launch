from collections.abc import Callable

from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealthReport
from swiss_ai_model_launch.launchers.job_status import JobStatus


class DisplayState:
    def __init__(self, sources: list[str] | None = None) -> None:
        self.cluster: str | None = None
        self.partition: str | None = None
        self.job_id: int | None = None
        self.job_status: JobStatus | None = None
        self.model_health: ModelHealth = ModelHealth.NOT_DEPLOYED
        self.served_model_name: str | None = None
        self.replica_report: ReplicaHealthReport | None = None
        # Log sources shown as tabs (e.g. "Master", "Replica 0", …, "Router").
        # Each maps to its (stdout, stderr); `active_source` is the tab currently
        # in view, so the monitor only fetches that source's logs.
        self.sources: list[str] = sources or ["Master"]
        self.active_source: str = self.sources[0]
        self.source_logs: dict[str, tuple[str, str]] = {source: ("", "") for source in self.sources}
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

    def set_replica_report(self, report: ReplicaHealthReport) -> None:
        self.replica_report = report
        self._notify()

    def set_active_source(self, source: str) -> None:
        # No notify: the tab switch is already rendered; this only tells the
        # monitor which source's logs to fetch next.
        self.active_source = source

    def set_source_log(self, source: str, out: str, err: str) -> None:
        self.source_logs[source] = (out, err)
        self._notify()
