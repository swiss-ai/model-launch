from pydantic import BaseModel, model_validator


class LaunchArgs(BaseModel):
    job_name: str
    served_model_name: str
    account: str
    partition: str

    workers: int = 1
    nodes_per_worker: int = 1
    nodes: int | None = None

    time: str = "00:05:00"
    reservation: str | None = None
    environment: str

    framework: str
    framework_args: str = ""
    pre_launch_cmds: str = ""
    worker_port: int = 5000
    use_router: bool = False
    router_args: str = ""
    disable_ocf: bool = False
    telemetry_endpoint: str | None = None
    metrics_remote_write_url: str = (
        "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    )
    metrics_agent_binary: str = "/capstor/store/cscs/swissai/infra01/ocf-share/vmagent"
    dcgm_exporter_binary: str = (
        "/capstor/store/cscs/swissai/infra01/ocf-share/dcgm-exporter"
    )
    disable_dcgm_exporter: bool = False
    disable_metrics: bool = False

    @model_validator(mode="after")
    def set_defaults(self) -> "LaunchArgs":
        if self.nodes is None:
            self.nodes = self.workers * self.nodes_per_worker
        return self
