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
    metrics_remote_write_url: str = "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    metrics_agent_binary: str = "/capstor/store/cscs/swissai/infra01/ocf-share/vmagent"

    @model_validator(mode="after")
    def set_defaults(self) -> "LaunchArgs":
        if self.nodes is None:
            self.nodes = self.workers * self.nodes_per_worker
        return self

    def to_job_env(self) -> dict[str, str]:
        return {
            "FRAMEWORK": self.framework,
            "SML_ENVIRONMENT": self.environment,
            "FRAMEWORK_ARGS": self.framework_args,
            "PRE_LAUNCH_CMDS": self.pre_launch_cmds,
            "WORKERS": str(self.workers),
            "NODES_PER_WORKER": str(self.nodes_per_worker),
            "WORKER_PORT": str(self.worker_port),
            "USE_ROUTER": "true" if self.use_router else "false",
            "ROUTER_ENVIRONMENT": self.environment,
            "ROUTER_ARGS": self.router_args,
            "USE_OCF": "false" if self.disable_ocf else "true",
            "SERVED_MODEL_NAME": self.served_model_name,
            "METRICS_REMOTE_WRITE_URL": self.metrics_remote_write_url or "",
            "METRICS_AGENT_BIN": self.metrics_agent_binary,
            "TELEMETRY_ENDPOINT": self.telemetry_endpoint or "",
            "SML_TIME": self.time,
        }
