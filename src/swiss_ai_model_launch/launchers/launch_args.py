import re
import warnings

from pydantic import BaseModel, Field, model_validator

from swiss_ai_model_launch.launchers.topology import Topology

# The framework's HTTP server port is hardcoded across the system: it's
# auto-injected as ``--port`` into framework_args, used as OCF's
# ``--service.port``, and embedded in the router's worker URLs. Exposing
# it as a knob just creates ways for the three to drift.
FRAMEWORK_PORT = 8080

TELEMETRY_ENDPOINT = "https://sml-dev.swissai.svc.cscs.ch/launches"

_PORT_FLAG_RE = re.compile(r"(?:^|\s)--port(?:[\s=])")


def time_str_to_seconds(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


class LaunchArgs(BaseModel):
    job_name: str
    served_model_name: str
    account: str
    partition: str

    topology: Topology = Field(default_factory=Topology)

    time: str = "02:00:00"
    environment: str

    framework: str
    framework_args: str = ""
    pre_launch_cmds: str = ""
    use_router: bool = False
    router_args: str = ""
    disable_ocf: bool = False
    ocf_bootstrap_addr: str | None = None
    dev: bool = False
    telemetry_endpoint: str | None = None
    metrics_remote_write_url: str = "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    metrics_agent_binary: str = "/capstor/store/cscs/swissai/infra01/ocf-share/vmagent"
    dcgm_exporter_binary: str = "/capstor/store/cscs/swissai/infra01/ocf-share/dcgm-exporter"
    disable_dcgm_exporter: bool = False
    disable_metrics: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "LaunchArgs":
        if not self.disable_metrics and not self.metrics_remote_write_url:
            raise ValueError("Metrics require a remote write URL when metrics are enabled.")
        if _PORT_FLAG_RE.search(self.framework_args):
            warnings.warn(
                f"`--port` in framework_args is redundant; the framework port is hardcoded "
                f"to {FRAMEWORK_PORT} and auto-injected. Setting it manually risks desyncing "
                f"the framework, OCF, and the router.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @property
    def total_nodes(self) -> int:
        return self.topology.replicas * self.topology.nodes_per_replica

    def to_sbatch_args(self, *, reservation: str | None = None) -> list[str]:
        args = [
            f"--job-name={self.job_name}",
            f"--account={self.account}",
            f"--time={self.time}",
            "--exclusive",
            f"--nodes={self.total_nodes}",
            f"--partition={self.partition}",
            "--output=logs/%j/log.out",
            "--error=logs/%j/log.err",
        ]
        if reservation:
            args.append(f"--reservation={reservation}")
        return args

    def to_job_env(self) -> dict[str, str]:
        framework_args = f"--port {FRAMEWORK_PORT} {self.framework_args}".strip()
        return {
            "FRAMEWORK": self.framework,
            "SML_ENVIRONMENT": self.environment,
            "FRAMEWORK_ARGS": framework_args,
            "PRE_LAUNCH_CMDS": self.pre_launch_cmds,
            "REPLICAS": str(self.topology.replicas),
            "NODES_PER_REPLICA": str(self.topology.nodes_per_replica),
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
