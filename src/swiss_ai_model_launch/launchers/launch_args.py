import re
import warnings

from pydantic import BaseModel, Field, model_validator

# The framework's HTTP server port is hardcoded across the system: it's
# auto-injected as ``--port`` into framework_args, used as OCF's
# ``--service.port``, and embedded in the router's worker URLs. Exposing
# it as a knob just creates ways for the three to drift.
FRAMEWORK_PORT = 8080

_PORT_FLAG_RE = re.compile(r"(?:^|\s)--port(?:[\s=])")


def time_str_to_seconds(t: str) -> int:
    """Convert a SLURM-style HH:MM:SS duration to total seconds."""
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


class Topology(BaseModel):
    """Hardware layout for a launch.

    A *replica* is one independent inference engine instance — what the
    router load-balances over. Sharding within a replica (TP/PP/DP/EP) is
    configured by the user via ``framework_args``; this layer doesn't
    touch it.
    """

    replicas: int = 1
    nodes_per_replica: int = 1


class LaunchArgs(BaseModel):
    job_name: str
    served_model_name: str
    account: str
    partition: str

    topology: Topology = Field(default_factory=Topology)

    time: str = "02:00:00"
    reservation: str | None = None
    environment: str

    framework: str
    framework_args: str = ""
    pre_launch_cmds: str = ""
    use_router: bool = False
    router_args: str = ""
    disable_ocf: bool = False
    # OCF bootstrap multiaddr. None means "use the prod default baked into
    # framework.py"; CLI plumbing (`--dev`, `--otela-bootstrap-addr`) sets
    # this to override.
    ocf_bootstrap_addr: str | None = None
    # When true, OCF_BIN is resolved against /ocfbin/dev/otela-<arch> (the
    # rolling main-branch build symlink) instead of /ocfbin/prod/otela-<arch>
    # (the latest tagged release symlink). Set by the CLI's `--dev` flag.
    dev: bool = False
    telemetry_endpoint: str | None = None
    metrics_remote_write_url: str = "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    metrics_agent_binary: str = "/capstor/store/cscs/swissai/infra01/ocf-share/vmagent"
    dcgm_exporter_binary: str = "/capstor/store/cscs/swissai/infra01/ocf-share/dcgm-exporter"
    disable_dcgm_exporter: bool = False
    disable_metrics: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "LaunchArgs":
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

    def to_sbatch_args(self) -> list[str]:
        args = [
            f"--job-name={self.job_name}",
            f"--account={self.account}",
            f"--time={self.time}",
            "--exclusive",
            f"--nodes={self.total_nodes}",
            f"--partition={self.partition}",
            "--output=logs/%j/log.out",
            "--error=logs/%j/log.out",
        ]
        if self.reservation:
            args.append(f"--reservation={self.reservation}")
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
