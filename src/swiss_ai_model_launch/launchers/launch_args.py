import math
import re
import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from swiss_ai_model_launch.launchers.topology import Topology

# Routing strategy across replicas. OCF (default): OpenTela load-balances across
# the replica peers on the mesh. SGL: an in-job SGLang router fronts the replicas
# and becomes the served endpoint.
RouterMode = Literal["OCF", "SGL"]
ROUTER_OCF: RouterMode = "OCF"
ROUTER_SGL: RouterMode = "SGL"

# The framework's HTTP server port is hardcoded across the system: it's
# auto-injected as ``--port`` into framework_args, used as OCF's
# ``--service.port``, and embedded in the router's worker URLs. Exposing
# it as a knob just creates ways for the three to drift.
FRAMEWORK_PORT = 8080

TELEMETRY_ENDPOINT = "https://sml-dev.swissai.svc.cscs.ch/launches"

# SLURM caps a single job at 12h on the target clusters. A model that needs to
# stay up longer is served by a chain of consecutive jobs (see --consecutive),
# each running for at most this cap.
DEFAULT_MAX_JOB_TIME = "12:00:00"

_PORT_FLAG_RE = re.compile(r"(?:^|\s)--port(?:[\s=])")


def time_str_to_seconds(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_time_str(seconds: int) -> str:
    # SLURM's finest time-limit granularity is a minute, so never emit 00:00:00.
    seconds = max(seconds, 60)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def plan_consecutive_offsets(total_seconds: int, job_seconds: int, handover_seconds: int) -> list[int]:
    """Start offsets (seconds from the chain's base time) for a consecutive chain.

    Each job runs for ``job_seconds`` (the SLURM per-job cap). A successor starts
    ``handover_seconds`` before its predecessor's time limit, so the spacing
    between starts is ``job_seconds - handover_seconds`` and the overlap gives the
    fresh job time to become healthy before the old one expires. The number of
    jobs is the minimum whose continuous coverage —
    ``(n - 1) * interval + job_seconds`` — reaches ``total_seconds``.

    Returns ``[0]`` (a single job) when the requested total fits inside one job.
    """
    if job_seconds <= 0:
        raise ValueError("job time must be positive")
    if handover_seconds < 0 or handover_seconds >= job_seconds:
        raise ValueError("handover time must be in the range [0, job time)")
    if total_seconds <= job_seconds:
        return [0]
    interval = job_seconds - handover_seconds
    n = math.ceil((total_seconds - job_seconds) / interval) + 1
    return [i * interval for i in range(n)]


class LaunchArgs(BaseModel):
    job_name: str
    served_model_name: str
    account: str
    partition: str

    topology: Topology = Field(default_factory=Topology)

    time: str = "02:00:00"
    # Consecutive-chain scheduling. The head job carries an absolute SLURM
    # --begin (its anchor); every successor instead carries a SLURM --dependency
    # of the form "after:<prev>+<minutes>" so it starts a fixed delay after its
    # predecessor *actually* begins — making the chain robust to queue delay
    # rather than pinned to wall-clock times guessed at submission. Both are None
    # for an ordinary single launch. previous_job_id is the predecessor this job
    # cancels from inside once all its replicas are healthy (see the in-job
    # replica health checker).
    begin: str | None = None
    dependency: str | None = None
    previous_job_id: int | None = None
    environment: str

    framework: str
    framework_args: str = ""
    pre_launch_cmds: str = ""
    router: RouterMode = ROUTER_OCF
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
        if self.begin:
            args.append(f"--begin={self.begin}")
        if self.dependency:
            args.append(f"--dependency={self.dependency}")
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
            "ROUTER": self.router,
            "ROUTER_ENVIRONMENT": self.environment,
            "ROUTER_ARGS": self.router_args,
            "USE_OCF": "false" if self.disable_ocf else "true",
            "SERVED_MODEL_NAME": self.served_model_name,
            "METRICS_REMOTE_WRITE_URL": self.metrics_remote_write_url or "",
            "METRICS_AGENT_BIN": self.metrics_agent_binary,
            "TELEMETRY_ENDPOINT": self.telemetry_endpoint or "",
            "SML_TIME": self.time,
            "SML_PREVIOUS_JOB_ID": str(self.previous_job_id) if self.previous_job_id is not None else "",
        }
