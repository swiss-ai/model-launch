import re
import warnings
from typing import Any

from pydantic import BaseModel, Field, model_validator

# The framework's HTTP server port is hardcoded across the system: it's
# auto-injected as ``--port`` into framework_args, used as OCF's
# ``--service.port``, and embedded in the router's worker URLs. Exposing
# it as a knob just creates ways for the three to drift.
FRAMEWORK_PORT = 8080

_PORT_FLAG_RE = re.compile(r"(?:^|\s)--port(?:[\s=])")


class Topology(BaseModel):
    """Hardware layout for a launch.

    A *replica* is one independent inference engine instance — what the
    router load-balances over. Sharding within a replica (TP/PP/DP/EP) is
    configured by the user via ``framework_args``; this layer doesn't
    touch it.
    """

    replicas: int = 1
    nodes_per_replica: int = 1


_LEGACY_TOPOLOGY_KEYS = {
    "workers": "replicas",
    "nodes_per_worker": "nodes_per_replica",
}


class LaunchArgs(BaseModel):
    job_name: str
    served_model_name: str
    account: str
    partition: str

    topology: Topology = Field(default_factory=Topology)

    time: str = "00:05:00"
    reservation: str | None = None
    environment: str

    framework: str
    framework_args: str = ""
    pre_launch_cmds: str = ""
    use_router: bool = False
    router_args: str = ""
    disable_ocf: bool = False
    telemetry_endpoint: str | None = None
    metrics_remote_write_url: str = "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
    metrics_agent_binary: str = "/capstor/store/cscs/swissai/infra01/ocf-share/vmagent"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_topology_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        topo = dict(data.get("topology") or {})
        for legacy, new in _LEGACY_TOPOLOGY_KEYS.items():
            if legacy in data:
                warnings.warn(
                    f"`{legacy}` is deprecated; use `topology.{new}` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                topo.setdefault(new, data.pop(legacy))
        if "worker_port" in data:
            warnings.warn(
                "`worker_port` is no longer configurable; the framework port is hardcoded "
                f"to {FRAMEWORK_PORT}. Drop the argument.",
                DeprecationWarning,
                stacklevel=2,
            )
            data.pop("worker_port")
        if "nodes" in data:
            warnings.warn(
                "`nodes` is no longer configurable; total nodes is derived from "
                "`topology.replicas * topology.nodes_per_replica`. Drop the argument.",
                DeprecationWarning,
                stacklevel=2,
            )
            data.pop("nodes")
        if topo:
            data["topology"] = topo
        return data

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
