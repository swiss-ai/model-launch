"""Replica-specific health checking via the OpenTela DNT mesh.

The end-to-end check in ``checker.py`` only proves that *one* replica answers
through the public gateway (OpenTela load-balances randomly across peers). To
confirm that *every* replica of a multi-replica launch is alive we query the DNT
table directly, find the peers registered under the served model name, and probe
each one through its dedicated ``/v1/p2p/<peer-id>/...`` route.

The DNT HTTP API lives on the bootstrap node's internal IP (port 8092) and is
**not reachable from outside the cluster network** — so the actual probing runs
on a compute node via a helper SLURM job (see ``Launcher.check_replicas_health``)
that executes the standalone ``assets/replica_probe.py`` and prints a JSON
report. This module renders that helper script and parses the report it emits.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from importlib.resources import files

from swiss_ai_model_launch.assets.replica_probe import REPORT_BEGIN, REPORT_END
from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth

# The DNT HTTP API is served on this port on the OpenTela bootstrap host.
DNT_HTTP_PORT = 8092

_IP4_RE = re.compile(r"^/ip4/([^/]+)/")
_PROBE_TEXT = files("swiss_ai_model_launch.assets").joinpath("replica_probe.py").read_text()
_PROBE_HEREDOC = "__SML_REPLICA_PROBE_EOF__"


@dataclass(frozen=True)
class ReplicaHealth:
    """Health of a single replica, identified by its OpenTela peer ID.

    ``last_seen`` is the epoch-seconds of the mesh's most recent heartbeat from
    the peer (``None`` if the DNT table didn't carry one).
    """

    peer_id: str
    health: ModelHealth
    last_seen: int | None = None


@dataclass(frozen=True)
class ReplicaHealthReport:
    """Per-replica health for one served model.

    Replicas are reported individually (``replicas``); ``all_healthy`` and
    ``complete`` are convenience aggregates for callers that want a single
    verdict. ``table_error`` is set when the probe could not reach the DNT table
    at all (distinct from finding zero replicas).
    """

    served_model_name: str
    expected_replicas: int
    replicas: tuple[ReplicaHealth, ...] = ()
    table_error: str | None = None
    checked_at: int | None = None

    @property
    def found(self) -> int:
        return len(self.replicas)

    @property
    def all_healthy(self) -> bool:
        """True when at least one replica was found and every one is HEALTHY."""
        return (
            self.table_error is None
            and bool(self.replicas)
            and all(r.health == ModelHealth.HEALTHY for r in self.replicas)
        )

    @property
    def complete(self) -> bool:
        """True when all expected replicas are present and HEALTHY."""
        return self.all_healthy and (self.expected_replicas <= 0 or self.found >= self.expected_replicas)


def dnt_base_url_from_bootstrap(bootstrap_addr: str) -> str:
    """Derive the DNT HTTP base URL from an OCF bootstrap multiaddr.

    The DNT API shares the bootstrap node's IP, on ``DNT_HTTP_PORT``. E.g.
    ``/ip4/148.187.108.178/tcp/43905/p2p/Qm...`` -> ``http://148.187.108.178:8092``.
    """
    match = _IP4_RE.match(bootstrap_addr)
    if not match:
        raise ValueError(f"Cannot derive DNT host from bootstrap address: {bootstrap_addr!r}")
    return f"http://{match.group(1)}:{DNT_HTTP_PORT}"


def render_probe_script(
    served_model_name: str,
    api_key: str,
    dnt_base_url: str,
    timeout_seconds: int = 10,
    refresh_interval_seconds: int = 0,
) -> str:
    """Render the bash body of the helper job that runs the replica probe.

    Returns the script body only (no shebang / ``#SBATCH`` header — each launcher
    prepends its own). The probe's config is passed via the environment and the
    probe source is materialised on the node from an embedded heredoc, so the SML
    package need not be installed there.

    ``refresh_interval_seconds`` > 0 makes the probe loop, re-emitting a fresh
    report every interval until the job is cancelled (used for live TUI updates);
    0 probes once and exits (used for the one-shot integration-test check).
    """
    env_lines = "\n".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in (
            ("SML_SERVED_MODEL_NAME", served_model_name),
            ("SML_CSCS_API_KEY", api_key),
            ("SML_DNT_BASE_URL", dnt_base_url),
            ("SML_REPLICA_TIMEOUT", str(timeout_seconds)),
            ("SML_REPLICA_INTERVAL", str(refresh_interval_seconds)),
        )
    )
    return (
        "set -euo pipefail\n\n"
        f"{env_lines}\n\n"
        'probe="$(mktemp /tmp/sml_replica_probe.XXXXXX.py)"\n'
        "trap 'rm -f \"$probe\"' EXIT\n"
        f"cat > \"$probe\" <<'{_PROBE_HEREDOC}'\n"
        f"{_PROBE_TEXT.rstrip()}\n"
        f"{_PROBE_HEREDOC}\n"
        'python3 "$probe"\n'
    )


def _to_health(value: object) -> ModelHealth:
    try:
        return ModelHealth(value)
    except ValueError:
        return ModelHealth.ERROR


def _to_epoch(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _extract_report(stdout: str) -> str | None:
    begin = stdout.rfind(REPORT_BEGIN)
    if begin == -1:
        return None
    end = stdout.find(REPORT_END, begin)
    if end == -1:
        return None
    return stdout[begin + len(REPORT_BEGIN) : end].strip()


def parse_report(stdout: str, served_model_name: str, expected_replicas: int) -> ReplicaHealthReport:
    """Parse the JSON report the probe printed into the helper job's stdout."""
    payload = _extract_report(stdout)
    if payload is None:
        return ReplicaHealthReport(
            served_model_name, expected_replicas, table_error="no replica report found in job output"
        )
    try:
        data = json.loads(payload)
    except ValueError as exc:
        return ReplicaHealthReport(served_model_name, expected_replicas, table_error=f"malformed replica report: {exc}")
    raw_replicas = data.get("replicas", []) if isinstance(data, dict) else []
    replicas = tuple(
        ReplicaHealth(
            peer_id=str(r.get("peer_id", "")),
            health=_to_health(r.get("health")),
            last_seen=_to_epoch(r.get("last_seen")),
        )
        for r in raw_replicas
        if isinstance(r, dict)
    )
    table_error = data.get("table_error") if isinstance(data, dict) else None
    checked_at = _to_epoch(data.get("checked_at")) if isinstance(data, dict) else None
    return ReplicaHealthReport(served_model_name, expected_replicas, replicas, table_error, checked_at)
