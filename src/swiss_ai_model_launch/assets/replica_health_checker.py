"""In-job replica health checker.

Runs on the batch node of the model's *own* SLURM job (its source is embedded
into the rendered ``master.sh``). The batch node shares the job's internal
network, so this checks every replica directly — no OpenTela bootstrap, no p2p
routing, no API key. Each cycle it:

  * GETs ``http://<replica-head-ip>:<framework-port>/health`` for each replica,
  * once a replica is healthy, resolves its OpenTela peer id by running
    ``/v1/self`` *on that node* via a one-off ``srun`` step (the endpoint is
    localhost-only, and the DNT table carries no internal IP to match against),
  * records the last time the replica was seen HEALTHY,

then writes a JSON report **atomically** (temp file + ``os.replace``) to
``SML_HEALTH_REPORT_PATH`` so a concurrent reader (local or via FirecREST
download) never observes a half-written file. Loops every ``SML_HEALTH_INTERVAL``
seconds.

Stdlib only — runs under whatever ``python3`` the node provides, without the SML
package installed. HPC batch hosts can ship a Python as old as 3.6, so this file
must stay 3.6-compatible: NO ``from __future__ import annotations`` (3.7+), NO
PEP 604 ``X | Y`` unions or builtin-generic subscripts (``list[str]``) in
evaluated positions, and NO ``subprocess.run(capture_output=/text=)`` (3.7+).
Use ``typing`` aliases and ``stdout=/stderr=PIPE`` + ``universal_newlines`` instead.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Peer ids are stable, so stop retrying resolution after a few attempts to avoid
# launching a job step every cycle when /v1/self is unreachable.
_MAX_PEER_ATTEMPTS = 5
_SRUN_TIMEOUT_SECONDS = 30.0


def _http_get(url: str, timeout: float) -> Optional[Tuple[int, bytes]]:
    request = urllib.request.Request(url, method="GET")  # noqa: S310 - fixed internal http endpoint
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return (response.getcode() or 0), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def check_health(node_ip: str, framework_port: int, timeout: float) -> str:
    """Probe a replica's framework /health. Returns a ``ModelHealth`` string."""
    result = _http_get(f"http://{node_ip}:{framework_port}/health", timeout)
    if result is None:
        return "ERROR"
    status, _ = result
    return "HEALTHY" if 200 <= status < 300 else "NOT_RESPONDING"


def _self_fetch_code(ocf_port: int) -> str:
    # Executed on the target node: /v1/self is localhost-only and returns the
    # node's own Peer record, whose "id" is the libp2p peer id.
    return (
        "import json, urllib.request; "
        f"print(json.load(urllib.request.urlopen('http://localhost:{ocf_port}/v1/self', timeout=5)).get('id') or '')"
    )


def resolve_peer_id(host: str, ocf_port: int, timeout: float) -> Optional[str]:
    """Fetch a node's own OpenTela peer id by querying ``/v1/self`` on that node.

    ``/v1/self`` only answers to localhost, so we run a one-off ``srun --overlap``
    step on the node itself (sharing the live allocation). Returns ``None`` on any
    failure (no host, srun unavailable, timeout, OCF not up).
    """
    if not host:
        return None
    cmd = [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={host}",
        "python3",
        "-c",
        _self_fetch_code(ocf_port),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    peer_id = result.stdout.strip()
    return peer_id or None


def build_report(
    replica_ips: List[str],
    replica_hosts: List[str],
    nodes_per_replica: int,
    framework_port: int,
    ocf_port: int,
    timeout: float,
    peer_ids: Dict[int, str],
    peer_attempts: Dict[int, int],
    last_seen: Dict[int, int],
    now: int,
) -> Dict[str, Any]:
    """Probe every replica once and assemble the report dict.

    The cache dicts are mutated in place across cycles: ``peer_ids`` is resolved
    once a replica is healthy (peer ids are stable), ``last_seen`` freezes at the
    last HEALTHY observation so a downed replica's "ago" keeps growing, and a
    replica that has never been healthy reads as ``NOT_DEPLOYED`` (still starting)
    rather than a failure.
    """
    replicas: List[Dict[str, Any]] = []
    for index, node_ip in enumerate(replica_ips):
        host = replica_hosts[index] if index < len(replica_hosts) else ""
        health = check_health(node_ip, framework_port, timeout)
        if health == "HEALTHY":
            last_seen[index] = now
            if index not in peer_ids and peer_attempts.get(index, 0) < _MAX_PEER_ATTEMPTS:
                peer_attempts[index] = peer_attempts.get(index, 0) + 1
                resolved = resolve_peer_id(host, ocf_port, _SRUN_TIMEOUT_SECONDS)
                if resolved:
                    peer_ids[index] = resolved
        elif index not in last_seen:
            health = "NOT_DEPLOYED"
        replicas.append(
            {
                "node_rank": index * nodes_per_replica,
                "node_ip": node_ip,
                "peer_id": peer_ids.get(index),
                "health": health,
                "last_seen": last_seen.get(index),
            }
        )
    return {"checked_at": now, "replicas": replicas}


def write_report_atomically(report: Dict[str, Any], report_path: str) -> None:
    directory = os.path.dirname(report_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{report_path}.tmp"
    with open(tmp_path, "w") as handle:
        json.dump(report, handle)
    os.replace(tmp_path, report_path)  # atomic on POSIX (same directory)


def main() -> int:
    report_path = os.environ["SML_HEALTH_REPORT_PATH"]
    framework_port = int(os.environ.get("SML_HEALTH_FRAMEWORK_PORT") or "8080")
    ocf_port = int(os.environ.get("SML_HEALTH_OCF_PORT") or "8092")
    interval = float(os.environ.get("SML_HEALTH_INTERVAL") or "30")
    timeout = float(os.environ.get("SML_HEALTH_TIMEOUT") or "10")
    nodes_per_replica = int(os.environ.get("SML_HEALTH_NODES_PER_REPLICA") or "1")
    replica_ips = os.environ.get("SML_HEALTH_REPLICA_IPS", "").split()
    replica_hosts = os.environ.get("SML_HEALTH_REPLICA_HOSTS", "").split()

    peer_ids: Dict[int, str] = {}
    peer_attempts: Dict[int, int] = {}
    last_seen: Dict[int, int] = {}
    while True:
        # Never let a transient failure (e.g. a momentary write error) kill the
        # checker — log it and keep going so the report self-heals next cycle.
        try:
            report = build_report(
                replica_ips,
                replica_hosts,
                nodes_per_replica,
                framework_port,
                ocf_port,
                timeout,
                peer_ids,
                peer_attempts,
                last_seen,
                int(time.time()),
            )
            write_report_atomically(report, report_path)
        except Exception as exc:  # resilience: keep the loop alive on any error
            sys.stderr.write(f"replica health checker: iteration failed: {exc}\n")
            sys.stderr.flush()
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
