"""Self-contained, stdlib-only replica health probe.

This module is never imported by the running CLI/MCP at deploy time. Instead its
*source text* is embedded into a helper SLURM job (see
``cli/healthcheck/replica_checker.py``) and executed on a compute node that sits
inside the cluster's internal network — the only place the OpenTela DNT HTTP API
(``http://<bootstrap-ip>:8092``) is reachable.

The probe fetches the DNT table, selects the peers (replicas) registered under a
given ``served_model_name``, sends a small chat-completion request to each one
through its dedicated ``/v1/p2p/<peer-id>/...`` route, and prints a JSON report
between two sentinel lines so the launcher can recover it from the job log.

It depends only on the Python standard library so it runs under whatever
``python3`` the compute node provides, without the SML package installed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Sentinels delimiting the JSON report inside the job's stdout. Kept in sync with
# ``cli/healthcheck/replica_checker.py`` (a unit test asserts they match).
REPORT_BEGIN = "===SML_REPLICA_REPORT_BEGIN==="
REPORT_END = "===SML_REPLICA_REPORT_END==="

# OpenTela builds each LLM service's identity_group from the framework's
# /v1/models response, one "model=<id>" entry per model (see OpenTela's
# registrar.go). SML launches the framework with --served-model-name <name>, so
# that id — and thus the mesh label — is "model=<served_model_name>". Some
# deployments may additionally tag "served_model_name=<name>"; accept either.
_LABEL_PREFIXES = ("model=", "served_model_name=")
_PROBE_MESSAGE = {"role": "user", "content": "Say hello."}


def _as_epoch(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def matching_peers(table: dict[str, Any], served_model_name: str) -> list[dict[str, Any]]:
    """Return ``{"peer_id", "last_seen"}`` for every replica of ``served_model_name``.

    ``table`` is the parsed ``/v1/dnt/table`` response: a mapping from
    ``/<peer-id>`` to a peer object whose ``service[].identity_group`` array
    carries ``model=<name>`` entries and whose top-level ``last_seen`` is the
    epoch-seconds of the mesh's most recent heartbeat from that peer. Peers are
    returned with ids stripped of the leading slash, in first-seen order,
    de-duplicated.
    """
    targets = tuple(prefix + served_model_name for prefix in _LABEL_PREFIXES)
    peers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, peer in table.items():
        if not isinstance(peer, dict):
            continue
        services = peer.get("service")
        if not isinstance(services, list):
            continue
        registered = any(
            isinstance(svc, dict)
            and isinstance(svc.get("identity_group"), list)
            and any(target in svc["identity_group"] for target in targets)
            for svc in services
        )
        if not registered:
            continue
        peer_id = peer.get("id") or key.lstrip("/")
        if isinstance(peer_id, str) and peer_id and peer_id not in seen:
            seen.add(peer_id)
            peers.append({"peer_id": peer_id, "last_seen": _as_epoch(peer.get("last_seen"))})
    return peers


def peer_ids_for_model(table: dict[str, Any], served_model_name: str) -> list[str]:
    """Return just the peer IDs of every replica registered under ``served_model_name``."""
    return [peer["peer_id"] for peer in matching_peers(table, served_model_name)]


def fetch_table(base_url: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/dnt/table"
    request = urllib.request.Request(url, method="GET")  # noqa: S310 - fixed internal http endpoint
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        raw = response.read()
    data: Any = json.loads(raw)
    return data if isinstance(data, dict) else {}


def check_replica(base_url: str, peer_id: str, served_model_name: str, api_key: str, timeout: float) -> str:
    """Send a chat-completion request to a single replica and classify the outcome.

    Returns a string matching ``ModelHealth``: ``HEALTHY`` on a 2xx response,
    ``NOT_RESPONDING`` on an HTTP error status, ``ERROR`` on a transport/timeout
    failure.
    """
    url = f"{base_url.rstrip('/')}/v1/p2p/{peer_id}/v1/_service/llm/v1/chat/completions"
    payload = json.dumps({"model": served_model_name, "messages": [_PROBE_MESSAGE], "stream": False}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed internal http endpoint
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310
            pass
        return "HEALTHY"
    except urllib.error.HTTPError:
        return "NOT_RESPONDING"
    except (urllib.error.URLError, TimeoutError, OSError):
        return "ERROR"


def run_probe(base_url: str, served_model_name: str, api_key: str, timeout: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "served_model_name": served_model_name,
        "checked_at": int(time.time()),
        "replicas": [],
    }
    try:
        table = fetch_table(base_url, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        report["table_error"] = str(exc)
        return report
    report["replicas"] = [
        {
            "peer_id": peer["peer_id"],
            "last_seen": peer["last_seen"],
            "health": check_replica(base_url, peer["peer_id"], served_model_name, api_key, timeout),
        }
        for peer in matching_peers(table, served_model_name)
    ]
    return report


def main() -> int:
    served_model_name = os.environ.get("SML_SERVED_MODEL_NAME", "")
    api_key = os.environ.get("SML_CSCS_API_KEY", "")
    base_url = os.environ.get("SML_DNT_BASE_URL", "")
    timeout = float(os.environ.get("SML_REPLICA_TIMEOUT") or "10")
    # SML_REPLICA_INTERVAL > 0 keeps the probe running, re-emitting a fresh
    # report every <interval> seconds (the client reads the latest block). 0
    # means probe once and exit.
    interval = float(os.environ.get("SML_REPLICA_INTERVAL") or "0")
    while True:
        report = run_probe(base_url, served_model_name, api_key, timeout)
        sys.stdout.write(f"{REPORT_BEGIN}\n{json.dumps(report)}\n{REPORT_END}\n")
        sys.stdout.flush()
        if interval <= 0:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
