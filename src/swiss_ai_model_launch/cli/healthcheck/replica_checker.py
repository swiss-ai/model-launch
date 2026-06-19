"""Replica-specific health: types + report parsing.

The end-to-end check in ``checker.py`` only proves that *one* replica answers
through the public gateway (OpenTela load-balances randomly across peers). To
confirm that *every* replica is alive, the model's own SLURM job runs an in-job
checker (``assets/replica_health_checker.py``) that probes each replica's
framework ``/health`` directly over the internal network and writes a JSON report
to ``~/.sml/logs/<job_id>/replica_health.json``. The CLI reads that file (locally
for the SLURM launcher, via FirecREST download otherwise).

This module holds the report data types and the parser for that JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth


@dataclass(frozen=True)
class ReplicaHealth:
    """Health of a single replica.

    ``last_seen`` is the epoch-seconds of the last time the in-job checker saw
    the replica HEALTHY (``None`` if never). ``peer_id`` is the OpenTela peer id
    (best-effort, ``None`` if unresolved); ``node_rank``/``node_ip``/``node_host``
    identify the replica head (``node_host`` is its SLURM node name, used to open
    an interactive shell on the node from the TUI).
    """

    health: ModelHealth
    peer_id: str | None = None
    last_seen: int | None = None
    node_rank: int | None = None
    node_ip: str | None = None
    node_host: str | None = None


@dataclass(frozen=True)
class ReplicaHealthReport:
    """Per-replica health for one served model.

    Replicas are reported individually (``replicas``); ``all_healthy`` and
    ``complete`` are convenience aggregates. ``error`` is set when the report
    itself could not be read/parsed (distinct from a replica being unhealthy).
    """

    served_model_name: str
    expected_replicas: int
    replicas: tuple[ReplicaHealth, ...] = ()
    error: str | None = None
    checked_at: int | None = None

    @property
    def found(self) -> int:
        return len(self.replicas)

    @property
    def all_healthy(self) -> bool:
        """True when at least one replica was found and every one is HEALTHY."""
        return (
            self.error is None and bool(self.replicas) and all(r.health == ModelHealth.HEALTHY for r in self.replicas)
        )

    @property
    def complete(self) -> bool:
        """True when all expected replicas are present and HEALTHY."""
        return self.all_healthy and (self.expected_replicas <= 0 or self.found >= self.expected_replicas)


def _to_health(value: object) -> ModelHealth:
    try:
        return ModelHealth(value)
    except ValueError:
        return ModelHealth.ERROR


def _to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _to_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def parse_health_report(report_json: str, served_model_name: str, expected_replicas: int) -> ReplicaHealthReport:
    """Parse the JSON written by the in-job checker into a report.

    On malformed input, returns a report carrying ``error`` rather than raising.
    """
    try:
        data = json.loads(report_json)
    except ValueError as exc:
        return ReplicaHealthReport(served_model_name, expected_replicas, error=f"malformed replica report: {exc}")
    if not isinstance(data, dict):
        return ReplicaHealthReport(served_model_name, expected_replicas, error="malformed replica report")
    raw_replicas = data.get("replicas", [])
    replicas = tuple(
        ReplicaHealth(
            health=_to_health(r.get("health")),
            peer_id=_to_str(r.get("peer_id")),
            last_seen=_to_int(r.get("last_seen")),
            node_rank=_to_int(r.get("node_rank")),
            node_ip=_to_str(r.get("node_ip")),
            node_host=_to_str(r.get("node_host")),
        )
        for r in (raw_replicas if isinstance(raw_replicas, list) else [])
        if isinstance(r, dict)
    )
    return ReplicaHealthReport(served_model_name, expected_replicas, replicas, None, _to_int(data.get("checked_at")))
