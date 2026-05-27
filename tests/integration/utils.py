import asyncio

import pytest

from swiss_ai_model_launch.cli.healthcheck import ModelHealth, check_model_health
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launcher import Launcher


async def wait_for_job_running(
    launcher: Launcher,
    job_id: int,
    timeout_min: int,
    poll_interval_seconds: int = 15,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_min * 60
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(poll_interval_seconds)
        status = await launcher.get_job_status(job_id)
        print(f"[job {job_id}] status: {status.value}")
        if status == JobStatus.RUNNING:
            return
        if status == JobStatus.TIMEOUT:
            pytest.fail(f"Job {job_id} timed out before becoming RUNNING.")
    pytest.fail(f"Job {job_id} didn't reach RUNNING within {timeout_min} mins.")


async def wait_for_model_healthy(
    launcher: Launcher,
    job_id: int,
    model_name: str,
    api_key: str,
    timeout_min: int,
    poll_interval_seconds: int = 30,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_min * 60
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(poll_interval_seconds)
        status = await launcher.get_job_status(job_id)
        if status != JobStatus.RUNNING:
            pytest.fail(f"Job {job_id} left RUNNING (now {status.value}) before '{model_name}' became HEALTHY.")
        health = await check_model_health(model_name, api_key)
        print(f"[{model_name}] health: {health.value}")
        if health == ModelHealth.HEALTHY:
            return
    pytest.fail(f"'{model_name}' didn't become HEALTHY within {timeout_min} mins.")


async def wait_for_all_replicas_healthy(
    launcher: Launcher,
    job_id: int,
    model_name: str,
    expected_replicas: int,
    timeout_min: int,
    poll_interval_seconds: int = 30,
) -> None:
    """Assert every launched replica is HEALTHY, per the job's own health report.

    The end-to-end check only proves one replica answers through the gateway;
    the model's job writes a per-replica report (``logs/<job_id>/replica_health.json``)
    that we read here and require to show the full expected count all HEALTHY.
    """
    deadline = asyncio.get_event_loop().time() + timeout_min * 60
    last_report = None
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(poll_interval_seconds)
        report = await launcher.get_replica_health(job_id, model_name, expected_replicas)
        if report is None:
            continue  # the in-job checker hasn't written the first report yet
        last_report = report
        print(f"[{model_name}] replica report: {report}")
        if report.error is None and report.found == expected_replicas and report.all_healthy:
            return
    if last_report is None:
        pytest.fail(f"No replica health report appeared for '{model_name}' within {timeout_min} mins.")
    if last_report.error is not None:
        pytest.fail(f"Replica report for '{model_name}' was unreadable: {last_report.error}")
    if last_report.found != expected_replicas:
        pytest.fail(f"Expected {expected_replicas} replica(s) for '{model_name}', report had {last_report.found}.")
    unhealthy = [
        f"rank {r.node_rank}={r.health.value}" for r in last_report.replicas if r.health != ModelHealth.HEALTHY
    ]
    pytest.fail(f"Not all replicas of '{model_name}' became healthy within {timeout_min} mins: {', '.join(unhealthy)}")
