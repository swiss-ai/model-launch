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


async def check_all_replicas_healthy(
    launcher: Launcher,
    model_name: str,
    api_key: str,
    expected_replicas: int,
    timeout_min: int,
) -> None:
    """Assert that every replica registered under ``model_name`` is HEALTHY.

    The end-to-end check only proves one replica answers through the gateway;
    this submits a helper job that probes each replica via the DNT mesh and
    verifies the full expected count is alive.
    """
    report = await launcher.check_replicas_health(
        model_name,
        api_key,
        expected_replicas=expected_replicas,
        timeout_seconds=timeout_min * 60,
    )
    print(f"[{model_name}] replica report: {report}")
    if report.table_error is not None:
        pytest.fail(f"Replica check for '{model_name}' could not query the DNT table: {report.table_error}")
    if report.found != expected_replicas:
        pytest.fail(f"Expected {expected_replicas} replica(s) for '{model_name}', DNT registered {report.found}.")
    unhealthy = [r for r in report.replicas if r.health != ModelHealth.HEALTHY]
    if unhealthy:
        details = ", ".join(f"{r.peer_id}={r.health.value}" for r in unhealthy)
        pytest.fail(f"Not all replicas of '{model_name}' are healthy: {details}")
