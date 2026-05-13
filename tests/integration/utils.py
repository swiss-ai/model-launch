import asyncio

import pytest

from swiss_ai_model_launch.cli.healthcheck import ModelHealth, check_model_health
from swiss_ai_model_launch.launchers.launcher import JobStatus, Launcher


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
    model_name: str,
    api_key: str,
    timeout_min: int,
    poll_interval_seconds: int = 30,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_min * 60
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(poll_interval_seconds)
        health = await check_model_health(model_name, api_key)
        print(f"[{model_name}] health: {health.value}")
        if health == ModelHealth.HEALTHY:
            return
    pytest.fail(f"'{model_name}' didn't become HEALTHY within {timeout_min} mins.")
