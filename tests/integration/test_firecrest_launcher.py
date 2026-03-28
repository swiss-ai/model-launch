import asyncio
import importlib.resources
import json
import os

import firecrest as f7t
import pytest

from swiss_ai_model_launch.cli.healthcheck import ModelHealth, check_model_health
from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import JobStatus

_LAUNCH_TIMEOUT = 60
_HEALTH_TIMEOUT = 120

_ASSERTS = importlib.resources.files("swiss_ai_model_launch.assets")
_MODEL_JSON = _ASSERTS.joinpath("models.json")
_LAUNCH_REQUESTS = [
    pytest.param(
        LaunchRequest.model_validate(entry),
        id=f"{entry['vendor']}/{entry['model_name']}/{entry['framework']}",
        marks=[pytest.mark.full]
        + (
            [pytest.mark.lightweight] if entry.get("_include_in_lightweight_ci") else []
        ),
    )
    for entry in json.loads(_MODEL_JSON.read_text())
]

_REQUIRED_ENV_VARS = [
    "FIRECREST_URL",
    "FIRECREST_TOKEN_URI",
    "FIRECREST_CLIENT_ID",
    "FIRECREST_CLIENT_SECRET",
    "FIRECREST_SYSTEM",
    "FIRECREST_USERNAME",
    "FIRECREST_ACCOUNT",
    "FIRECREST_PARTITION",
    "CSCS_API_KEY",
    "RESERVATION",
]


@pytest.fixture(scope="function")  # type: ignore[misc]
def env() -> dict[str, str]:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        pytest.fail(
            "Missing required environment variables: " + ", ".join(missing),
            pytrace=False,
        )
    return {v: os.environ[v] for v in _REQUIRED_ENV_VARS}


@pytest.fixture(scope="function")  # type: ignore[misc]
def launcher(env: dict[str, str]) -> FirecRESTLauncher:
    client = f7t.v2.AsyncFirecrest(
        firecrest_url=env["FIRECREST_URL"],
        authorization=f7t.ClientCredentialsAuth(
            client_id=env["FIRECREST_CLIENT_ID"],
            client_secret=env["FIRECREST_CLIENT_SECRET"],
            token_uri=env["FIRECREST_TOKEN_URI"],
        ),
    )
    return FirecRESTLauncher(
        client=client,
        system_name=env["FIRECREST_SYSTEM"],
        username=env["FIRECREST_USERNAME"],
        account=env["FIRECREST_ACCOUNT"],
        partition=env["FIRECREST_PARTITION"],
        reservation=env["RESERVATION"] or None,
    )


@pytest.fixture(scope="function")  # type: ignore[misc]
def cscs_api_key(env: dict[str, str]) -> str:
    return env["CSCS_API_KEY"]


async def _wait_for_job_running(
    launcher: FirecRESTLauncher,
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


async def _wait_for_model_healthy(
    served_model_name: str,
    api_key: str,
    timeout_min: int,
    poll_interval_seconds: int = 30,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_min * 60
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(poll_interval_seconds)
        health = await check_model_health(served_model_name, api_key)
        print(f"[{served_model_name}] health: {health.value}")
        if health == ModelHealth.HEALTHY:
            return
    pytest.fail(
        f"Model '{served_model_name}' didn't become HEALTHY within {timeout_min} mins."
    )


@pytest.mark.parametrize("launch_request", _LAUNCH_REQUESTS)  # type: ignore[misc]
async def test_launch_apertus_and_health(
    launcher: FirecRESTLauncher,
    cscs_api_key: str,
    launch_request: LaunchRequest,
) -> None:
    job_id, served_model_name = await launcher.launch_model(launch_request)
    print(f"Submitted job_id={job_id}, served_model_name={served_model_name}")

    assert isinstance(job_id, int)
    assert served_model_name

    try:
        await _wait_for_job_running(launcher, job_id, _LAUNCH_TIMEOUT)
        await _wait_for_model_healthy(served_model_name, cscs_api_key, _HEALTH_TIMEOUT)
    finally:
        await launcher.cancel_job(job_id)
