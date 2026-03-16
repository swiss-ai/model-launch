import asyncio
import os

import firecrest as f7t
import pytest

from swiss_ai_model_launch.cli.healthcheck import ModelHealth, check_model_health
from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import JobStatus

APERTUS_REQUEST = LaunchRequest(
    vendor="swiss-ai",
    model_name="Apertus-8B-Instruct-2509",
    framework="sglang",
    environment=None,
    workers=1,
    nodes_per_worker=1,
    time="00:30:00",
)

_REQUIRED_ENV_VARS = [
    "FIRECREST_URL",
    "FIRECREST_TOKEN_URI",
    "FIRECREST_CLIENT_ID",
    "FIRECREST_CLIENT_SECRET",
    "FIRECREST_SYSTEM",
    "FIRECREST_ACCOUNT",
    "FIRECREST_PARTITION",
    "CSCS_API_KEY",
]

_missing_env_vars = [v for v in _REQUIRED_ENV_VARS if os.environ.get(v) is None]
if _missing_env_vars:
    pytest.fail(
        "Missing required environment variables: " + ", ".join(_missing_env_vars),
        pytrace=False,
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def launcher() -> FirecRESTLauncher:
    client = f7t.v2.AsyncFirecrest(
        firecrest_url=os.environ["FIRECREST_URL"],
        authorization=f7t.ClientCredentialsAuth(
            client_id=os.environ["FIRECREST_CLIENT_ID"],
            client_secret=os.environ["FIRECREST_CLIENT_SECRET"],
            token_uri=os.environ["FIRECREST_TOKEN_URI"],
        ),
    )
    return FirecRESTLauncher(
        client=client,
        system_name=os.environ["FIRECREST_SYSTEM"],
        username=os.environ["FIRECREST_USERNAME"],
        account=os.environ["FIRECREST_ACCOUNT"],
        partition=os.environ["FIRECREST_PARTITION"],
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def cscs_api_key() -> str:
    return os.environ["CSCS_API_KEY"]


async def _wait_for_job_running(
    launcher: FirecRESTLauncher,
    job_id: int,
    timeout_minutes: int,
    poll_interval_seconds: int = 15,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_minutes * 60
    while asyncio.get_event_loop().time() < deadline:
        status = await launcher.get_job_status(job_id)
        print(f"[job {job_id}] status: {status.value}")
        if status == JobStatus.RUNNING:
            return
        if status == JobStatus.TIMEOUT:
            pytest.fail(f"Job {job_id} timed out before becoming RUNNING.")
        await asyncio.sleep(poll_interval_seconds)
    pytest.fail(
        f"Job {job_id} did not reach RUNNING within the deadline of {timeout_minutes} "
        "minutes."
    )


async def _wait_for_model_healthy(
    served_model_name: str,
    api_key: str,
    timeout_minutes: int,
    poll_interval_seconds: int = 30,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_minutes * 60
    while asyncio.get_event_loop().time() < deadline:
        health = await check_model_health(served_model_name, api_key)
        print(f"[{served_model_name}] health: {health.value}")
        if health == ModelHealth.HEALTHY:
            return
        await asyncio.sleep(poll_interval_seconds)
    pytest.fail(
        f"Model '{served_model_name}' did not become HEALTHY within the deadline of "
        f"{timeout_minutes} minutes."
    )


async def test_launch_apertus_and_health(
    launcher: FirecRESTLauncher, cscs_api_key: str
) -> None:
    launch_timeout = int(os.environ.get("LAUNCH_TIMEOUT_MINUTES", "10"))
    health_timeout = int(os.environ.get("HEALTH_TIMEOUT_MINUTES", "20"))

    job_id, served_model_name = await launcher.launch_model(APERTUS_REQUEST)
    print(f"Submitted job_id={job_id}, served_model_name={served_model_name}")

    assert isinstance(job_id, int)
    assert served_model_name

    await _wait_for_job_running(launcher, job_id, launch_timeout)
    await _wait_for_model_healthy(served_model_name, cscs_api_key, health_timeout)
