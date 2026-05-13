import importlib.resources
import json
import os

import firecrest as f7t
import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from tests.integration.utils import wait_for_job_running, wait_for_model_healthy

_LAUNCH_TIMEOUT = 60
_HEALTH_TIMEOUT = 120

_ASSERTS = importlib.resources.files("swiss_ai_model_launch.assets")
_MODEL_JSON = _ASSERTS.joinpath("models.json")
_LAUNCH_REQUESTS = [
    pytest.param(
        LaunchRequest(
            **ModelCatalogEntry.model_validate(entry).model_dump(),
            workers=1,
            time="03:00:00",
        ),
        id=f"{entry['model']}/{entry['framework']}",
        marks=[pytest.mark.std, pytest.mark.comprehensive]
        + ([pytest.mark.lightweight] if entry.get("_include_in_lightweight_ci") else []),
    )
    for entry in json.loads(_MODEL_JSON.read_text())
]

_REQUIRED_ENV_VARS = [
    "SML_CSCS_API_KEY",
    "SML_FIRECREST_CLIENT_ID",
    "SML_FIRECREST_CLIENT_SECRET",
    "SML_FIRECREST_SYSTEM",
    "SML_FIRECREST_TOKEN_URI",
    "SML_FIRECREST_URL",
    "SML_PARTITION",
    "SML_RESERVATION",
]


@pytest.fixture(scope="function")  # type: ignore[misc]
def env() -> dict[str, str]:
    missing = [v for v in _REQUIRED_ENV_VARS if os.environ.get(v) is None]
    if missing:
        pytest.fail(
            "Missing required environment variables: " + ", ".join(missing),
            pytrace=False,
        )
    return {v: os.environ[v] for v in _REQUIRED_ENV_VARS}


@pytest.fixture(scope="function")  # type: ignore[misc]
async def launcher(env: dict[str, str]) -> FirecRESTLauncher:
    client = f7t.v2.AsyncFirecrest(
        firecrest_url=env["SML_FIRECREST_URL"],
        authorization=f7t.ClientCredentialsAuth(
            client_id=env["SML_FIRECREST_CLIENT_ID"],
            client_secret=env["SML_FIRECREST_CLIENT_SECRET"],
            token_uri=env["SML_FIRECREST_TOKEN_URI"],
        ),
    )
    return await FirecRESTLauncher.from_client(
        client=client,
        system_name=env["SML_FIRECREST_SYSTEM"],
        partition=env["SML_PARTITION"],
        reservation=env["SML_RESERVATION"] or None,
    )


@pytest.fixture(scope="function")  # type: ignore[misc]
def cscs_api_key(env: dict[str, str]) -> str:
    return env["SML_CSCS_API_KEY"]


@pytest.mark.parametrize("launch_request", _LAUNCH_REQUESTS)  # type: ignore[misc]
async def test_launch_apertus_and_health(
    launcher: FirecRESTLauncher,
    cscs_api_key: str,
    launch_request: LaunchRequest,
) -> None:
    job_id, served_model_name = await launcher.launch_model(launch_request)

    assert isinstance(job_id, int)
    assert served_model_name

    try:
        await wait_for_job_running(launcher, job_id, _LAUNCH_TIMEOUT)
        await wait_for_model_healthy(served_model_name, cscs_api_key, _HEALTH_TIMEOUT)
    finally:
        await launcher.cancel_job(job_id)
