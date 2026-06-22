import os
from collections.abc import AsyncIterator
from pathlib import Path

import firecrest as f7t
import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.loadtest.cluster import ClusterLoadtestConfig, submit_cluster_loadtest
from swiss_ai_model_launch.loadtest.models import LoadtestConfig, ServerConfig
from swiss_ai_model_launch.loadtest.setup import DEFAULT_CLUSTER_CONTAINER_IMAGE, K6_SCRIPT
from tests.integration.utils import wait_for_job_running

_LOADTEST_RUNNING_TIMEOUT_MIN = 20
_LOADTEST_SERVER_URL = "https://api.swissai.svc.cscs.ch"

_REQUIRED_ENV_VARS = [
    "SML_CSCS_API_KEY",
    "SML_FIRECREST_CLIENT_ID",
    "SML_FIRECREST_CLIENT_SECRET",
    "SML_SYSTEM",
    "SML_FIRECREST_TOKEN_URI",
    "SML_FIRECREST_URL",
    "SML_LOADTEST_MODEL",
    "SML_LOADTEST_PROMPTS_FILE",
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
async def launcher(env: dict[str, str]) -> AsyncIterator[FirecRESTLauncher]:
    client = f7t.v2.AsyncFirecrest(
        firecrest_url=env["SML_FIRECREST_URL"],
        authorization=f7t.ClientCredentialsAuth(
            client_id=env["SML_FIRECREST_CLIENT_ID"],
            client_secret=env["SML_FIRECREST_CLIENT_SECRET"],
            token_uri=env["SML_FIRECREST_TOKEN_URI"],
            min_token_validity=90,
        ),
    )
    try:
        yield await FirecRESTLauncher.from_client(
            client=client,
            system_name=env["SML_SYSTEM"],
            partition=env["SML_PARTITION"],
            reservation=env["SML_RESERVATION"] or None,
        )
    finally:
        await client.close_session()


@pytest.mark.comprehensive
async def test_submit_cluster_loadtest_starts_cluster_job(
    launcher: FirecRESTLauncher,
    env: dict[str, str],
    tmp_path: Path,
) -> None:
    submission = await submit_cluster_loadtest(
        launcher=launcher,
        server=ServerConfig(
            url=os.environ.get("SML_LOADTEST_SERVER_URL", _LOADTEST_SERVER_URL),
            api_key=env["SML_CSCS_API_KEY"],
            model=env["SML_LOADTEST_MODEL"],
            is_swissai=True,
        ),
        bench=LoadtestConfig(
            scenario=os.environ.get("SML_LOADTEST_SCENARIO", "throughput"),
            think_time="0",
            max_tokens=os.environ.get("SML_LOADTEST_MAX_TOKENS", "16"),
        ),
        k6_script=K6_SCRIPT,
        prompts_file=Path(env["SML_LOADTEST_PROMPTS_FILE"]),
        summary_path=tmp_path / "summary.json",
        cluster=ClusterLoadtestConfig(
            container_image=str(DEFAULT_CLUSTER_CONTAINER_IMAGE),
            wait=False,
            reservation=env["SML_RESERVATION"] or None,
        ),
    )

    try:
        await wait_for_job_running(launcher, submission.job_id, _LOADTEST_RUNNING_TIMEOUT_MIN)
    finally:
        await launcher.cancel_job(submission.job_id)
