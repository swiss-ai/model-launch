import importlib.resources
import json
import os
from collections.abc import AsyncIterator

import firecrest as f7t
import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from tests.integration.utils import wait_for_all_replicas_healthy, wait_for_job_running, wait_for_model_healthy

# Timeouts in minutes. Overridable via env so a failing run can be made to time
# out quickly for debugging (e.g. SML_TEST_REPLICA_TIMEOUT=5) without changing
# the committed defaults.
_LAUNCH_TIMEOUT = int(os.environ.get("SML_TEST_LAUNCH_TIMEOUT") or "60")
_HEALTH_TIMEOUT = int(os.environ.get("SML_TEST_HEALTH_TIMEOUT") or "60")
_REPLICA_TIMEOUT = int(os.environ.get("SML_TEST_REPLICA_TIMEOUT") or "60")

_ASSERTS = importlib.resources.files("swiss_ai_model_launch.assets")
_MODEL_JSON = _ASSERTS.joinpath("models.json")
_CATALOG_ENTRIES = json.loads(_MODEL_JSON.read_text())

_LAUNCH_REQUESTS = [
    pytest.param(
        LaunchRequest(
            **ModelCatalogEntry.model_validate(entry).model_dump(),
            replicas=1,
            time="03:00:00",
        ),
        id=f"{entry['model']}/{entry['framework']}",
        marks=[pytest.mark.comprehensive]
        + ([pytest.mark.lightweight] if entry.get("_include_in_lightweight_ci") else []),
    )
    for entry in _CATALOG_ENTRIES
]

# Std suite: Apertus 8B (sglang + vllm), Apertus 70B (sglang), Gemma 3 27B (vllm),
# and GLM-5.1-FP8 (sglang) across three topologies. The with/without-router axis is
# meaningful for multi-replica configurations. vllm/2r-router combinations are skipped
# because the multi-replica router is sglang-only (master script invokes
# sglang_router.launch_router, which the vllm container does not ship).
_STD_MODELS = (
    "swiss-ai/Apertus-8B-Instruct-2509",
    "swiss-ai/Apertus-70B-Instruct-2509",
    "google/gemma-3-27b-it",
    "zai-org/GLM-5.1-FP8",
)
_STD_CONFIGS: list[tuple[str, int, bool]] = [
    # (config_id, replicas, use_router)
    ("1r", 1, False),
    ("2r", 2, False),
    ("2r-router", 2, True),
]
_STD_LAUNCH_REQUESTS = [
    pytest.param(
        LaunchRequest.from_catalog_entry(
            ModelCatalogEntry.model_validate(entry),
            replicas=replicas,
            time="04:00:00",
            router="sglang" if use_router else "opentela",
        ),
        id=f"{entry['model']}/{entry['framework']}/{config_id}",
        marks=[pytest.mark.std],
    )
    for entry in _CATALOG_ENTRIES
    if entry["model"] in _STD_MODELS
    for config_id, replicas, use_router in _STD_CONFIGS
    if not (entry["framework"] == "vllm" and use_router)
]

_REQUIRED_ENV_VARS = [
    "SML_CSCS_API_KEY",
    "SML_FIRECREST_CLIENT_ID",
    "SML_FIRECREST_CLIENT_SECRET",
    "SML_SYSTEM",
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


@pytest.fixture(scope="function")  # type: ignore[misc]
def cscs_api_key(env: dict[str, str]) -> str:
    return env["SML_CSCS_API_KEY"]


@pytest.mark.parametrize("launch_request", _LAUNCH_REQUESTS + _STD_LAUNCH_REQUESTS)  # type: ignore[misc]
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
        await wait_for_model_healthy(launcher, job_id, served_model_name, cscs_api_key, _HEALTH_TIMEOUT)
        # For multi-replica launches the e2e check only proves *one* replica
        # answers, so additionally confirm every replica is healthy.
        if launch_request.replicas > 1:
            await wait_for_all_replicas_healthy(
                launcher,
                job_id,
                served_model_name,
                launch_request.replicas,
                _REPLICA_TIMEOUT,
            )
    finally:
        await launcher.cancel_job(job_id)
