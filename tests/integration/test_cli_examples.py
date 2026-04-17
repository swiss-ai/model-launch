import asyncio
import os
import re
from pathlib import Path

import firecrest as f7t
import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from tests.integration.utils import wait_for_job_running, wait_for_model_healthy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples" / "clariden" / "cli"

_JOB_SUBMISSION_TIMEOUT_SEC = 180
_LAUNCH_TIMEOUT_MIN = 60
_HEALTH_TIMEOUT_MIN = 120

# ROCm examples are excluded: there is no FirecREST integration for ROCm targets,
# so `sml advanced` can't actually submit those jobs from CI.
_EXCLUDE_PATTERNS = ("rocm", "experiment-")


def _discover_examples() -> list[Path]:
    scripts = sorted(_EXAMPLES_DIR.glob("**/*.sh"))
    return [
        s for s in scripts if not any(pat in str(s.relative_to(_EXAMPLES_DIR)).lower() for pat in _EXCLUDE_PATTERNS)
    ]


_EXAMPLE_SCRIPTS = [
    pytest.param(p, id=str(p.relative_to(_REPO_ROOT)), marks=pytest.mark.full) for p in _discover_examples()
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
async def cancel_launcher(env: dict[str, str]) -> FirecRESTLauncher:
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


@pytest.mark.parametrize("script", _EXAMPLE_SCRIPTS)  # type: ignore[misc]
async def test_cli_example_launches_and_health(
    script: Path,
    cancel_launcher: FirecRESTLauncher,
    env: dict[str, str],
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "bash",
        str(script),
        cwd=_REPO_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=_JOB_SUBMISSION_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        proc.kill()
        pytest.fail(f"{script.name} did not complete submission within {_JOB_SUBMISSION_TIMEOUT_SEC}s")

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    assert proc.returncode == 0, f"{script.name} exited {proc.returncode}\n---\n{stdout}"

    job_match = re.search(r"Job submitted:\s*(\d+)", stdout)
    assert job_match, f"{script.name}: no 'Job submitted: <id>' line in output\n---\n{stdout}"
    job_id = int(job_match.group(1))

    served_match = re.search(r"Served model name:\s*(\S+)", stdout)
    assert served_match, f"{script.name}: no 'Served model name: <name>' line in output\n---\n{stdout}"
    served_model_name = served_match.group(1)

    try:
        await wait_for_job_running(cancel_launcher, job_id, _LAUNCH_TIMEOUT_MIN)
        await wait_for_model_healthy(served_model_name, env["SML_CSCS_API_KEY"], _HEALTH_TIMEOUT_MIN)
    finally:
        await cancel_launcher.cancel_job(job_id)
