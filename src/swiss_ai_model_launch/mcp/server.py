import asyncio
import getpass
import grp
import os
import re
import subprocess
from typing import Annotated, Any, Literal

import fastmcp
import firecrest as f7t
from fastmcp import Context

from swiss_ai_model_launch.cli.configuration import InitConfig
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, check_model_health
from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher, SlurmLauncher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import JobStatus
from swiss_ai_model_launch.launchers.utils import create_salt

_POLL_INTERVAL_SECONDS = 10
_TERMINAL_STATUSES = {JobStatus.TIMEOUT, JobStatus.UNKNOWN}

_SYSTEM = os.environ.get("SML_SYSTEM")
_PARTITION = os.environ.get("SML_PARTITION")
_RESERVATION = os.environ.get("SML_RESERVATION")

_launcher: Launcher | None = None


mcp = fastmcp.FastMCP(
    name="sml",
    instructions=(
        "Swiss AI Model Launcher MCP server. "
        "Use these tools to list, launch, monitor, and cancel AI model jobs "
        "on HPC clusters via SML. "
        "Call `establish` first to set a default system so you don't need to pass "
        "`system` on every subsequent tool call."
    ),
)


def _build_firecrest_client(config: InitConfig) -> f7t.v2.AsyncFirecrest:
    return f7t.v2.AsyncFirecrest(
        firecrest_url=config.get_non_none_value("firecrest_url"),
        authorization=f7t.ClientCredentialsAuth(
            client_id=config.get_non_none_value("firecrest_client_id"),
            client_secret=config.get_non_none_value("firecrest_client_secret"),
            token_uri=config.get_non_none_value("firecrest_token_uri"),
        ),
    )


async def _create_launcher(
    system: str | None,
    partition: str | None,
    reservation: str | None,
) -> Launcher:
    if not partition:
        raise RuntimeError("No partition specified. Call `establish` first, or set the SML_PARTITION env var.")
    config = InitConfig.load()
    launcher_type = config.get_non_none_value("launcher")
    telemetry_endpoint = config.get_value("telemetry_endpoint")

    if launcher_type == "firecrest":
        if not system:
            raise RuntimeError("No system specified. Call `establish` first, or set the SML_SYSTEM env var.")
        return await FirecRESTLauncher.from_client(
            client=_build_firecrest_client(config),
            system_name=system,
            partition=partition,
            reservation=reservation,
            telemetry_endpoint=telemetry_endpoint,
        )
    elif launcher_type == "slurm":
        return SlurmLauncher(
            system_name="local",
            username=getpass.getuser(),
            account=grp.getgrgid(os.getgid()).gr_name,
            partition=partition,
            reservation=reservation,
            telemetry_endpoint=telemetry_endpoint,
        )
    else:
        raise RuntimeError(f"Launcher type '{launcher_type}' is not supported.")


async def _get_launcher() -> Launcher:
    global _launcher
    if not InitConfig.exists():
        raise RuntimeError(
            "SML is not configured. Open a terminal and run `sml init` first, then restart the MCP server."
        )
    if _launcher is None:
        _launcher = await _create_launcher(system=_SYSTEM, partition=_PARTITION, reservation=_RESERVATION)
    return _launcher


if InitConfig.exists() and InitConfig.load().get_value("launcher") == "firecrest":

    @mcp.tool
    async def list_systems() -> list[dict[str, Any]]:
        """List all HPC systems accessible via FirecREST, along with their available
        SLURM partitions and active reservations. Use this before calling `establish`
        to pick the right system, partition, and reservation.
        """
        config = InitConfig.load()
        client = _build_firecrest_client(config)
        systems = await client.systems()
        result = []
        for system in systems:
            system_name = system["name"]
            partitions, reservations = await asyncio.gather(
                client.partitions(system_name),
                client.reservations(system_name),
            )
            result.append(
                {
                    "system": system_name,
                    "partitions": [p["name"] for p in partitions],
                    "reservations": [r["name"] for r in reservations],
                }
            )
        return result

    @mcp.tool
    async def establish(
        system: Annotated[str, "HPC system name (e.g. 'clariden')."],
        partition: Annotated[str, "SLURM partition (e.g. 'normal')."],
        reservation: Annotated[str | None, "SLURM reservation name."] = None,
    ) -> str:
        """Set the target system, partition, and reservation for all subsequent tool calls.

        Connects to the specified FirecREST system and initialises the launcher for
        this session. All subsequent calls to `launch_model`, `get_job_status`,
        `get_job_logs`, and `cancel_job` will use these settings without requiring
        you to pass them again.

        Call `list_systems` first to discover the available systems, partitions, and
        reservations. Call `establish` again at any point to switch to a different
        system, partition, or reservation — the previous launcher is replaced
        immediately.

        If the environment variables SML_SYSTEM, SML_PARTITION, and SML_RESERVATION
        are already set, the session is initialised automatically on first use.
        Values passed here always take precedence over those variables.
        """
        global _launcher
        _launcher = await _create_launcher(system=system, partition=partition, reservation=reservation)
        parts = [f"system='{system}'", f"partition='{partition}'"]
        if reservation:
            parts.append(f"reservation='{reservation}'")
        return "Session established: " + ", ".join(parts) + "."

else:

    @mcp.tool
    async def list_systems() -> list[dict[str, Any]]:
        """List the local SLURM cluster's available partitions and active reservations.
        Use this before calling `establish` to pick the right partition and reservation.
        """

        def _run(args: list[str]) -> str:
            return subprocess.run(args, capture_output=True, text=True).stdout  # noqa: S603

        partition_out = _run(["scontrol", "show", "partition", "--oneliner"])
        reservation_out = _run(["scontrol", "show", "reservation"])
        partitions = re.findall(r"PartitionName=(\S+)", partition_out)
        reservations = re.findall(r"ReservationName=(\S+)", reservation_out)
        return [{"system": "local", "partitions": partitions, "reservations": reservations}]

    @mcp.tool
    async def establish(  # type: ignore[misc]
        partition: Annotated[str, "SLURM partition (e.g. 'normal')."],
        reservation: Annotated[str | None, "SLURM reservation name."] = None,
    ) -> str:
        """Set the target partition and reservation for all subsequent tool calls.

        Initialises the local SLURM launcher for this session. All subsequent calls
        to `launch_model`, `get_job_status`, `get_job_logs`, and `cancel_job` will
        use these settings without requiring you to pass them again.

        Call `list_systems` first to discover the available partitions and
        reservations on the local cluster. Call `establish` again at any point to
        switch to a different partition or reservation — the previous launcher is
        replaced immediately.

        If the environment variables SML_PARTITION and SML_RESERVATION are already
        set, the session is initialised automatically on first use. Values passed
        here always take precedence over those variables.
        """
        global _launcher
        _launcher = await _create_launcher(system=None, partition=partition, reservation=reservation)
        parts = [f"partition='{partition}'"]
        if reservation:
            parts.append(f"reservation='{reservation}'")
        return "Session established: " + ", ".join(parts) + "."


@mcp.tool
async def list_preconfigured_models() -> Any:
    """List all preconfigured models available for launch."""
    try:
        launcher = await _get_launcher()
    except RuntimeError as e:
        return str(e)
    return [{"model": e.model, "framework": e.framework} for e in await launcher.get_preconfigured_models()]


@mcp.tool
async def launch_preconfigured_model(
    ctx: Context,
    model: Annotated[
        str,
        "Model in 'vendor/name' format (e.g. 'swiss-ai/Apertus-70B'). "
        "Use `list_preconfigured_models` to see available models.",
    ],
    framework: Annotated[Literal["sglang", "vllm"], "Inference framework."],
    workers: Annotated[int, "Number of workers."] = 1,
    time: Annotated[str, "Job time limit in HH:MM:SS format (e.g. '03:00:00')."] = "03:00:00",
    use_router: Annotated[bool, "Enable router for load balancing across workers."] = False,
) -> str:
    """Launch a preconfigured model on an HPC cluster and stream logs until the model is healthy or the job terminates.

    Looks up the model in the catalogue by vendor/name and framework, then submits a
    SLURM job using the preconfigured settings (nodes, environment, framework arguments).
    Use `list_preconfigured_models` to see what is available before calling this tool.

    Streams stdout, stderr, and health status while the job runs. Returns when the
    model is healthy or the job reaches a terminal state.
    """
    try:
        launcher = await _get_launcher()
    except RuntimeError as e:
        return str(e)
    catalogue = await launcher.get_preconfigured_models()
    entry = next(
        (e for e in catalogue if e.model == model and e.framework == framework),
        None,
    )
    if entry is None:
        return (
            f"Model '{model}' with framework '{framework}' was not found in the catalogue. "
            "Use `list_preconfigured_models` to see available models."
        )
    request = LaunchRequest.from_catalog_entry(
        entry,
        workers=workers,
        time=time,
        served_model_name=f"{model}-{create_salt(4)}",
        use_router=use_router,
    )
    job_id, served = await launcher.launch_model(request)
    await ctx.info(f"Job submitted — job_id={job_id}, served_model_name={served}")
    config = InitConfig.load()
    cscs_api_key = config.get_value("cscs_api_key")
    stdout_lines_sent = 0
    stderr_lines_sent = 0
    ever_healthy = False
    seen_active = False
    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        job_status = await launcher.get_job_status(job_id)
        if job_status in (JobStatus.PENDING, JobStatus.RUNNING):
            seen_active = True
        stdout, stderr = await launcher.get_job_logs(job_id)
        stdout_lines = stdout.splitlines() if stdout else []
        stderr_lines = stderr.splitlines() if stderr else []
        for line in stdout_lines[stdout_lines_sent:]:
            await ctx.info(f"[stdout] {line}")
        for line in stderr_lines[stderr_lines_sent:]:
            await ctx.info(f"[stderr] {line}")
        stdout_lines_sent = len(stdout_lines)
        stderr_lines_sent = len(stderr_lines)
        if cscs_api_key:
            health = await check_model_health(served, cscs_api_key)
            if health == ModelHealth.NOT_RESPONDING and not ever_healthy:
                health = ModelHealth.NOT_DEPLOYED
            ever_healthy = ever_healthy or health == ModelHealth.HEALTHY
            await ctx.info(f"[status] job={job_status.value}, health={health.value}")
            if health == ModelHealth.HEALTHY:
                return f"Model {served} is healthy. Job ID: {job_id}."
        else:
            await ctx.info(f"[status] job={job_status.value}")
        if seen_active and job_status in _TERMINAL_STATUSES:
            return f"Job {job_id} terminated with status {job_status.value}."


@mcp.tool
async def get_job_status(
    job_id: Annotated[int, "SLURM job ID to query."],
) -> str:
    """Get the status of a running or queued SML job."""
    try:
        launcher = await _get_launcher()
    except RuntimeError as e:
        return str(e)
    return (await launcher.get_job_status(job_id)).value


@mcp.tool
async def get_job_logs(
    job_id: Annotated[int, "SLURM job ID to retrieve logs for."],
    ctx: Context,
) -> str:
    """Retrieve and stream stdout and stderr logs for a job."""
    try:
        launcher = await _get_launcher()
    except RuntimeError as e:
        return str(e)
    stdout, stderr = await launcher.get_job_logs(job_id)
    if stdout:
        await ctx.info("=== stdout ===")
        for line in stdout.splitlines():
            await ctx.info(line)
    if stderr:
        await ctx.info("=== stderr ===")
        for line in stderr.splitlines():
            await ctx.info(line)
    stdout_lines = len(stdout.splitlines()) if stdout else 0
    stderr_lines = len(stderr.splitlines()) if stderr else 0
    return f"Streamed {stdout_lines} stdout line(s) and {stderr_lines} stderr line(s) for job {job_id}."


@mcp.tool
async def cancel_job(
    job_id: Annotated[int, "SLURM job ID to cancel."],
) -> str:
    """Cancel a running or queued SML job."""
    try:
        launcher = await _get_launcher()
    except RuntimeError as e:
        return str(e)
    await launcher.cancel_job(job_id)
    return f"Job {job_id} cancelled."
