import asyncio
import getpass
import grp
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import fastmcp
import firecrest as f7t
import yaml
from fastmcp import Context

from swiss_ai_model_launch.cli.healthcheck import ModelHealth, check_model_health
from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher, SlurmLauncher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import JobStatus
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.utils import create_salt

if TYPE_CHECKING:
    from swiss_ai_model_launch.cli.configuration import InitConfig

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

_CONFIG_DIR = Path(d) if (d := os.environ.get("SML_CONFIG_DIR")) else Path.home() / ".sml"
_CONTEXT_FILE = _CONFIG_DIR / "context.yml"

_SYSTEM = os.environ.get("SML_SYSTEM")
_PARTITION = os.environ.get("SML_PARTITION")
_RESERVATION = os.environ.get("SML_RESERVATION")

_POLL_INTERVAL_SECONDS = 10
_TERMINAL_STATUSES = {JobStatus.TIMEOUT, JobStatus.UNKNOWN}


def _load_context() -> dict[str, str]:
    if not _CONTEXT_FILE.exists():
        return {}
    with _CONTEXT_FILE.open() as f:
        return yaml.safe_load(f) or {}


def _save_context(context: dict[str, str]) -> None:
    _CONTEXT_FILE.parent.mkdir(exist_ok=True)
    with _CONTEXT_FILE.open("w") as f:
        yaml.dump(context, f)


def _load_config() -> "InitConfig":
    from swiss_ai_model_launch.cli.configuration import InitConfig

    if not InitConfig.exists():
        raise RuntimeError("SML is not configured. Run `sml init` first.")
    return InitConfig.load()


def _build_firecrest_client(config: "InitConfig") -> f7t.v2.AsyncFirecrest:
    return f7t.v2.AsyncFirecrest(
        firecrest_url=config.get_non_none_value("firecrest_url"),
        authorization=f7t.ClientCredentialsAuth(
            client_id=config.get_non_none_value("firecrest_client_id"),
            client_secret=config.get_non_none_value("firecrest_client_secret"),
            token_uri=config.get_non_none_value("firecrest_token_uri"),
        ),
    )


async def _make_launcher(
    system: str | None = None,
    partition: str | None = None,
    reservation: str | None = None,
) -> Launcher:
    config = _load_config()
    launcher_type = config.get_non_none_value("launcher")
    telemetry_endpoint = config.get_value("telemetry_endpoint")

    context = _load_context()
    system = system or _SYSTEM or context.get("system_name")
    partition = partition or _PARTITION or context.get("default_partition")
    reservation = reservation or _RESERVATION

    if partition is None:
        raise ValueError("`partition` is required. Pass it explicitly, set SML_PARTITION, or call `establish`.")

    if launcher_type == "firecrest":
        if system is None:
            raise ValueError("`system` is required when using a FirecREST-based launcher.")
        client = _build_firecrest_client(config)
        return await FirecRESTLauncher.from_client(
            client=client,
            system_name=system,
            partition=partition,
            reservation=reservation,
            telemetry_endpoint=telemetry_endpoint,
        )

    if launcher_type == "slurm":
        return SlurmLauncher(
            system_name="local",
            username=getpass.getuser(),
            account=grp.getgrgid(os.getgid()).gr_name,
            partition=partition,
            reservation=reservation,
            telemetry_endpoint=telemetry_endpoint,
        )

    raise NotImplementedError(f"Launcher type '{launcher_type}' is not supported.")


# establish's signature varies based on pre-configured env vars:
#   0 — nothing pre-configured      → system, partition, reservation
#   1 — SML_SYSTEM                  → partition, reservation
#   2+ — SML_SYSTEM + SML_PARTITION → reservation only

if _SYSTEM and _PARTITION:
    _level = 2
elif _SYSTEM:
    _level = 1
else:
    _level = 0

_ESTABLISH_DOC = (
    "Set session defaults for all subsequent MCP tool calls. "
    "If the user has not mentioned a SLURM reservation, ask them before proceeding — "
    "reservations are not persisted and must be passed explicitly on each call that needs one."
)


async def _establish_impl(system: str | None, partition: str | None, reservation: str | None) -> str:
    context = _load_context()
    if system is not None:
        context["system_name"] = system
    if partition is not None:
        context["default_partition"] = partition
    _save_context(context)
    msg = f"Established default system: '{system or _SYSTEM}'"
    if partition or _PARTITION:
        msg += f", partition: '{partition or _PARTITION}'"
    if reservation:
        msg += f", reservation: '{reservation}' (not persisted — pass it explicitly on each call)"
    return msg + "."


if _level == 0:

    async def establish(
        system: Annotated[str, "HPC system name (e.g. 'daint')."],
        partition: Annotated[str | None, "Default SLURM partition (e.g. 'normal')."] = None,
        reservation: Annotated[str | None, "SLURM reservation for this session."] = None,
    ) -> str:
        return await _establish_impl(system, partition, reservation)

elif _level == 1:

    async def establish(  # type: ignore[misc]
        partition: Annotated[str | None, "Default SLURM partition (e.g. 'normal')."] = None,
        reservation: Annotated[str | None, "SLURM reservation for this session."] = None,
    ) -> str:
        return await _establish_impl(None, partition, reservation)

else:

    async def establish(  # type: ignore[misc]
        reservation: Annotated[str | None, "SLURM reservation for this session."] = None,
    ) -> str:
        return await _establish_impl(None, None, reservation)


establish.__doc__ = _ESTABLISH_DOC
mcp.add_tool(establish)


@mcp.tool
async def list_preconfigured_models(
    system: Annotated[str | None, "Target HPC system name (required for FirecREST launchers)."] = None,
    partition: Annotated[str | None, "SLURM partition (defaults to the one set via `establish`)."] = None,
) -> list[dict[str, Any]]:
    """List all preconfigured models available for launch."""
    launcher = await _make_launcher(system=system, partition=partition)
    return [e.model_dump() for e in await launcher.get_preconfigured_models()]


@mcp.tool
async def launch_model(
    vendor: Annotated[str, "Model vendor (e.g. 'swiss-ai', 'Qwen')."],
    model_name: Annotated[str, "Model name (e.g. 'Apertus-70B')."],
    framework: Annotated[Literal["sglang", "vllm"], "Inference framework."],
    ctx: Context,
    system: Annotated[str | None, "Target HPC system name (required for FirecREST launchers)."] = None,
    partition: Annotated[str | None, "SLURM partition (defaults to the one set via `establish`)."] = None,
    reservation: Annotated[str | None, "SLURM reservation name (optional)."] = None,
    workers: Annotated[int, "Number of workers."] = 1,
    time: Annotated[str, "Job time limit in HH:MM:SS format."] = "03:00:00",
    use_router: Annotated[bool, "Enable router for load balancing across workers."] = False,
    nodes_per_worker: Annotated[int | None, "Override nodes per worker (defaults to catalogue value)."] = None,
    framework_args: Annotated[str | None, "Override framework arguments (defaults to catalogue value)."] = None,
    pre_launch_cmds: Annotated[str | None, "Override pre-launch commands (defaults to catalogue value)."] = None,
) -> str:
    """Launch a model on an HPC cluster and stream logs until the model is healthy or the job terminates."""
    launcher = await _make_launcher(system=system, partition=partition, reservation=reservation)
    catalogue = await launcher.get_preconfigured_models()
    entry = next(
        (e for e in catalogue if e.vendor == vendor and e.model_name == model_name and e.framework == framework),
        ModelCatalogEntry(vendor=vendor, model_name=model_name, framework=framework),
    )
    overrides = {
        k: v
        for k, v in {
            "nodes_per_worker": nodes_per_worker,
            "framework_args": framework_args,
            "pre_launch_cmds": pre_launch_cmds,
        }.items()
        if v is not None
    }
    if overrides:
        entry = entry.model_copy(update=overrides)
    request = LaunchRequest.from_catalog_entry(
        entry,
        workers=workers,
        time=time,
        served_model_name=f"{vendor}/{model_name}-{create_salt(4)}",
        use_router=use_router,
    )
    job_id, served = await launcher.launch_model(request)
    await ctx.info(f"Job submitted — job_id={job_id}, served_model_name={served}")
    config = _load_config()
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
    system: Annotated[str | None, "Target HPC system name (required for FirecREST launchers)."] = None,
    partition: Annotated[str | None, "SLURM partition the job is running on."] = None,
) -> str:
    """Get the status of a running or queued SML job."""
    launcher = await _make_launcher(system=system, partition=partition)
    return (await launcher.get_job_status(job_id)).value


@mcp.tool
async def get_job_logs(
    job_id: Annotated[int, "SLURM job ID to retrieve logs for."],
    ctx: Context,
    system: Annotated[str | None, "Target HPC system name (required for FirecREST launchers)."] = None,
    partition: Annotated[str | None, "SLURM partition the job is running on."] = None,
) -> str:
    """Retrieve and stream stdout and stderr logs for a job."""
    launcher = await _make_launcher(system=system, partition=partition)
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
    system: Annotated[str | None, "Target HPC system name (required for FirecREST launchers)."] = None,
    partition: Annotated[str | None, "SLURM partition the job is running on."] = None,
) -> str:
    """Cancel a running or queued SML job."""
    launcher = await _make_launcher(system=system, partition=partition)
    await launcher.cancel_job(job_id)
    return f"Job {job_id} cancelled."
