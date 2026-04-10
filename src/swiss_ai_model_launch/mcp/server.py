"""FastMCP server exposing SML capabilities as MCP tools."""

import getpass
import grp
import os
from typing import TYPE_CHECKING, Annotated, Any, Literal

import fastmcp
import firecrest as f7t

from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher, SlurmLauncher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.utils import create_salt

if TYPE_CHECKING:
    from swiss_ai_model_launch.cli.configuration import InitConfig

mcp = fastmcp.FastMCP(
    name="sml",
    instructions=(
        "Swiss AI Model Launcher MCP server. "
        "Use these tools to list, launch, monitor, and cancel AI model jobs "
        "on HPC clusters via SML."
    ),
)


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
    system: str | None,
    partition: str,
    reservation: str | None = None,
) -> Launcher:
    config = _load_config()
    launcher_type = config.get_non_none_value("launcher")
    telemetry_endpoint = config.get_value("telemetry_endpoint")

    if launcher_type == "firecrest":
        if system is None:
            raise ValueError(
                "`system` is required when using a FirecREST-based launcher."
            )
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


async def list_preconfigured_models(
    system: Annotated[
        str | None,
        "Target HPC system name (required for FirecREST launchers, e.g. 'daint').",
    ] = None,
    partition: Annotated[
        str,
        "SLURM partition to query (e.g. 'normal', 'gpu').",
    ] = "normal",
) -> list[dict[str, Any]]:
    """List all preconfigured models available for launch."""
    launcher = await _make_launcher(system=system, partition=partition)
    models = await launcher.get_preconfigured_models()
    return [m.model_dump() for m in models]


async def launch_model(
    partition: Annotated[str, "SLURM partition to run the job on (e.g. 'normal')."],
    vendor: Annotated[str, "Model vendor (e.g. 'swiss-ai', 'Qwen')."],
    model_name: Annotated[str, "Model name (e.g. 'Apertus-70B')."],
    framework: Annotated[Literal["sglang", "vllm"], "Inference framework."],
    workers: Annotated[int, "Number of workers."],
    time: Annotated[str, "Job time limit in HH:MM:SS format."],
    system: Annotated[
        str | None,
        "Target HPC system name (required for FirecREST launchers).",
    ] = None,
    use_router: Annotated[
        bool, "Enable router for load balancing across workers."
    ] = False,
    reservation: Annotated[str | None, "SLURM reservation name (optional)."] = None,
    nodes_per_worker: Annotated[int, "Number of nodes per worker."] = 1,
    framework_args: Annotated[
        str | None, "Extra arguments forwarded to the framework."
    ] = None,
    pre_launch_cmds: Annotated[
        str | None, "Shell commands to run before launch."
    ] = None,
) -> dict[str, Any]:
    """Launch a model on an HPC cluster. Returns the job ID and served model name."""
    launcher = await _make_launcher(
        system=system, partition=partition, reservation=reservation
    )
    served_model_name = f"{vendor}/{model_name}-{create_salt(4)}"
    request = LaunchRequest(
        vendor=vendor,
        model_name=model_name,
        framework=framework,
        workers=workers,
        nodes_per_worker=nodes_per_worker,
        time=time,
        served_model_name=served_model_name,
        framework_args=framework_args,
        pre_launch_cmds=pre_launch_cmds,
        use_router=use_router,
    )
    job_id, served = await launcher.launch_model(request)
    return {"job_id": job_id, "served_model_name": served}


async def get_job_status(
    job_id: Annotated[int, "SLURM job ID to query."],
    partition: Annotated[str, "SLURM partition the job is running on."],
    system: Annotated[
        str | None,
        "Target HPC system name (required for FirecREST launchers).",
    ] = None,
) -> str:
    """Get the status of a running or queued SML job."""
    launcher = await _make_launcher(system=system, partition=partition)
    status = await launcher.get_job_status(job_id)
    return status.value


async def get_job_logs(
    job_id: Annotated[int, "SLURM job ID to retrieve logs for."],
    partition: Annotated[str, "SLURM partition the job is running on."],
    system: Annotated[
        str | None,
        "Target HPC system name (required for FirecREST launchers).",
    ] = None,
) -> dict[str, str]:
    """Retrieve stdout and stderr logs for a job."""
    launcher = await _make_launcher(system=system, partition=partition)
    stdout, stderr = await launcher.get_job_logs(job_id)
    return {"stdout": stdout, "stderr": stderr}


async def cancel_job(
    job_id: Annotated[int, "SLURM job ID to cancel."],
    partition: Annotated[str, "SLURM partition the job is running on."],
    system: Annotated[
        str | None,
        "Target HPC system name (required for FirecREST launchers).",
    ] = None,
) -> str:
    """Cancel a running or queued SML job."""
    launcher = await _make_launcher(system=system, partition=partition)
    await launcher.cancel_job(job_id)
    return f"Job {job_id} cancelled."


mcp.add_tool(list_preconfigured_models)
mcp.add_tool(launch_model)
mcp.add_tool(get_job_status)
mcp.add_tool(get_job_logs)
mcp.add_tool(cancel_job)
