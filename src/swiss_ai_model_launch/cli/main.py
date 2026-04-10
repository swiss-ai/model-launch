import argparse
import asyncio
import getpass
import grp
import importlib.metadata
import logging
import os
import re
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, cast

import firecrest as f7t

from swiss_ai_model_launch.cli.configuration import InitConfig
from swiss_ai_model_launch.cli.configuration.models import (
    ChainConfiguration,
    GetValueFn,
    OptionsConfiguration,
    OptionsDict,
    TextConfiguration,
)
from swiss_ai_model_launch.cli.display import DisplayState, LiveDisplay
from swiss_ai_model_launch.cli.healthcheck import check_model_health
from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth
from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher, SlurmLauncher
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.utils import create_salt
from swiss_ai_model_launch.mcp import mcp as _mcp

_OptionsFactory = Callable[[], Awaitable[OptionsDict]] | Callable[[GetValueFn], Awaitable[OptionsDict]] | None
_DefaultFactory = Callable[[], Awaitable[str | None]] | Callable[[GetValueFn], Awaitable[str | None]] | None


def _make_firecrest_launcher_config(
    systems_factory: _OptionsFactory = None,
) -> ChainConfiguration:
    _empty: OptionsDict = {}
    return ChainConfiguration(
        name="firecrest_launcher_configuration",
        chain=[
            OptionsConfiguration(
                name="firecrest_system",
                prompt="Choose the target system to launch the model on.",
                options_factory=systems_factory,
                options=None if systems_factory else _empty,
                env_var="SML_FIRECREST_SYSTEM",
            ),
        ],
    )


def _make_partition_config(
    partitions_factory: _OptionsFactory = None,
) -> ChainConfiguration:
    _empty: OptionsDict = {}
    return ChainConfiguration(
        name="partition_configuration",
        chain=[
            OptionsConfiguration(
                name="partition",
                prompt="Choose the partition to launch the model on.",
                options_factory=partitions_factory,
                options=None if partitions_factory else _empty,
                env_var="SML_PARTITION",
            ),
        ],
    )


def _make_reservation_config() -> ChainConfiguration:
    return ChainConfiguration(
        name="reservation_configuration",
        chain=[
            TextConfiguration(
                name="reservation",
                prompt="SLURM reservation name (optional, leave blank to skip).",
                env_var="SML_RESERVATION",
            ),
        ],
    )


def _make_launch_request_config(
    vendor_models_factory: _OptionsFactory = None,
    frameworks_factory: _OptionsFactory = None,
    workers_default_factory: _DefaultFactory = None,
    use_router_factory: _OptionsFactory = None,
    time_default_factory: _DefaultFactory = None,
) -> ChainConfiguration:
    """Build the launch request config.

    Pass factories for interactive/runtime use; omit them to get a static shell
    suitable only for parser registration.
    """
    _empty: OptionsDict = {}
    _router_options: OptionsDict = {
        "yes": ("Yes", "Use router to load balance across workers"),
        "no": ("No", "Do not use router"),
    }
    return ChainConfiguration(
        name="launcher_request_configuration",
        chain=[
            OptionsConfiguration(
                name="model",
                prompt="Choose the model to launch.",
                options_factory=vendor_models_factory,
                options=None if vendor_models_factory else _empty,
            ),
            OptionsConfiguration(
                name="framework",
                prompt="Choose the framework to run the model with.",
                options_factory=frameworks_factory,
                options=None if frameworks_factory else _empty,
            ),
            TextConfiguration(
                name="workers",
                prompt="Number of workers to use for running the model.",
                validator=((lambda v: v.isdigit() and int(v) > 0) if workers_default_factory else None),
                default_factory=workers_default_factory,
            ),
            OptionsConfiguration(
                name="use_router",
                prompt="Use router to load balance across workers.",
                options_factory=use_router_factory,
                options=None if use_router_factory else _router_options,
            ),
            TextConfiguration(
                name="time",
                prompt="Time duration for running the model (in format HH:MM:SS).",
                validator=lambda v: bool(re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", v)),
                default_factory=time_default_factory,
            ),
        ],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sml",
        description="Swiss AI Model Launcher",
    )
    _meta = importlib.metadata.metadata("swiss-ai-model-launch")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"sml {_meta['Version']}",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=False)

    init_parser = subparsers.add_parser("init", help="Initialize SML configuration")
    InitConfig().add_to_parser(init_parser)

    preconfigured_parser = subparsers.add_parser("preconfigured", help="Launch a model with guided prompts")
    _make_firecrest_launcher_config().add_to_parser(preconfigured_parser)
    _make_partition_config().add_to_parser(preconfigured_parser)
    _make_reservation_config().add_to_parser(preconfigured_parser)
    _make_launch_request_config().add_to_parser(preconfigured_parser)

    advanced_parser = subparsers.add_parser("advanced", help="Launch a model with advanced configuration")
    _make_firecrest_launcher_config().add_to_parser(advanced_parser)
    _make_partition_config().add_to_parser(advanced_parser)
    advanced_parser.add_argument(
        "--serving-framework",
        dest="framework",
        required=True,
        help="Inference framework to use (e.g. sglang, vllm).",
    )
    advanced_parser.add_argument(
        "--slurm-environment",
        dest="slurm_environment",
        required=True,
        metavar="PATH",
        help="Local path to the environment .toml file.",
    )
    advanced_parser.add_argument(
        "--framework-args",
        dest="framework_args",
        default="",
        metavar="ARGS",
        help="Arguments forwarded to the inference framework.",
    )
    advanced_parser.add_argument(
        "--slurm-workers",
        dest="workers",
        type=int,
        default=1,
        help="Number of workers (default: 1).",
    )
    advanced_parser.add_argument(
        "--slurm-nodes-per-worker",
        dest="nodes_per_worker",
        type=int,
        default=1,
        help="Number of nodes per worker (default: 1).",
    )
    advanced_parser.add_argument(
        "--slurm-nodes",
        dest="nodes",
        type=int,
        default=None,
        help="Total number of nodes. Defaults to workers * nodes-per-worker.",
    )
    advanced_parser.add_argument(
        "--slurm-time",
        dest="time",
        default="00:05:00",
        metavar="HH:MM:SS",
        help="Job time limit (default: 00:05:00).",
    )
    advanced_parser.add_argument(
        "--slurm-reservation",
        dest="reservation",
        default=None,
        metavar="RESERVATION",
        help="SLURM reservation name (optional).",
    )
    advanced_parser.add_argument(
        "--served-model-name",
        dest="served_model_name",
        default=None,
        help="Name under which the model will be served. Auto-generated if omitted.",
    )
    advanced_parser.add_argument(
        "--worker-port",
        dest="worker_port",
        type=int,
        default=5000,
        help="Port used by workers (default: 5000).",
    )
    advanced_parser.add_argument(
        "--use-router",
        dest="use_router",
        action="store_true",
        help="Enable router to load balance across workers.",
    )
    advanced_parser.add_argument(
        "--router-args",
        dest="router_args",
        default="",
        metavar="ARGS",
        help="Arguments forwarded to the router.",
    )
    advanced_parser.add_argument(
        "--disable-ocf",
        dest="disable_ocf",
        action="store_true",
        help="Disable OCF.",
    )
    advanced_parser.add_argument(
        "--pre-launch-cmds",
        dest="pre_launch_cmds",
        default="",
        metavar="CMDS",
        help="Commands to run before launching the model.",
    )
    advanced_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch the interactive TUI after submitting the job.",
    )

    preconfigured_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch the interactive TUI after submitting the job.",
    )

    subparsers.add_parser("mcp", help="Start the SML MCP server")

    return parser


async def _run_initial_configuration_wizard(args: argparse.Namespace) -> None:
    config = InitConfig()
    await config.aconfigure(args=args)
    config.save()
    print("SML is configured and ready to use! Please restart the program.")


def _get_firecrest_client_from_init_config(config: InitConfig) -> f7t.v2.AsyncFirecrest:
    return f7t.v2.AsyncFirecrest(
        firecrest_url=config.get_non_none_value("firecrest_url"),
        authorization=f7t.ClientCredentialsAuth(
            client_id=config.get_non_none_value("firecrest_client_id"),
            client_secret=config.get_non_none_value("firecrest_client_secret"),
            token_uri=config.get_non_none_value("firecrest_token_uri"),
        ),
    )


async def _get_firecrest_launcher_with_client(
    client: f7t.v2.AsyncFirecrest,
    telemetry_endpoint: str | None = None,
    args: argparse.Namespace | None = None,
    non_interactive: bool = False,
) -> FirecRESTLauncher:
    async def _get_systems() -> dict[str, tuple[str, str]]:
        return {sys["name"]: (sys["name"], sys["ssh"]["host"]) for sys in await client.systems()}

    firecrest_config = _make_firecrest_launcher_config(systems_factory=_get_systems)
    await firecrest_config.aconfigure(args=args, non_interactive=non_interactive)
    system_name = firecrest_config.get_non_none_value("firecrest_system")

    async def _get_partitions() -> dict[str, tuple[str, str]]:
        return {part["name"]: (part["name"], part["name"]) for part in await client.partitions(system_name)}

    partition_config = _make_partition_config(partitions_factory=_get_partitions)
    await partition_config.aconfigure(args=args, non_interactive=non_interactive)

    if non_interactive:
        reservation = getattr(args, "reservation", None) if args else None
    else:
        reservation_config = _make_reservation_config()
        await reservation_config.aconfigure(args=args)
        reservation = reservation_config.get_value("reservation") or None

    return await FirecRESTLauncher.from_client(
        client=client,
        system_name=system_name,
        partition=partition_config.get_non_none_value("partition"),
        reservation=reservation,
        telemetry_endpoint=telemetry_endpoint,
    )


async def _get_slurm_launcher(
    telemetry_endpoint: str | None = None,
    args: argparse.Namespace | None = None,
    non_interactive: bool = False,
) -> SlurmLauncher:
    async def _get_partitions() -> dict[str, tuple[str, str]]:
        proc = await asyncio.create_subprocess_exec(
            "sinfo",
            "-h",
            "-o",
            "%P",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        partitions = [p.rstrip("*") for p in stdout.decode().split() if p.strip()]
        return {p: (p, p) for p in partitions}

    partition_config = _make_partition_config(partitions_factory=_get_partitions)
    await partition_config.aconfigure(args=args, non_interactive=non_interactive)

    if non_interactive:
        reservation = getattr(args, "reservation", None) if args else None
    else:
        reservation_config = _make_reservation_config()
        await reservation_config.aconfigure(args=args)
        reservation = reservation_config.get_value("reservation") or None

    return SlurmLauncher(
        system_name="local",
        username=getpass.getuser(),
        account=grp.getgrgid(os.getgid()).gr_name,
        partition=partition_config.get_non_none_value("partition"),
        reservation=reservation,
        telemetry_endpoint=telemetry_endpoint,
    )


def _split_vendor_model(combined: str) -> tuple[str, str]:
    vendor, model_name = combined.split("/", 1)
    return vendor, model_name


async def _get_preconfigured_default(
    get_value_from_context: GetValueFn, preconfigured: list[LaunchRequest], field: str
) -> str | None:
    combined = get_value_from_context("model")
    if combined is None:
        return None
    vendor, model_name = _split_vendor_model(combined)
    framework = get_value_from_context("framework")
    match = next(
        (
            lr
            for lr in preconfigured
            if lr.vendor == vendor and lr.model_name == model_name and lr.framework == framework
        ),
        None,
    )
    if match is None:
        return None
    return str(getattr(match, field))


async def _get_router_options(get_value: GetValueFn) -> dict[str, tuple[str, str]]:
    workers = get_value("workers")
    if workers is not None and int(workers) > 1:
        return {
            "yes": ("Yes", "Use router to load balance across workers"),
            "no": ("No", "Do not use router"),
        }
    return {
        "no": ("No", "Do not use router"),
    }


async def _get_launch_request(launcher: Launcher, args: argparse.Namespace | None = None) -> LaunchRequest:
    preconfigured_launch_requests = await launcher.get_preconfigured_models()

    async def _get_vendor_models() -> dict[str, tuple[str, str]]:
        seen: dict[str, tuple[str, str]] = {}
        for lr in preconfigured_launch_requests:
            key = f"{lr.vendor}/{lr.model_name}"
            if key not in seen:
                seen[key] = (lr.model_name, lr.vendor)
        return seen

    async def _get_frameworks(
        get_value_from_context: GetValueFn,
    ) -> dict[str, tuple[str, str]]:
        combined = get_value_from_context("model")
        if combined is None:
            return {}
        vendor, model_name = _split_vendor_model(combined)
        return {
            lr.framework: (lr.framework, lr.framework)
            for lr in preconfigured_launch_requests
            if lr.model_name == model_name and lr.vendor == vendor
        }

    launch_req_config = _make_launch_request_config(
        vendor_models_factory=_get_vendor_models,
        frameworks_factory=_get_frameworks,
        workers_default_factory=lambda get_value: _get_preconfigured_default(
            get_value, preconfigured_launch_requests, "workers"
        ),
        use_router_factory=lambda get_value: _get_router_options(get_value),
        time_default_factory=lambda get_value: _get_preconfigured_default(
            get_value, preconfigured_launch_requests, "time"
        ),
    )
    await launch_req_config.aconfigure(args=args)

    vendor, model_name = _split_vendor_model(launch_req_config.get_non_none_value("model"))
    framework = launch_req_config.get_non_none_value("framework")
    preconfigured = next(
        (
            lr
            for lr in preconfigured_launch_requests
            if lr.vendor == vendor and lr.model_name == model_name and lr.framework == framework
        ),
        None,
    )
    return LaunchRequest(
        vendor=vendor,
        model_name=model_name,
        framework=framework,
        environment=preconfigured.environment if preconfigured else None,
        workers=int(launch_req_config.get_non_none_value("workers")),
        nodes_per_worker=preconfigured.nodes_per_worker if preconfigured else 1,
        time=launch_req_config.get_non_none_value("time"),
        served_model_name=f"{vendor}/{model_name}-{create_salt(4)}",
        framework_args=preconfigured.framework_args if preconfigured else None,
        pre_launch_cmds=preconfigured.pre_launch_cmds if preconfigured else None,
        use_router=launch_req_config.get_non_none_value("use_router") == "yes",
    )


_logger = logging.getLogger(__name__)


async def _create_launcher(
    config: InitConfig,
    args: argparse.Namespace,
    non_interactive: bool = False,
) -> Launcher:
    launcher_type = config.get_non_none_value("launcher")
    telemetry_endpoint = config.get_value("telemetry_endpoint")

    if launcher_type == "slurm" and getattr(args, "firecrest_system", None):
        _logger.warning("--firecrest-system is ignored when using the SLURM launcher")

    if launcher_type == "firecrest":
        firecrest_client = _get_firecrest_client_from_init_config(config)
        return cast(
            Launcher,
            await _get_firecrest_launcher_with_client(
                firecrest_client,
                telemetry_endpoint=telemetry_endpoint,
                args=args,
                non_interactive=non_interactive,
            ),
        )
    elif launcher_type == "slurm":
        return cast(
            Launcher,
            await _get_slurm_launcher(
                telemetry_endpoint=telemetry_endpoint,
                args=args,
                non_interactive=non_interactive,
            ),
        )
    else:
        raise NotImplementedError(f"Launcher {launcher_type} is not supported yet.")


async def _run_monitor(
    launcher: Launcher,
    launch_coro: Coroutine[Any, Any, tuple[int, str]],
    cscs_api_key: str,
) -> None:
    state = DisplayState()
    state.update(cluster=launcher.system_name, partition=launcher.partition)

    async def _monitor() -> None:
        job_id, served = await launch_coro
        state.update(
            job_id=job_id,
            served_model_name=served,
            model_health=ModelHealth.NOT_DEPLOYED,
        )
        ever_healthy = False
        while True:
            await asyncio.sleep(5)

            job_status = await launcher.get_job_status(job_id)
            state.update(job_status=job_status)

            model_health = await check_model_health(served, cscs_api_key)
            if model_health == ModelHealth.NOT_RESPONDING and not ever_healthy:
                model_health = ModelHealth.NOT_DEPLOYED
            ever_healthy = ever_healthy or model_health == ModelHealth.HEALTHY
            state.update(model_health=model_health)

            o, e = await launcher.get_job_logs(job_id)
            state.set_out_log(o)
            state.set_err_log(e)

    kill_job = await LiveDisplay(state).run(_monitor())
    if kill_job and state.job_id is not None:
        await launcher.cancel_job(state.job_id)


async def _run_preconfigured(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await _create_launcher(config, args)
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_request = await _get_launch_request(launcher, args)
    launch_coro = launcher.launch_model(launch_request)
    if args.tui:
        await _run_monitor(launcher, launch_coro, cscs_api_key)
    else:
        job_id, served = await launch_coro
        print(f"Job submitted: {job_id}")
        print(f"Served model name: {served}")
        print(f"Logs: {launcher.get_log_dir(job_id)}")


async def _run_advanced(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await _create_launcher(config, args, non_interactive=True)
    cscs_api_key = config.get_non_none_value("cscs_api_key")

    if args.served_model_name:
        served_model_name = args.served_model_name
    else:
        match = re.search(r"--served-model-name\s+(\S+)", args.framework_args or "")
        if not match:
            raise ValueError(
                "--served-model-name must be provided either as a direct argument "
                "or via --served-model-name inside --framework-args"
            )
        served_model_name = match.group(1)
    job_name = f"sml_{served_model_name.replace('/', '_')}_{create_salt(8)}"

    launch_args = LaunchArgs(
        job_name=job_name,
        served_model_name=served_model_name,
        account=launcher.account,
        partition=launcher.partition,
        workers=args.workers,
        nodes_per_worker=args.nodes_per_worker,
        nodes=args.nodes,
        time=args.time,
        reservation=args.reservation or None,
        environment=args.slurm_environment,
        framework=args.framework,
        framework_args=args.framework_args,
        pre_launch_cmds=args.pre_launch_cmds,
        worker_port=args.worker_port,
        use_router=args.use_router,
        router_args=args.router_args,
        disable_ocf=args.disable_ocf,
        telemetry_endpoint=config.get_value("telemetry_endpoint"),
    )

    launch_coro = launcher.launch_with_args(launch_args)
    if args.tui:
        await _run_monitor(launcher, launch_coro, cscs_api_key)
    else:
        job_id, served = await launch_coro
        print(f"Job submitted: {job_id}")
        print(f"Served model name: {served}")
        print(f"Logs: {launcher.get_log_dir(job_id)}")


def _run_mcp() -> None:
    _mcp.run()


async def _main(args: argparse.Namespace) -> None:
    subcommand = args.subcommand
    if subcommand == "init":
        await _run_initial_configuration_wizard(args)
    elif subcommand == "preconfigured":
        await _run_preconfigured(args)
    elif subcommand == "advanced":
        await _run_advanced(args)
    elif subcommand == "mcp":
        _run_mcp()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.subcommand is None:
        default = "preconfigured" if InitConfig.exists() else "init"
        args = parser.parse_args([default])
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
