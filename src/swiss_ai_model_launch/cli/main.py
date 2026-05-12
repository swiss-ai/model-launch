import argparse
import asyncio
import getpass
import grp
import importlib.metadata
import logging
import os
import re
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from pathlib import Path
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
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.utils import create_salt
from swiss_ai_model_launch.loadtest.cluster import ClusterLoadtestConfig, submit_cluster_loadtest
from swiss_ai_model_launch.loadtest.models import LoadtestConfig, ServerConfig, load_scenarios
from swiss_ai_model_launch.loadtest.setup import (
    DEFAULT_CLUSTER_CONTAINER_IMAGE,
    resolve_k6_script,
    resolve_prompts_file,
)
from swiss_ai_model_launch.mcp import mcp as _mcp

_OptionsFactory = Callable[[], Awaitable[OptionsDict]] | Callable[[GetValueFn], Awaitable[OptionsDict]] | None
_DEFAULT_LOADTEST_SERVER_URL = "https://api.swissai.svc.cscs.ch"
_DEFAULT_LOADTEST_READY_TIMEOUT_SECONDS = 1000000
_DEFAULT_LOADTEST_READY_POLL_SECONDS = 10
_LOADTEST_READY_PROGRESS_SECONDS = 300
_DEFAULT_LOADTEST_METRICS_REMOTE_WRITE_URL = "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"


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
    use_router_factory: _OptionsFactory = None,
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
                validator=lambda v: v.isdigit() and int(v) > 0,
                default="1",
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
                default="03:00:00",
            ),
        ],
    )


def _add_loadtest_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_cancel: bool = True,
    include_health_wait: bool = True,
) -> None:
    parser.add_argument(
        "--loadtest-k6-script",
        dest="loadtest_k6_script",
        default=None,
        help="k6 JavaScript file to stage into the cluster loadtest job (env: SML_LOADTEST_K6_SCRIPT).",
    )
    parser.add_argument(
        "--loadtest-metrics-remote-write-url",
        dest="loadtest_metrics_remote_write_url",
        default=os.environ.get("SML_LOADTEST_METRICS_REMOTE_WRITE_URL", _DEFAULT_LOADTEST_METRICS_REMOTE_WRITE_URL),
        help=(
            "Prometheus remote-write URL for k6 metrics "
            f"(default: {_DEFAULT_LOADTEST_METRICS_REMOTE_WRITE_URL}; env: SML_LOADTEST_METRICS_REMOTE_WRITE_URL)."
        ),
    )
    parser.add_argument(
        "--loadtest-metrics-remote-write",
        dest="loadtest_metrics_remote_write",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Prometheus remote-write output for k6 metrics (default: true).",
    )
    parser.add_argument(
        "--wait-for-loadtest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for the cluster loadtest job and download/copy the summary (default: true).",
    )
    parser.add_argument(
        "--loadtest-server-url",
        dest="loadtest_server_url",
        default=_DEFAULT_LOADTEST_SERVER_URL,
        help=f"OpenAI-compatible API base URL for k6 traffic (default: {_DEFAULT_LOADTEST_SERVER_URL}).",
    )
    parser.add_argument(
        "--loadtest-scenario",
        dest="loadtest_scenario",
        default="throughput",
        help="Built-in or custom loadtest scenario name (default: throughput).",
    )
    parser.add_argument(
        "--loadtest-max-tokens",
        dest="loadtest_max_tokens",
        default=None,
        help="Override max output tokens, or pass 'prompt' to use each prompt corpus entry's max_tokens.",
    )
    parser.add_argument(
        "--loadtest-ignore-eos",
        dest="loadtest_ignore_eos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether loadtest requests ignore EOS and generate until max tokens.",
    )
    parser.add_argument(
        "--loadtest-prompts-file",
        dest="loadtest_prompts_file",
        default=None,
        help=(
            "Cluster-visible prompt corpus JSON path. Defaults to SML_LOADTEST_PROMPTS_FILE, "
            "then the shared cluster path."
        ),
    )
    parser.add_argument(
        "--loadtest-prompt-seed",
        dest="loadtest_prompt_seed",
        type=int,
        default=1,
        help="Seed for deterministic prompt shuffling in k6 (default: 1).",
    )
    if include_health_wait:
        parser.add_argument(
            "--wait-until-healthy",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Wait for the target model to become healthy before running k6 (default: true).",
        )
        parser.add_argument(
            "--loadtest-ready-timeout",
            dest="loadtest_ready_timeout",
            type=int,
            default=_DEFAULT_LOADTEST_READY_TIMEOUT_SECONDS,
            help="Seconds to wait for the target model to become healthy.",
        )
    if include_cancel:
        parser.add_argument(
            "--cancel-after-loadtest",
            action="store_true",
            help="Cancel the launched SLURM job after the loadtest finishes.",
        )


def _add_advanced_launch_arguments(
    parser: argparse.ArgumentParser,
    *,
    tui_default: bool | None,
) -> None:
    parser.add_argument(
        "--serving-framework",
        dest="framework",
        required=True,
        help="Inference framework to use (e.g. sglang, vllm).",
    )
    parser.add_argument(
        "--slurm-environment",
        dest="slurm_environment",
        required=True,
        metavar="PATH",
        help="Local path to the environment .toml file.",
    )
    parser.add_argument(
        "--framework-args",
        dest="framework_args",
        default="",
        metavar="ARGS",
        help="Arguments forwarded to the inference framework.",
    )
    parser.add_argument(
        "--slurm-workers",
        dest="workers",
        type=int,
        default=1,
        help="Number of workers (default: 1).",
    )
    parser.add_argument(
        "--slurm-nodes-per-worker",
        dest="nodes_per_worker",
        type=int,
        default=1,
        help="Number of nodes per worker (default: 1).",
    )
    parser.add_argument(
        "--slurm-nodes",
        dest="nodes",
        type=int,
        default=None,
        help="Total number of nodes. Defaults to workers * nodes-per-worker.",
    )
    parser.add_argument(
        "--slurm-time",
        dest="time",
        default="00:05:00",
        metavar="HH:MM:SS",
        help="Job time limit (default: 00:05:00).",
    )
    parser.add_argument(
        "--slurm-reservation",
        dest="reservation",
        default=None,
        metavar="RESERVATION",
        help="SLURM reservation name (optional).",
    )
    parser.add_argument(
        "--served-model-name",
        dest="served_model_name",
        default=None,
        help="Name under which the model will be served. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--worker-port",
        dest="worker_port",
        type=int,
        default=5000,
        help="Port used by workers (default: 5000).",
    )
    parser.add_argument(
        "--use-router",
        dest="use_router",
        action="store_true",
        help="Enable router to load balance across workers.",
    )
    parser.add_argument(
        "--router-args",
        dest="router_args",
        default="",
        metavar="ARGS",
        help="Arguments forwarded to the router.",
    )
    parser.add_argument(
        "--disable-ocf",
        dest="disable_ocf",
        action="store_true",
        help="Disable OCF.",
    )
    parser.add_argument(
        "--disable-dcgm-exporter",
        dest="disable_dcgm_exporter",
        action="store_true",
        help="Disable the DCGM exporter.",
    )
    parser.add_argument(
        "--disable-metrics",
        dest="disable_metrics",
        action="store_true",
        help="Disable metrics collection.",
    )
    parser.add_argument(
        "--pre-launch-cmds",
        dest="pre_launch_cmds",
        default="",
        metavar="CMDS",
        help="Commands to run before launching the model.",
    )
    if tui_default is not None:
        parser.add_argument(
            "--tui",
            action=argparse.BooleanOptionalAction,
            default=tui_default,
            help="Launch the interactive TUI after submitting the job.",
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
    _add_advanced_launch_arguments(advanced_parser, tui_default=False)

    preconfigured_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch the interactive TUI after submitting the job.",
    )

    loadtest_parser = subparsers.add_parser("loadtest", help="Launch models and run loadtests")
    loadtest_subparsers = loadtest_parser.add_subparsers(dest="loadtest_command", required=True)

    loadtest_preconfigured_parser = loadtest_subparsers.add_parser(
        "preconfigured",
        help="Launch one preconfigured model, wait until healthy, then run a loadtest",
    )
    _make_firecrest_launcher_config().add_to_parser(loadtest_preconfigured_parser)
    _make_partition_config().add_to_parser(loadtest_preconfigured_parser)
    _make_reservation_config().add_to_parser(loadtest_preconfigured_parser)
    _make_launch_request_config().add_to_parser(loadtest_preconfigured_parser)
    _add_loadtest_arguments(loadtest_preconfigured_parser)

    loadtest_advanced_parser = loadtest_subparsers.add_parser(
        "advanced",
        help="Launch a model with the same arguments as `sml advanced`, then run a loadtest",
    )
    _make_firecrest_launcher_config().add_to_parser(loadtest_advanced_parser)
    _make_partition_config().add_to_parser(loadtest_advanced_parser)
    _add_advanced_launch_arguments(loadtest_advanced_parser, tui_default=None)
    _add_loadtest_arguments(loadtest_advanced_parser)

    loadtest_run_parser = loadtest_subparsers.add_parser(
        "run",
        help="Run a loadtest against an already launched model or external OpenAI-compatible URL",
    )
    _make_firecrest_launcher_config().add_to_parser(loadtest_run_parser)
    _make_partition_config().add_to_parser(loadtest_run_parser)
    _make_reservation_config().add_to_parser(loadtest_run_parser)
    loadtest_run_parser.add_argument(
        "--loadtest-model",
        dest="loadtest_model",
        default=None,
        help="Model name to health-check and send in OpenAI-compatible requests.",
    )
    _add_loadtest_arguments(loadtest_run_parser, include_cancel=False)

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
            min_token_validity=90,
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
    catalogue = await launcher.get_preconfigured_models()

    async def _get_vendor_models() -> dict[str, tuple[str, str]]:
        seen: dict[str, tuple[str, str]] = {}
        for entry in catalogue:
            if entry.model not in seen:
                seen[entry.model] = (entry.model, entry.model)
        return seen

    async def _get_frameworks(
        get_value_from_context: GetValueFn,
    ) -> dict[str, tuple[str, str]]:
        model = get_value_from_context("model")
        if model is None:
            return {}
        return {entry.framework: (entry.framework, entry.framework) for entry in catalogue if entry.model == model}

    launch_req_config = _make_launch_request_config(
        vendor_models_factory=_get_vendor_models,
        frameworks_factory=_get_frameworks,
        use_router_factory=lambda get_value: _get_router_options(get_value),
    )
    await launch_req_config.aconfigure(args=args)

    model = launch_req_config.get_non_none_value("model")
    framework = launch_req_config.get_non_none_value("framework")
    catalogue_entry: ModelCatalogEntry | None = next(
        (e for e in catalogue if e.model == model and e.framework == framework),
        None,
    )
    if catalogue_entry is None:
        catalogue_entry = ModelCatalogEntry(model=model, framework=framework)
    return LaunchRequest.from_catalog_entry(
        catalogue_entry,
        workers=int(launch_req_config.get_non_none_value("workers")),
        time=launch_req_config.get_non_none_value("time"),
        served_model_name=f"{model}-{create_salt(4)}",
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


def _make_advanced_launch_args(args: argparse.Namespace, launcher: Launcher, config: InitConfig) -> LaunchArgs:
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

    return LaunchArgs(
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
        disable_dcgm_exporter=args.disable_dcgm_exporter,
        disable_metrics=args.disable_metrics,
        telemetry_endpoint=config.get_value("telemetry_endpoint"),
    )


async def _run_advanced(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await _create_launcher(config, args, non_interactive=True)
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_args = _make_advanced_launch_args(args, launcher, config)

    launch_coro = launcher.launch_with_args(launch_args)
    if args.tui:
        await _run_monitor(launcher, launch_coro, cscs_api_key)
    else:
        job_id, served = await launch_coro
        print(f"Job submitted: {job_id}")
        print(f"Served model name: {served}")
        print(f"Logs: {launcher.get_log_dir(job_id)}")


def _loadtest_results_dir(run_id: int | str) -> Path:
    config_dir = Path(os.environ.get("SML_CONFIG_DIR", str(Path.home() / ".sml")))
    return config_dir / "loadtest" / str(run_id)


def _make_loadtest_config_from_values(
    *,
    scenario: str,
    max_tokens: str | None,
    ignore_eos: bool | None,
    prompt_seed: int = 1,
) -> LoadtestConfig:
    scenarios = {s.name: s for s in load_scenarios()}
    scenario_cfg = scenarios.get(scenario)
    if scenario_cfg is None:
        raise ValueError(f"Unknown loadtest scenario '{scenario}'. Available: {', '.join(sorted(scenarios))}")

    resolved_max_tokens = None if max_tokens == "prompt" else max_tokens or scenario_cfg.max_tokens
    return LoadtestConfig(
        scenario=scenario,
        think_time=scenario_cfg.think_time or "2",
        max_tokens=resolved_max_tokens,
        request_timeout=None,
        prompt_labels=scenario_cfg.prompt_labels,
        ignore_eos=ignore_eos,
        prompt_seed=prompt_seed,
    )


def _make_loadtest_config(args: argparse.Namespace) -> LoadtestConfig:
    return _make_loadtest_config_from_values(
        scenario=args.loadtest_scenario,
        max_tokens=args.loadtest_max_tokens,
        ignore_eos=args.loadtest_ignore_eos,
        prompt_seed=args.loadtest_prompt_seed,
    )


def _make_loadtest_server(args: argparse.Namespace, api_key: str, model_name: str) -> ServerConfig:
    return ServerConfig(
        url=args.loadtest_server_url.rstrip("/"),
        api_key=api_key,
        model=model_name,
        is_swissai=True,
    )


def _make_cluster_loadtest_config(
    args: argparse.Namespace,
    *,
    reservation: str | None = None,
) -> ClusterLoadtestConfig:
    if getattr(args, "cancel_after_loadtest", False) and not args.wait_for_loadtest:
        raise ValueError("--cancel-after-loadtest requires --wait-for-loadtest.")
    if args.loadtest_ready_timeout <= 0:
        raise ValueError("--loadtest-ready-timeout must be greater than 0.")
    return ClusterLoadtestConfig(
        container_image=str(DEFAULT_CLUSTER_CONTAINER_IMAGE),
        wait=args.wait_for_loadtest,
        reservation=reservation or getattr(args, "reservation", None),
        metrics_remote_write_url=(
            args.loadtest_metrics_remote_write_url if args.loadtest_metrics_remote_write else None
        ),
    )


async def _run_k6_on_cluster(
    *,
    launcher: Launcher,
    server: ServerConfig,
    loadtest_config: LoadtestConfig,
    summary_path: Path,
    args: argparse.Namespace,
    k6_script: str | Path | None = None,
    reservation: str | None = None,
) -> None:
    k6_script_path = resolve_k6_script(k6_script or args.loadtest_k6_script)
    if not k6_script_path.exists():
        raise FileNotFoundError(f"k6 script not found: {k6_script_path}")

    prompts_file = resolve_prompts_file(args.loadtest_prompts_file)

    cluster_config = _make_cluster_loadtest_config(args, reservation=reservation)
    print(f"Loadtest container image: {cluster_config.container_image}")
    print(f"Loadtest prompts file: {prompts_file}")
    if cluster_config.metrics_remote_write_url:
        print(f"Loadtest metrics remote write: {cluster_config.metrics_remote_write_url}")
    job_id = await submit_cluster_loadtest(
        launcher=launcher,
        server=server,
        bench=loadtest_config,
        k6_script=k6_script_path,
        prompts_file=prompts_file,
        summary_path=summary_path,
        cluster=cluster_config,
    )
    print(f"Cluster loadtest job submitted: {job_id}")
    if cluster_config.wait:
        print(f"Loadtest summary: {summary_path}")


async def _wait_until_model_healthy(
    served_model_name: str,
    api_key: str,
    *,
    server_url: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    last_reported_health: ModelHealth | None = None
    next_progress_at = started + _LOADTEST_READY_PROGRESS_SECONDS
    print(f"Waiting for model health: {served_model_name} via {server_url.rstrip('/')}")

    while True:
        health = await check_model_health(served_model_name, api_key, base_url=server_url)
        if health == ModelHealth.HEALTHY:
            print(f"Model is healthy: {served_model_name}")
            return

        elapsed = loop.time() - started
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for model to become healthy: {served_model_name}"
            )

        now = loop.time()
        if health != last_reported_health:
            print(f"Model health is {health.value}; waiting quietly...")
            last_reported_health = health
        elif now >= next_progress_at:
            elapsed_seconds = round(elapsed)
            print(f"Still waiting for model health after {elapsed_seconds}s; current status: {health.value}")
            next_progress_at = now + _LOADTEST_READY_PROGRESS_SECONDS

        await asyncio.sleep(poll_interval_seconds)


async def _run_loadtest_for_submitted_job(
    *,
    launcher: Launcher,
    job_id: int,
    served_model_name: str,
    cscs_api_key: str,
    args: argparse.Namespace,
    loadtest_config: LoadtestConfig,
    k6_script: str | Path | None = None,
    loadtest_reservation: str | None = None,
) -> None:
    try:
        if args.wait_until_healthy:
            await _wait_until_model_healthy(
                served_model_name,
                cscs_api_key,
                server_url=args.loadtest_server_url,
                timeout_seconds=args.loadtest_ready_timeout,
                poll_interval_seconds=_DEFAULT_LOADTEST_READY_POLL_SECONDS,
            )

        results_dir = _loadtest_results_dir(job_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = results_dir / f"summary_{loadtest_config.scenario}_{timestamp}.json"
        server = _make_loadtest_server(args, cscs_api_key, served_model_name)
        await _run_k6_on_cluster(
            launcher=launcher,
            server=server,
            loadtest_config=loadtest_config,
            summary_path=summary_path,
            args=args,
            k6_script=k6_script,
            reservation=loadtest_reservation,
        )
    finally:
        if args.cancel_after_loadtest:
            print(f"Cancelling job after loadtest: {job_id}")
            await launcher.cancel_job(job_id)


async def _submit_and_run_loadtest(
    *,
    launcher: Launcher,
    launch_coro: Coroutine[Any, Any, tuple[int, str]],
    cscs_api_key: str,
    args: argparse.Namespace,
    loadtest_config: LoadtestConfig,
    k6_script: str | Path | None = None,
    loadtest_reservation: str | None = None,
) -> None:
    _make_cluster_loadtest_config(args, reservation=loadtest_reservation)
    job_id, served = await launch_coro
    print(f"Job submitted: {job_id}")
    print(f"Served model name: {served}")
    print(f"Logs: {launcher.get_log_dir(job_id)}")
    await _run_loadtest_for_submitted_job(
        launcher=launcher,
        job_id=job_id,
        served_model_name=served,
        cscs_api_key=cscs_api_key,
        args=args,
        loadtest_config=loadtest_config,
        k6_script=k6_script,
        loadtest_reservation=loadtest_reservation,
    )


async def _run_loadtest_against_existing_model(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        raise ValueError("Run `sml init` first so SML can submit the cluster loadtest job.")

    config = InitConfig.load()
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    loadtest_model = args.loadtest_model
    request_model = loadtest_model or ""
    loadtest_config = _make_loadtest_config(args)
    launcher = await _create_launcher(config, args, non_interactive=True)

    if args.wait_until_healthy and loadtest_model:
        await _wait_until_model_healthy(
            loadtest_model,
            cscs_api_key,
            server_url=args.loadtest_server_url,
            timeout_seconds=args.loadtest_ready_timeout,
            poll_interval_seconds=_DEFAULT_LOADTEST_READY_POLL_SECONDS,
        )
    elif args.wait_until_healthy:
        print("Skipping model health check because --loadtest-model was not provided.")

    run_id = loadtest_model.replace("/", "_") if loadtest_model else "external"
    results_dir = _loadtest_results_dir(run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"summary_{loadtest_config.scenario}_{timestamp}.json"
    server = _make_loadtest_server(args, cscs_api_key, request_model)
    await _run_k6_on_cluster(
        launcher=launcher,
        server=server,
        loadtest_config=loadtest_config,
        summary_path=summary_path,
        args=args,
    )


async def _run_loadtest_preconfigured(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await _create_launcher(config, args)
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_request = await _get_launch_request(launcher, args)
    loadtest_config = _make_loadtest_config(args)

    await _submit_and_run_loadtest(
        launcher=launcher,
        launch_coro=launcher.launch_model(launch_request),
        cscs_api_key=cscs_api_key,
        args=args,
        loadtest_config=loadtest_config,
    )


async def _run_loadtest_advanced(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await _create_launcher(config, args, non_interactive=True)
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_args = _make_advanced_launch_args(args, launcher, config)
    loadtest_config = _make_loadtest_config(args)

    await _submit_and_run_loadtest(
        launcher=launcher,
        launch_coro=launcher.launch_with_args(launch_args),
        cscs_api_key=cscs_api_key,
        args=args,
        loadtest_config=loadtest_config,
    )


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
    elif subcommand == "loadtest":
        if args.loadtest_command == "preconfigured":
            await _run_loadtest_preconfigured(args)
        elif args.loadtest_command == "advanced":
            await _run_loadtest_advanced(args)
        elif args.loadtest_command == "run":
            await _run_loadtest_against_existing_model(args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.subcommand is None:
        default = "preconfigured" if InitConfig.exists() else "init"
        args = parser.parse_args([default])
    if args.subcommand == "mcp":
        _run_mcp()
    else:
        asyncio.run(_main(args))


if __name__ == "__main__":
    main()
