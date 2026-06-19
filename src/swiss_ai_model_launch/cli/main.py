import argparse
import asyncio
import getpass
import grp
import importlib.metadata
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable, Coroutine
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
from swiss_ai_model_launch.cli.display import ChainJobView, DisplayState, LiveDisplay
from swiss_ai_model_launch.cli.healthcheck import ReplicaHealthReport, check_model_health
from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth
from swiss_ai_model_launch.cli.loadtest import add_loadtest_parser, run_loadtest_command
from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher, SlurmLauncher
from swiss_ai_model_launch.launchers.framework import OCF_BOOTSTRAP_ADDR_DEV, render_master, render_rank_scripts
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import (
    DEFAULT_MAX_JOB_TIME,
    ROUTER_OCF,
    ROUTER_SGL,
    TELEMETRY_ENDPOINT,
    LaunchArgs,
    RouterMode,
    time_str_to_seconds,
)
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import ScheduledJob
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.topology import Topology
from swiss_ai_model_launch.launchers.utils import create_salt, render_sbatch_header
from swiss_ai_model_launch.mcp import mcp as _mcp

_OptionsFactory = Callable[[], Awaitable[OptionsDict]] | Callable[[GetValueFn], Awaitable[OptionsDict]] | None


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


def _make_slurm_account_config() -> ChainConfiguration:
    return ChainConfiguration(
        name="slurm_account_configuration",
        chain=[
            TextConfiguration(
                name="slurm_account",
                prompt="SLURM account (optional, leave blank to use your default group).",
                env_var="SML_ACCOUNT",
            ),
        ],
    )


def _make_launch_request_config(
    vendor_models_factory: _OptionsFactory = None,
    frameworks_factory: _OptionsFactory = None,
    router_factory: _OptionsFactory = None,
) -> ChainConfiguration:
    """Build the launch request config.

    Pass factories for interactive/runtime use; omit them to get a static shell
    suitable only for parser registration.
    """
    _empty: OptionsDict = {}
    _router_options: OptionsDict = {
        ROUTER_OCF: ("OCF", "OpenTela load-balances across the replica peers on the mesh"),
        ROUTER_SGL: ("SGL", "In-job SGLang router fronts the replicas (needs replicas > 1)"),
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
                name="replicas",
                prompt="Number of replicas (independent inference engine instances) to launch.",
                validator=lambda v: v.isdigit() and int(v) > 0,
                default="1",
            ),
            OptionsConfiguration(
                name="router",
                prompt="Routing strategy across replicas (OCF = OpenTela mesh, SGL = in-job SGLang router).",
                options_factory=router_factory,
                options=None if router_factory else _router_options,
            ),
            TextConfiguration(
                name="time",
                prompt="Time duration for running the model (in format HH:MM:SS).",
                validator=lambda v: bool(re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", v)),
                default="03:00:00",
            ),
        ],
    )


def _add_advanced_launch_arguments(
    advanced_parser: argparse.ArgumentParser,
    *,
    tui_default: bool | None,
) -> None:
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
        "--slurm-replicas",
        dest="replicas",
        type=int,
        default=1,
        help="Number of independent inference engine instances (default: 1).",
    )
    advanced_parser.add_argument(
        "--slurm-nodes-per-replica",
        dest="nodes_per_replica",
        type=int,
        default=1,
        help=(
            "Number of nodes spanned by one replica (default: 1). "
            "Set this to match your TP/PP/DP/EP layout in --framework-args."
        ),
    )
    advanced_parser.add_argument(
        "--time",
        dest="time",
        default="02:00:00",
        metavar="HH:MM:SS",
        help=(
            "Total time the model should stay up (default: 02:00:00). If it "
            "exceeds the per-job cap (--max-job-time), pass --consecutive to "
            "serve it with a chain of jobs."
        ),
    )
    advanced_parser.add_argument(
        "--consecutive",
        dest="consecutive",
        action="store_true",
        help=(
            "Required when --time exceeds the per-job cap. Pre-schedules a chain "
            "of jobs (each up to --max-job-time) at absolute begin times so the "
            "model stays up for the full --time; each job cancels its predecessor "
            "once its replicas are healthy."
        ),
    )
    advanced_parser.add_argument(
        "--handover-time",
        dest="handover_time",
        default="03:00:00",
        metavar="HH:MM:SS",
        help=(
            "Overlap window for consecutive jobs (default: 03:00:00). Each job "
            "starts this long before its predecessor's time limit, giving the "
            "fresh job time to become healthy before the old one expires."
        ),
    )
    advanced_parser.add_argument(
        "--max-job-time",
        dest="max_job_time",
        default=DEFAULT_MAX_JOB_TIME,
        metavar="HH:MM:SS",
        help=f"Per-job SLURM time cap for consecutive chains (default: {DEFAULT_MAX_JOB_TIME}).",
    )
    advanced_parser.add_argument(
        "--slurm-reservation",
        dest="reservation",
        default=os.environ.get("SML_RESERVATION"),
        metavar="RESERVATION",
        help="SLURM reservation name (optional, env: SML_RESERVATION).",
    )
    advanced_parser.add_argument(
        "--served-model-name",
        dest="served_model_name",
        default=None,
        help="Name under which the model will be served. Auto-generated if omitted.",
    )
    advanced_parser.add_argument(
        "--router",
        dest="router",
        choices=[ROUTER_OCF, ROUTER_SGL],
        type=str.upper,
        default=ROUTER_OCF,
        help=(
            "Routing strategy across replicas. "
            f"'{ROUTER_OCF}' (default): OpenTela load-balances across the replica peers "
            f"on the mesh. '{ROUTER_SGL}': an in-job SGLang router fronts the replicas and "
            "becomes the served endpoint (needs replicas > 1)."
        ),
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
        "--otela-bootstrap-addr",
        dest="otela_bootstrap_addr",
        default=None,
        metavar="MULTIADDR",
        help=(
            "Override the OCF bootstrap multiaddr "
            "(e.g. /ip4/<host>/tcp/<port>/p2p/<peer-id>). "
            "Takes precedence over --dev. Defaults to the prod peer."
        ),
    )
    advanced_parser.add_argument(
        "--dev",
        dest="dev",
        action="store_true",
        help=("Shorthand for the dev OCF bootstrap peer. Ignored if --otela-bootstrap-addr is also set."),
    )
    advanced_parser.add_argument(
        "--disable-dcgm-exporter",
        dest="disable_dcgm_exporter",
        action="store_true",
        help="Disable the DCGM exporter.",
    )
    advanced_parser.add_argument(
        "--disable-metrics",
        dest="disable_metrics",
        action="store_true",
        help="Disable metrics collection.",
    )
    advanced_parser.add_argument(
        "--pre-launch-cmds",
        dest="pre_launch_cmds",
        default="",
        metavar="CMDS",
        help="Commands to run before launching the model.",
    )
    if tui_default is not None:
        advanced_parser.add_argument(
            "--tui",
            action=argparse.BooleanOptionalAction,
            default=tui_default,
            help="Launch the interactive TUI after submitting the job.",
        )
    advanced_parser.add_argument(
        "--output-script",
        dest="output_script",
        metavar="DIR",
        default=None,
        help=(
            "Render master.sh + per-shape rank scripts (head, follower, router) into "
            "the given directory and exit without submitting. Each file is "
            "independently shellcheckable; master.sh self-extracts the rank scripts "
            "at job start so you can also `sbatch DIR/master.sh` directly. The "
            "on-disk layout matches what a real submission writes to ~/.sml/job-<id>/."
        ),
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
    _make_slurm_account_config().add_to_parser(preconfigured_parser)
    _make_launch_request_config().add_to_parser(preconfigured_parser)

    advanced_parser = subparsers.add_parser("advanced", help="Launch a model with advanced configuration")
    _make_firecrest_launcher_config().add_to_parser(advanced_parser)
    _make_partition_config().add_to_parser(advanced_parser)
    _make_slurm_account_config().add_to_parser(advanced_parser)
    _add_advanced_launch_arguments(advanced_parser, tui_default=False)

    preconfigured_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch the interactive TUI after submitting the job.",
    )

    add_loadtest_parser(
        subparsers,
        make_firecrest_launcher_config=_make_firecrest_launcher_config,
        make_partition_config=_make_partition_config,
        make_reservation_config=_make_reservation_config,
        make_launch_request_config=_make_launch_request_config,
        add_advanced_launch_arguments=lambda p: _add_advanced_launch_arguments(p, tui_default=None),
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
        reservation = (
            (getattr(args, "reservation", None) if args else None) or os.environ.get("SML_RESERVATION") or None
        )
        account = (getattr(args, "slurm_account", None) if args else None) or os.environ.get("SML_ACCOUNT") or None
    else:
        reservation_config = _make_reservation_config()
        await reservation_config.aconfigure(args=args)
        reservation = reservation_config.get_value("reservation") or None

        slurm_account_config = _make_slurm_account_config()
        await slurm_account_config.aconfigure(args=args)
        account = slurm_account_config.get_value("slurm_account") or None

    return await FirecRESTLauncher.from_client(
        client=client,
        system_name=system_name,
        partition=partition_config.get_non_none_value("partition"),
        reservation=reservation,
        account=account,
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
        reservation = (
            (getattr(args, "reservation", None) if args else None) or os.environ.get("SML_RESERVATION") or None
        )
        account = (
            (getattr(args, "slurm_account", None) if args else None)
            or os.environ.get("SML_ACCOUNT")
            or grp.getgrgid(os.getgid()).gr_name
        )
    else:
        reservation_config = _make_reservation_config()
        await reservation_config.aconfigure(args=args)
        reservation = reservation_config.get_value("reservation") or None

        slurm_account_config = _make_slurm_account_config()
        await slurm_account_config.aconfigure(args=args)
        account = slurm_account_config.get_value("slurm_account") or grp.getgrgid(os.getgid()).gr_name

    return SlurmLauncher(
        system_name="local",
        username=getpass.getuser(),
        account=account,
        partition=partition_config.get_non_none_value("partition"),
        reservation=reservation,
        telemetry_endpoint=telemetry_endpoint,
    )


async def _get_router_options(get_value: GetValueFn) -> dict[str, tuple[str, str]]:
    replicas = get_value("replicas")
    # The in-job SGLang router only makes sense with more than one replica to
    # balance across; with a single replica only mesh-level OCF routing applies.
    if replicas is not None and int(replicas) > 1:
        return {
            ROUTER_OCF: ("OCF", "OpenTela load-balances across the replica peers on the mesh"),
            ROUTER_SGL: ("SGL", "In-job SGLang router fronts the replicas"),
        }
    return {
        ROUTER_OCF: ("OCF", "OpenTela load-balances across the replica peers on the mesh"),
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
        router_factory=lambda get_value: _get_router_options(get_value),
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
        replicas=int(launch_req_config.get_non_none_value("replicas")),
        time=launch_req_config.get_non_none_value("time"),
        served_model_name=f"{model}-{create_salt(4)}",
        router=cast(RouterMode, launch_req_config.get_non_none_value("router")),
    )


_logger = logging.getLogger(__name__)


async def _create_launcher(
    config: InitConfig,
    args: argparse.Namespace,
    non_interactive: bool = False,
) -> Launcher:
    launcher_type = config.get_non_none_value("launcher")

    if launcher_type == "slurm" and getattr(args, "firecrest_system", None):
        _logger.warning("--firecrest-system is ignored when using the SLURM launcher")

    if launcher_type == "firecrest":
        firecrest_client = _get_firecrest_client_from_init_config(config)
        return cast(
            Launcher,
            await _get_firecrest_launcher_with_client(
                firecrest_client,
                telemetry_endpoint=TELEMETRY_ENDPOINT,
                args=args,
                non_interactive=non_interactive,
            ),
        )
    elif launcher_type == "slurm":
        return cast(
            Launcher,
            await _get_slurm_launcher(
                telemetry_endpoint=TELEMETRY_ENDPOINT,
                args=args,
                non_interactive=non_interactive,
            ),
        )
    else:
        raise NotImplementedError(f"Launcher {launcher_type} is not supported yet.")


def _log_sources(expected_replicas: int, router: RouterMode) -> list[tuple[str, str, str]]:
    """The log sources shown as TUI tabs, as (label, stdout_file, stderr_file).

    The master's own orchestration output is in log.out/log.err; each replica's
    framework output is in replica_<r>.out/.err (see framework._render_replica_launches).
    """
    sources = [("Master", "log.out", "log.err")]
    sources += [(f"Replica {r}", f"replica_{r}.out", f"replica_{r}.err") for r in range(expected_replicas)]
    if router == ROUTER_SGL and expected_replicas > 1:
        sources.append(("Router", "router.out", "router.err"))
    return sources


def _focus_job(scheduled: list[ScheduledJob], statuses: dict[int, JobStatus]) -> ScheduledJob:
    """Pick the chain job whose replicas/logs the panels should follow.

    After a handover, the newest RUNNING job is what's actually serving, so it
    wins. Before anything is running, focus the next job still PENDING so the user
    watches it come up; once the whole chain is gone, fall back to the last job.
    Only RUNNING and PENDING jobs are candidates here, so any terminal state
    (COMPLETED / CANCELLED / FAILED / TIMEOUT, or UNKNOWN once squeue drops the
    job and sacct has nothing) is simply skipped.
    """
    running = [s for s in scheduled if statuses.get(s.job_id) == JobStatus.RUNNING]
    if running:
        return running[-1]
    pending = [s for s in scheduled if statuses.get(s.job_id) == JobStatus.PENDING]
    if pending:
        return pending[0]
    return scheduled[-1]


async def _as_single_chain(coro: Coroutine[Any, Any, tuple[int, str]]) -> list[ScheduledJob]:
    job_id, served = await coro
    return [ScheduledJob(job_id=job_id, served_model_name=served, begin=None)]


async def _run_monitor(
    launcher: Launcher,
    submit_coro: Coroutine[Any, Any, list[ScheduledJob]],
    cscs_api_key: str,
    *,
    expected_replicas: int,
    router: RouterMode = ROUTER_OCF,
) -> None:
    sources = _log_sources(expected_replicas, router)
    source_files = {label: (out_file, err_file) for label, out_file, err_file in sources}
    state = DisplayState([label for label, _, _ in sources])
    state.update(cluster=launcher.system_name, partition=launcher.partition)

    async def _monitor_model_health(served: str) -> None:
        # End-to-end probe through the public gateway. It can block for its full
        # timeout, so it runs in its own loop — see _monitor_jobs.
        ever_healthy = False
        while True:
            await asyncio.sleep(5)
            model_health = await check_model_health(served, cscs_api_key)
            if model_health == ModelHealth.NOT_RESPONDING and not ever_healthy:
                model_health = ModelHealth.NOT_DEPLOYED
            ever_healthy = ever_healthy or model_health == ModelHealth.HEALTHY
            state.update(model_health=model_health)

    async def _monitor_jobs(scheduled: list[ScheduledJob], served: str) -> None:
        # Job status, per-replica health, and logs — independent of the e2e
        # gateway probe so a slow probe never delays these updates. For a
        # consecutive chain we poll every job's status and follow the one that's
        # currently serving (see _focus_job) for logs, while showing the replicas
        # of *all* running jobs so an overlapping handover is visible. A single
        # launch keeps the old unlabelled single-report behaviour. Only the
        # focused job's active-tab logs are fetched, so log traffic stays bounded.
        is_chain = len(scheduled) > 1
        while True:
            await asyncio.sleep(5)
            statuses: dict[int, JobStatus] = {}
            for job in scheduled:
                status = await launcher.get_job_status(job.job_id)
                statuses[job.job_id] = status
                # For a chain, also pull the backend's real start/end so the panel
                # shows actual scheduled times rather than submission-time guesses.
                begin, end = await launcher.get_job_times(job.job_id) if is_chain else (None, None)
                state.set_chain_status(job.job_id, status, begin, end)
            focused = _focus_job(scheduled, statuses)
            state.update(job_id=focused.job_id, job_status=statuses[focused.job_id])
            if is_chain:
                running = [s for s in scheduled if statuses.get(s.job_id) == JobStatus.RUNNING]
                reports: list[tuple[int, ReplicaHealthReport]] = []
                for job in running:
                    report = await launcher.get_replica_health(job.job_id, served, expected_replicas)
                    if report is not None:
                        reports.append((job.job_id, report))
                # Always replace (even with []) so the panel tracks the current
                # RUNNING set — a finished/cancelled job's replicas stop showing
                # instead of lingering as stale per-job sections.
                state.set_replica_reports(reports)
            else:
                report = await launcher.get_replica_health(focused.job_id, served, expected_replicas)
                if report is not None:
                    state.set_replica_report(report)
            source = state.active_source
            out_file, err_file = source_files[source]
            out = await launcher.read_job_file(focused.job_id, out_file)
            err = await launcher.read_job_file(focused.job_id, err_file)
            state.set_source_log(source, out or "", err or "")

    async def _monitor() -> None:
        scheduled = await submit_coro
        served = scheduled[0].served_model_name
        # Seed only the dependency hint; the monitor fills begin/end with the
        # backend's real scheduled times on the first poll (see _monitor_jobs).
        state.set_chain([ChainJobView(job_id=s.job_id, after=s.after) for s in scheduled])
        state.update(
            job_id=scheduled[0].job_id,
            served_model_name=served,
            model_health=ModelHealth.NOT_DEPLOYED,
        )
        await asyncio.gather(_monitor_model_health(served), _monitor_jobs(scheduled, served))

    kill_job = await LiveDisplay(state).run(_monitor())
    if kill_job:
        # Kill the whole chain, not just the focused job, so a consecutive launch
        # leaves nothing scheduled behind.
        for job in state.chain:
            await launcher.cancel_job(job.job_id)


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
        await _run_monitor(
            launcher,
            _as_single_chain(launch_coro),
            cscs_api_key,
            expected_replicas=launch_request.replicas,
            router=launch_request.router,
        )
    else:
        job_id, served = await launch_coro
        print(f"Job submitted: {job_id}")
        print(f"Served model name: {served}")
        print(f"Logs: {launcher.get_log_dir(job_id)}")
        print(f"Tail: {launcher.get_tail_hint(job_id)}")


def build_launch_args_from_advanced(
    args: argparse.Namespace,
    *,
    account: str,
    partition: str,
    telemetry_endpoint: str | None = None,
) -> LaunchArgs:
    """Build a LaunchArgs from a parsed `sml advanced` namespace.

    Tests can drive this directly to validate that example shell scripts produce
    a valid LaunchArgs without going through the launcher / InitConfig.
    """
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

    ocf_bootstrap_addr: str | None
    if getattr(args, "otela_bootstrap_addr", None):
        ocf_bootstrap_addr = args.otela_bootstrap_addr
        if getattr(args, "dev", False):
            print(
                "warning: --dev ignored because --otela-bootstrap-addr was given.",
                file=sys.stderr,
            )
    elif getattr(args, "dev", False):
        ocf_bootstrap_addr = OCF_BOOTSTRAP_ADDR_DEV
    else:
        ocf_bootstrap_addr = None

    return LaunchArgs(
        job_name=job_name,
        served_model_name=served_model_name,
        account=account,
        partition=partition,
        topology=Topology(
            replicas=args.replicas,
            nodes_per_replica=args.nodes_per_replica,
        ),
        time=args.time,
        environment=args.slurm_environment,
        framework=args.framework,
        framework_args=args.framework_args,
        pre_launch_cmds=args.pre_launch_cmds,
        router=args.router,
        router_args=args.router_args,
        disable_ocf=args.disable_ocf,
        ocf_bootstrap_addr=ocf_bootstrap_addr,
        dev=getattr(args, "dev", False),
        disable_dcgm_exporter=args.disable_dcgm_exporter,
        disable_metrics=args.disable_metrics,
        telemetry_endpoint=telemetry_endpoint,
    )


async def _run_advanced(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await _create_launcher(config, args, non_interactive=True)
    cscs_api_key = config.get_non_none_value("cscs_api_key")

    launch_args = build_launch_args_from_advanced(
        args,
        account=launcher.account,
        partition=launcher.partition,
        telemetry_endpoint=TELEMETRY_ENDPOINT,
    )

    # Decide single vs. consecutive chain. --time is the total uptime; a job is
    # capped at --max-job-time, so anything longer needs --consecutive and is
    # served by a pre-scheduled chain (each job runs for the cap).
    total_seconds = time_str_to_seconds(args.time)
    max_job_seconds = time_str_to_seconds(args.max_job_time)
    consecutive = getattr(args, "consecutive", False)
    is_chain = consecutive and total_seconds > max_job_seconds
    if total_seconds > max_job_seconds and not consecutive:
        print(
            f"--time ({args.time}) exceeds the per-job cap (--max-job-time {args.max_job_time}). "
            f"Pass --consecutive to serve it with a chain of jobs.",
            file=sys.stderr,
        )
        return
    if consecutive and not is_chain:
        print(
            f"--consecutive ignored: --time ({args.time}) fits within one job "
            f"(--max-job-time {args.max_job_time}); submitting a single job.",
            file=sys.stderr,
        )
    if is_chain:
        if time_str_to_seconds(args.handover_time) >= max_job_seconds:
            print(
                f"--handover-time ({args.handover_time}) must be shorter than --max-job-time ({args.max_job_time}).",
                file=sys.stderr,
            )
            return
        # Each job in the chain runs for the per-job cap, not the full --time.
        launch_args = launch_args.model_copy(update={"time": args.max_job_time})

    if args.output_script:
        # Write master.sh + per-shape rank scripts to a user-supplied dir.
        # The on-disk shape matches what a live submission would write to
        # ~/.sml/job-<id>/ at job start (via master's self-extract), so
        # `sbatch <dir>/master.sh` is also a valid way to submit manually.
        out_dir = Path(args.output_script).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        master_path = out_dir / "master.sh"
        master_path.write_text(
            render_sbatch_header(launch_args, reservation=launcher.reservation) + render_master(launch_args)
        )
        master_path.chmod(0o755)

        written = ["master.sh"]
        for filename, content in render_rank_scripts(launch_args).items():
            path = out_dir / filename
            path.write_text(content)
            path.chmod(0o755)
            written.append(filename)

        print(f"Wrote {len(written)} file(s) to {out_dir}:", file=sys.stderr)
        for name in written:
            print(f"  {name}", file=sys.stderr)
        return

    if is_chain:
        submit_coro: Coroutine[Any, Any, list[ScheduledJob]] = launcher.launch_consecutive_with_args(
            launch_args,
            total_time=args.time,
            handover_time=args.handover_time,
        )
    else:
        submit_coro = _as_single_chain(launcher.launch_with_args(launch_args))

    if args.tui:
        await _run_monitor(
            launcher,
            submit_coro,
            cscs_api_key,
            expected_replicas=launch_args.topology.replicas,
            router=launch_args.router,
        )
    else:
        scheduled = await submit_coro
        if len(scheduled) > 1:
            print(f"Submitted a consecutive chain of {len(scheduled)} jobs:")
            for job in scheduled:
                when = f"runs {job.begin} → {job.end}" if job.begin else (job.after or "starts on handover")
                print(f"  Job {job.job_id} {when}")
            print(f"Served model name: {scheduled[0].served_model_name}")
            print(f"Logs (first job): {launcher.get_log_dir(scheduled[0].job_id)}")
        else:
            job = scheduled[0]
            print(f"Job submitted: {job.job_id}")
            print(f"Served model name: {job.served_model_name}")
            print(f"Logs: {launcher.get_log_dir(job.job_id)}")
            print(f"Tail: {launcher.get_tail_hint(job.job_id)}")


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
        await run_loadtest_command(
            args,
            create_launcher=_create_launcher,
            get_launch_request=_get_launch_request,
            build_launch_args_from_advanced=build_launch_args_from_advanced,
        )


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
