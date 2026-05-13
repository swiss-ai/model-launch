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
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs, Topology
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.launchers.utils import create_salt
from swiss_ai_model_launch.mcp import mcp as _mcp

_OptionsFactory = Callable[[], Awaitable[OptionsDict]] | Callable[[GetValueFn], Awaitable[OptionsDict]] | None


def _resolve_legacy(
    new_value: int | None, legacy_value: int | None, legacy_flag: str, new_flag: str, *, default: int
) -> int:
    if legacy_value is not None:
        print(f"warning: {legacy_flag} is deprecated; use {new_flag} instead.", file=sys.stderr)
        if new_value is None:
            return legacy_value
    return new_value if new_value is not None else default


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
        "yes": ("Yes", "Use router to load balance across replicas"),
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
                name="replicas",
                prompt="Number of replicas (independent inference engine instances) to launch.",
                validator=lambda v: v.isdigit() and int(v) > 0,
                default="1",
            ),
            OptionsConfiguration(
                name="use_router",
                prompt="Use router to load balance across replicas.",
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
        "--slurm-replicas",
        dest="replicas",
        type=int,
        default=None,
        help="Number of independent inference engine instances (default: 1).",
    )
    advanced_parser.add_argument(
        "--slurm-workers",
        dest="legacy_workers",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_parser.add_argument(
        "--slurm-nodes-per-replica",
        dest="nodes_per_replica",
        type=int,
        default=None,
        help=(
            "Number of nodes spanned by one replica (default: 1). "
            "Set this to match your TP/PP/DP/EP layout in --framework-args."
        ),
    )
    advanced_parser.add_argument(
        "--slurm-nodes-per-worker",
        dest="legacy_nodes_per_worker",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_parser.add_argument(
        "--slurm-nodes",
        dest="legacy_nodes",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
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
        "--replica-port",
        dest="legacy_replica_port",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_parser.add_argument(
        "--worker-port",
        dest="legacy_worker_port",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    advanced_parser.add_argument(
        "--use-router",
        dest="use_router",
        action="store_true",
        help="Enable router to load balance across replicas.",
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
    advanced_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        reservation = (
            (getattr(args, "reservation", None) if args else None) or os.environ.get("SML_RESERVATION") or None
        )
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
    replicas = get_value("replicas")
    if replicas is not None and int(replicas) > 1:
        return {
            "yes": ("Yes", "Use router to load balance across replicas"),
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
        replicas=int(launch_req_config.get_non_none_value("replicas")),
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


def build_launch_args_from_advanced(
    args: argparse.Namespace,
    *,
    account: str,
    partition: str,
    telemetry_endpoint: str | None = None,
) -> LaunchArgs:
    """Build a LaunchArgs from a parsed `sml advanced` namespace.

    Pure: no side effects beyond stderr warnings for legacy flags. Tests can
    drive this directly to validate that example shell scripts produce a
    valid LaunchArgs without going through the launcher / InitConfig.
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

    replicas = _resolve_legacy(args.replicas, args.legacy_workers, "--slurm-workers", "--slurm-replicas", default=1)
    nodes_per_replica = _resolve_legacy(
        args.nodes_per_replica,
        args.legacy_nodes_per_worker,
        "--slurm-nodes-per-worker",
        "--slurm-nodes-per-replica",
        default=1,
    )
    for legacy_value, legacy_flag in (
        (args.legacy_replica_port, "--replica-port"),
        (args.legacy_worker_port, "--worker-port"),
    ):
        if legacy_value is not None:
            print(
                f"warning: {legacy_flag} is no longer configurable; the framework port is hardcoded "
                "to 8080. Drop the flag.",
                file=sys.stderr,
            )
    if args.legacy_nodes is not None:
        print(
            "warning: --slurm-nodes is no longer configurable; total nodes is derived from "
            "--slurm-replicas * --slurm-nodes-per-replica. Drop the flag.",
            file=sys.stderr,
        )

    return LaunchArgs(
        job_name=job_name,
        served_model_name=served_model_name,
        account=account,
        partition=partition,
        topology=Topology(
            replicas=replicas,
            nodes_per_replica=nodes_per_replica,
        ),
        time=args.time,
        reservation=args.reservation or None,
        environment=args.slurm_environment,
        framework=args.framework,
        framework_args=args.framework_args,
        pre_launch_cmds=args.pre_launch_cmds,
        use_router=args.use_router,
        router_args=args.router_args,
        disable_ocf=args.disable_ocf,
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
        telemetry_endpoint=config.get_value("telemetry_endpoint"),
    )

    if args.output_script:
        from pathlib import Path

        from swiss_ai_model_launch.launchers.framework import render_master, render_rank_scripts
        from swiss_ai_model_launch.launchers.utils import render_sbatch_header

        # Write master.sh + per-shape rank scripts to a user-supplied dir.
        # The on-disk shape matches what a live submission would write to
        # ~/.sml/job-<id>/ at job start (via master's self-extract), so
        # `sbatch <dir>/master.sh` is also a valid way to submit manually.
        out_dir = Path(args.output_script).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        master_path = out_dir / "master.sh"
        master_path.write_text(render_sbatch_header(launch_args) + render_master(launch_args))
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
