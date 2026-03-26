import argparse
import asyncio
import re
from collections.abc import Awaitable, Callable

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
from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.utils import create_salt

_OptionsFactory = (
    Callable[[], Awaitable[OptionsDict]]
    | Callable[[GetValueFn], Awaitable[OptionsDict]]
    | None
)
_DefaultFactory = (
    Callable[[], Awaitable[str | None]]
    | Callable[[GetValueFn], Awaitable[str | None]]
    | None
)


def _make_firecrest_launcher_config(
    systems_factory: _OptionsFactory = None,
    partitions_factory: _OptionsFactory = None,
) -> ChainConfiguration:
    """Build the FirecREST launcher config.

    Pass factories for interactive/runtime use; omit them to get a static shell
    suitable only for parser registration (options={} marks dynamic choices).
    """
    _empty: OptionsDict = {}
    return ChainConfiguration(
        name="firecrest_launcher_configuration",
        chain=[
            OptionsConfiguration(
                name="firecrest_system",
                prompt="Choose the target system to launch the model on.",
                options_factory=systems_factory,
                options=None if systems_factory else _empty,
            ),
            OptionsConfiguration(
                name="firecrest_partition",
                prompt="Choose the partition to launch the model on.",
                options_factory=partitions_factory,
                options=None if partitions_factory else _empty,
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
                validator=(
                    (lambda v: v.isdigit() and int(v) > 0)
                    if workers_default_factory
                    else None
                ),
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
                validator=lambda v: bool(
                    re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", v)
                ),
                default_factory=time_default_factory,
            ),
        ],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sml",
        description="Swiss AI Model Launcher",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=False)

    init_parser = subparsers.add_parser("init", help="Initialize SML configuration")
    InitConfig().add_to_parser(init_parser)

    quickstart_parser = subparsers.add_parser(
        "quickstart", help="Launch a model with guided prompts"
    )
    _make_firecrest_launcher_config().add_to_parser(quickstart_parser)
    _make_launch_request_config().add_to_parser(quickstart_parser)

    subparsers.add_parser("advanced", help="Launch a model with advanced configuration")

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
) -> FirecRESTLauncher:
    async def _get_systems() -> dict[str, tuple[str, str]]:
        return {
            sys["name"]: (sys["name"], sys["ssh"]["host"])
            for sys in await client.systems()
        }

    async def _get_partitions(
        get_value_from_context: GetValueFn,
    ) -> dict[str, tuple[str, str]]:
        system = get_value_from_context("firecrest_system")
        if system is None:
            raise ValueError("firecrest_system is not set")
        return {
            part["name"]: (part["name"], part["name"])
            for part in await client.partitions(system)
        }

    launcher_config = _make_firecrest_launcher_config(
        systems_factory=_get_systems,
        partitions_factory=_get_partitions,
    )
    await launcher_config.aconfigure(args=args)

    system_name = launcher_config.get_non_none_value("firecrest_system")
    user_info = await client.userinfo(system_name)

    return FirecRESTLauncher(
        client,
        system_name=system_name,
        username=user_info["user"]["name"],
        account=user_info["group"]["name"],
        partition=launcher_config.get_non_none_value("firecrest_partition"),
        telemetry_endpoint=telemetry_endpoint,
    )


def _split_vendor_model(combined: str) -> tuple[str, str]:
    vendor, model_name = combined.split("::", 1)
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
            if lr.vendor == vendor
            and lr.model_name == model_name
            and lr.framework == framework
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


async def _get_launch_request(
    launcher: Launcher, args: argparse.Namespace | None = None
) -> LaunchRequest:
    preconfigured_launch_requests = await launcher.get_preconfigured_models()

    async def _get_vendor_models() -> dict[str, tuple[str, str]]:
        seen: dict[str, tuple[str, str]] = {}
        for lr in preconfigured_launch_requests:
            key = f"{lr.vendor}::{lr.model_name}"
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

    vendor, model_name = _split_vendor_model(
        launch_req_config.get_non_none_value("model")
    )
    framework = launch_req_config.get_non_none_value("framework")
    preconfigured = next(
        (
            lr
            for lr in preconfigured_launch_requests
            if lr.vendor == vendor
            and lr.model_name == model_name
            and lr.framework == framework
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


async def _run_quickstart(args: argparse.Namespace) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher_type = config.get_non_none_value("launcher")
    if launcher_type == "firecrest":
        firecrest_client = _get_firecrest_client_from_init_config(config)
        telemetry_endpoint = config.get_non_none_value("telemetry_endpoint")
        launcher = await _get_firecrest_launcher_with_client(
            firecrest_client,
            telemetry_endpoint=telemetry_endpoint,
            args=args,
        )
    else:
        raise NotImplementedError(f"Launcher {launcher_type} is not supported yet.")

    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_request = await _get_launch_request(launcher, args)

    state = DisplayState()
    state.update(cluster=launcher.system_name, partition=launcher.partition)

    async def _monitor() -> None:
        job_id, served_model_name = await launcher.launch_model(launch_request)
        state.update(job_id=job_id, served_model_name=served_model_name)
        while True:
            await asyncio.sleep(5)

            job_status = await launcher.get_job_status(job_id)
            state.update(job_status=job_status)

            model_health = await check_model_health(served_model_name, cscs_api_key)
            state.update(model_health=model_health)

            o, e = await launcher.get_job_logs(job_id)
            state.set_out_log(o)
            state.set_err_log(e)

    await LiveDisplay(state).run(_monitor())


async def _main(args: argparse.Namespace) -> None:
    subcommand = args.subcommand or ("quickstart" if InitConfig.exists() else "init")
    if subcommand == "init":
        await _run_initial_configuration_wizard(args)
    elif subcommand == "quickstart":
        await _run_quickstart(args)
    elif subcommand == "advanced":
        raise NotImplementedError("Advanced configuration is not yet implemented.")


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
