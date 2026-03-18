import asyncio
import re
from collections.abc import Awaitable, Callable

import firecrest as f7t

from swiss_ai_model_launch.cli.configuration import InitConfig
from swiss_ai_model_launch.cli.configuration.models import (
    ChainConfiguration,
    GetValueFn,
    OptionsConfiguration,
    TextConfiguration,
)
from swiss_ai_model_launch.cli.display import DisplayState, LiveDisplay
from swiss_ai_model_launch.cli.healthcheck import check_model_health
from swiss_ai_model_launch.launchers import FirecRESTLauncher, Launcher
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.utils import create_salt


async def _run_initial_configuration_wizard() -> None:
    config = InitConfig()
    await config.aconfigure()
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

    launcher_config = ChainConfiguration(
        name="firecrest_launcher_configuration",
        chain=[
            OptionsConfiguration(
                name="firecrest_system",
                prompt="Choose the target system to launch the model on.",
                options_factory=_get_systems,
            ),
            OptionsConfiguration(
                name="firecrest_partition",
                prompt="Choose the partition to launch the model on.",
                options_factory=_get_partitions,
            ),
        ],
    )
    await launcher_config.aconfigure()

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


async def _get_preconfigured_default(
    get_value_from_context: GetValueFn, preconfigured: list[LaunchRequest], field: str
) -> str | None:
    vendor = get_value_from_context("model_vendor")
    model_name = get_value_from_context("model_name")
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


def _make_served_model_name_default(
    preconfigured: list[LaunchRequest],
) -> Callable[[GetValueFn], Awaitable[str]]:
    async def _default(get_value: GetValueFn) -> str:
        value = await _get_preconfigured_default(
            get_value, preconfigured, "served_model_name"
        )
        if value and value != "None":
            return value
        return f"{get_value('model_vendor')}/{get_value('model_name')}-{create_salt(4)}"

    return _default


async def _get_launch_request(launcher: Launcher) -> LaunchRequest:
    preconfigured_launch_requests = await launcher.get_preconfigured_models()

    async def _get_vendors() -> dict[str, tuple[str, str]]:
        return {
            lr.vendor: (lr.vendor, lr.vendor) for lr in preconfigured_launch_requests
        }

    async def _get_models(
        get_value_from_context: GetValueFn,
    ) -> dict[str, tuple[str, str]]:
        vendor = get_value_from_context("model_vendor")
        return {
            lr.model_name: (lr.model_name, lr.model_name)
            for lr in preconfigured_launch_requests
            if lr.vendor == vendor
        }

    async def _get_frameworks(
        get_value_from_context: GetValueFn,
    ) -> dict[str, tuple[str, str]]:
        vendor = get_value_from_context("model_vendor")
        model_name = get_value_from_context("model_name")
        return {
            lr.framework: (lr.framework, lr.framework)
            for lr in preconfigured_launch_requests
            if lr.model_name == model_name and lr.vendor == vendor
        }

    launch_req_config = ChainConfiguration(
        name="launcher_request_configuration",
        chain=[
            OptionsConfiguration(
                name="model_vendor",
                prompt="Choose the model vendor.",
                options_factory=_get_vendors,
            ),
            OptionsConfiguration(
                name="model_name",
                prompt="Choose the model to launch.",
                options_factory=_get_models,
            ),
            OptionsConfiguration(
                name="framework",
                prompt="Choose the framework to run the model with.",
                options_factory=_get_frameworks,
            ),
            TextConfiguration(
                name="workers",
                prompt="Number of workers to use for running the model.",
                validator=lambda v: v.isdigit() and int(v) > 0,
                default_factory=lambda get_value: _get_preconfigured_default(
                    get_value, preconfigured_launch_requests, "workers"
                ),
            ),
            TextConfiguration(
                name="nodes_per_worker",
                prompt="Number of nodes to use per worker for running the model.",
                validator=lambda v: v.isdigit() and int(v) > 0,
                default_factory=lambda get_value: _get_preconfigured_default(
                    get_value, preconfigured_launch_requests, "nodes_per_worker"
                ),
            ),
            TextConfiguration(
                name="time",
                prompt="Time duration for running the model (in format HH:MM:SS).",
                validator=lambda v: bool(
                    re.fullmatch(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]", v)
                ),
                default_factory=lambda get_value: _get_preconfigured_default(
                    get_value, preconfigured_launch_requests, "time"
                ),
            ),
            TextConfiguration(
                name="served_model_name",
                prompt="Served model name.",
                validator=lambda s: len(s) > 0,
                default_factory=_make_served_model_name_default(
                    preconfigured_launch_requests
                ),
            ),
        ],
    )
    await launch_req_config.aconfigure()

    return LaunchRequest(
        vendor=launch_req_config.get_non_none_value("model_vendor"),
        model_name=launch_req_config.get_non_none_value("model_name"),
        framework=launch_req_config.get_non_none_value("framework"),
        workers=int(launch_req_config.get_non_none_value("workers")),
        nodes_per_worker=int(launch_req_config.get_non_none_value("nodes_per_worker")),
        time=launch_req_config.get_non_none_value("time"),
        served_model_name=launch_req_config.get_non_none_value("served_model_name"),
    )


async def _main() -> None:
    if not InitConfig.exists():
        await _run_initial_configuration_wizard()
        return

    config = InitConfig.load()
    launcher_type = config.get_non_none_value("launcher")
    if launcher_type == "firecrest":
        firecrest_client = _get_firecrest_client_from_init_config(config)
        telemetry_endpoint = config.get_non_none_value("telemetry_endpoint")
        launcher = await _get_firecrest_launcher_with_client(
            firecrest_client,
            telemetry_endpoint=telemetry_endpoint,
        )
    else:
        raise NotImplementedError(f"Launcher {launcher_type} is not supported yet.")

    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_request = await _get_launch_request(launcher)

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
            state.append_out_log(o)
            state.append_err_log(e)

    await LiveDisplay(state).run(_monitor())


def main() -> None:
    asyncio.run(_main())
