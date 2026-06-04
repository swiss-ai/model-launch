import argparse
import asyncio
import os
import re
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import questionary

from swiss_ai_model_launch.cli.configuration import InitConfig
from swiss_ai_model_launch.cli.healthcheck import check_model_health
from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth
from swiss_ai_model_launch.launchers import Launcher
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.loadtest.cluster import (
    DEFAULT_CLUSTER_LOADTEST_TIME,
    ClusterLoadtestConfig,
    submit_cluster_loadtest,
)
from swiss_ai_model_launch.loadtest.models import LoadtestConfig, ServerConfig, load_scenarios
from swiss_ai_model_launch.loadtest.setup import (
    DEFAULT_CLUSTER_CONTAINER_IMAGE,
    K6_SCRIPT,
    resolve_prompts_file,
)

_DEFAULT_LOADTEST_SERVER_URL = "https://api.swissai.svc.cscs.ch"
_DEFAULT_LOADTEST_READY_TIMEOUT_SECONDS = 1000000
_DEFAULT_LOADTEST_READY_POLL_SECONDS = 10
_LOADTEST_READY_PROGRESS_SECONDS = 300
_DEFAULT_LOADTEST_METRICS_REMOTE_WRITE_URL = "https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write"
_DEFAULT_LOADTEST_SCENARIO = "throughput"

_AddConfig = Callable[[], Any]
_AddAdvancedLaunchArguments = Callable[[argparse.ArgumentParser], None]


class _CreateLauncher(Protocol):
    async def __call__(
        self,
        config: InitConfig,
        args: argparse.Namespace,
        non_interactive: bool = False,
    ) -> Launcher: ...


class _GetLaunchRequest(Protocol):
    async def __call__(
        self,
        launcher: Launcher,
        args: argparse.Namespace | None = None,
    ) -> LaunchRequest: ...


class _BuildLaunchArgsFromAdvanced(Protocol):
    def __call__(
        self,
        args: argparse.Namespace,
        *,
        account: str,
        partition: str,
        telemetry_endpoint: str | None = None,
    ) -> LaunchArgs: ...


def _add_loadtest_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_cancel: bool = True,
    include_health_wait: bool = True,
) -> None:
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
        "--loadtest-job-time",
        dest="loadtest_job_time",
        default=DEFAULT_CLUSTER_LOADTEST_TIME,
        metavar="HH:MM:SS",
        help=f"SLURM time limit for the cluster k6 loadtest job (default: {DEFAULT_CLUSTER_LOADTEST_TIME}).",
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
        default=None,
        help=f"Built-in or custom loadtest scenario name (default: {_DEFAULT_LOADTEST_SCENARIO}).",
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


def add_loadtest_parser(
    subparsers: Any,
    *,
    make_firecrest_launcher_config: _AddConfig,
    make_partition_config: _AddConfig,
    make_reservation_config: _AddConfig,
    make_launch_request_config: _AddConfig,
    add_advanced_launch_arguments: _AddAdvancedLaunchArguments,
) -> None:
    loadtest_parser = subparsers.add_parser("loadtest", help="Launch models and run loadtests")
    loadtest_subparsers = loadtest_parser.add_subparsers(dest="loadtest_command", required=True)

    loadtest_preconfigured_parser = loadtest_subparsers.add_parser(
        "preconfigured",
        help="Launch one preconfigured model, wait until healthy, then run a loadtest",
    )
    make_firecrest_launcher_config().add_to_parser(loadtest_preconfigured_parser)
    make_partition_config().add_to_parser(loadtest_preconfigured_parser)
    make_reservation_config().add_to_parser(loadtest_preconfigured_parser)
    make_launch_request_config().add_to_parser(loadtest_preconfigured_parser)
    _add_loadtest_arguments(loadtest_preconfigured_parser)

    loadtest_advanced_parser = loadtest_subparsers.add_parser(
        "advanced",
        help="Launch a model with the same arguments as `sml advanced`, then run a loadtest",
    )
    make_firecrest_launcher_config().add_to_parser(loadtest_advanced_parser)
    make_partition_config().add_to_parser(loadtest_advanced_parser)
    add_advanced_launch_arguments(loadtest_advanced_parser)
    _add_loadtest_arguments(loadtest_advanced_parser)

    loadtest_run_parser = loadtest_subparsers.add_parser(
        "run",
        help="Run a loadtest against an already launched model or external OpenAI-compatible URL",
    )
    make_firecrest_launcher_config().add_to_parser(loadtest_run_parser)
    make_partition_config().add_to_parser(loadtest_run_parser)
    make_reservation_config().add_to_parser(loadtest_run_parser)
    loadtest_run_parser.add_argument(
        "--loadtest-model",
        dest="loadtest_model",
        default=None,
        help="Model name to health-check and send in OpenAI-compatible requests.",
    )
    _add_loadtest_arguments(loadtest_run_parser, include_cancel=False)


def _loadtest_results_dir(run_id: int | str) -> Path:
    config_dir = Path(os.environ.get("SML_CONFIG_DIR", str(Path.home() / ".sml")))
    return config_dir / "loadtest" / str(run_id)


def make_loadtest_config_from_values(
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


def make_loadtest_config(args: argparse.Namespace) -> LoadtestConfig:
    return make_loadtest_config_from_values(
        scenario=args.loadtest_scenario or _DEFAULT_LOADTEST_SCENARIO,
        max_tokens=args.loadtest_max_tokens,
        ignore_eos=args.loadtest_ignore_eos,
        prompt_seed=args.loadtest_prompt_seed,
    )


async def _prompt_loadtest_scenario(args: argparse.Namespace) -> None:
    if args.loadtest_scenario:
        return
    scenarios = load_scenarios()
    answer = await questionary.select(
        "Choose the loadtest scenario.",
        choices=[questionary.Choice(title=scenario.name, value=scenario.name) for scenario in scenarios],
    ).ask_async()
    args.loadtest_scenario = answer or _DEFAULT_LOADTEST_SCENARIO


def _make_loadtest_server(args: argparse.Namespace, api_key: str, model_name: str) -> ServerConfig:
    return ServerConfig(
        url=args.loadtest_server_url.rstrip("/"),
        api_key=api_key,
        model=model_name,
        is_swissai=True,
    )


def make_cluster_loadtest_config(
    args: argparse.Namespace,
    *,
    reservation: str | None = None,
) -> ClusterLoadtestConfig:
    if getattr(args, "cancel_after_loadtest", False) and not args.wait_for_loadtest:
        raise ValueError("--cancel-after-loadtest requires --wait-for-loadtest.")
    if args.loadtest_ready_timeout <= 0:
        raise ValueError("--loadtest-ready-timeout must be greater than 0.")
    if not re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", args.loadtest_job_time):
        raise ValueError("--loadtest-job-time must be in HH:MM:SS format.")
    return ClusterLoadtestConfig(
        container_image=str(DEFAULT_CLUSTER_CONTAINER_IMAGE),
        time=args.loadtest_job_time,
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
    reservation: str | None = None,
) -> None:
    if not K6_SCRIPT.exists():
        raise FileNotFoundError(f"k6 script not found: {K6_SCRIPT}")

    prompts_file = resolve_prompts_file(args.loadtest_prompts_file)

    cluster_config = make_cluster_loadtest_config(args, reservation=reservation)
    print(f"Loadtest container image: {cluster_config.container_image}")
    print(f"Loadtest prompts file: {prompts_file}")
    if cluster_config.metrics_remote_write_url:
        print(f"Loadtest metrics remote write: {cluster_config.metrics_remote_write_url}")
    submission = await submit_cluster_loadtest(
        launcher=launcher,
        server=server,
        bench=loadtest_config,
        k6_script=K6_SCRIPT,
        prompts_file=prompts_file,
        summary_path=summary_path,
        cluster=cluster_config,
    )
    print(f"Cluster loadtest job submitted: {submission.job_id}")
    print(f"Loadtest run label: {submission.run_label}")
    if cluster_config.wait:
        print(f"Loadtest summary: {summary_path}")


async def _wait_until_model_healthy(
    served_model_name: str,
    api_key: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    loop = asyncio.get_running_loop()
    started = loop.time()
    last_reported_health: ModelHealth | None = None
    next_progress_at = started + _LOADTEST_READY_PROGRESS_SECONDS
    print(f"Waiting for model health: {served_model_name}")

    while True:
        health = await check_model_health(served_model_name, api_key)
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
    loadtest_reservation: str | None = None,
) -> None:
    try:
        if args.wait_until_healthy:
            await _wait_until_model_healthy(
                served_model_name,
                cscs_api_key,
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
    loadtest_reservation: str | None = None,
) -> None:
    make_cluster_loadtest_config(args, reservation=loadtest_reservation)
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
        loadtest_reservation=loadtest_reservation,
    )


async def _run_loadtest_against_existing_model(
    args: argparse.Namespace,
    *,
    create_launcher: _CreateLauncher,
) -> None:
    if not InitConfig.exists():
        raise ValueError("Run `sml init` first so SML can submit the cluster loadtest job.")

    config = InitConfig.load()
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    loadtest_model = args.loadtest_model
    request_model = loadtest_model or ""
    loadtest_config = make_loadtest_config(args)
    launcher = await create_launcher(config, args, True)

    if args.wait_until_healthy and loadtest_model:
        await _wait_until_model_healthy(
            loadtest_model,
            cscs_api_key,
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


async def _run_loadtest_preconfigured(
    args: argparse.Namespace,
    *,
    create_launcher: _CreateLauncher,
    get_launch_request: _GetLaunchRequest,
) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await create_launcher(config, args)
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_request = await get_launch_request(launcher, args)
    await _prompt_loadtest_scenario(args)
    loadtest_config = make_loadtest_config(args)

    await _submit_and_run_loadtest(
        launcher=launcher,
        launch_coro=launcher.launch_model(launch_request),
        cscs_api_key=cscs_api_key,
        args=args,
        loadtest_config=loadtest_config,
    )


async def _run_loadtest_advanced(
    args: argparse.Namespace,
    *,
    create_launcher: _CreateLauncher,
    build_launch_args_from_advanced: _BuildLaunchArgsFromAdvanced,
) -> None:
    if not InitConfig.exists():
        print("SML is not configured. Run `sml init` first.")
        return

    config = InitConfig.load()
    launcher = await create_launcher(config, args, True)
    cscs_api_key = config.get_non_none_value("cscs_api_key")
    launch_args = build_launch_args_from_advanced(
        args,
        account=launcher.account,
        partition=launcher.partition,
        telemetry_endpoint=config.get_value("telemetry_endpoint"),
    )
    loadtest_config = make_loadtest_config(args)

    await _submit_and_run_loadtest(
        launcher=launcher,
        launch_coro=launcher.launch_with_args(launch_args),
        cscs_api_key=cscs_api_key,
        args=args,
        loadtest_config=loadtest_config,
    )


async def run_loadtest_command(
    args: argparse.Namespace,
    *,
    create_launcher: _CreateLauncher,
    get_launch_request: _GetLaunchRequest,
    build_launch_args_from_advanced: _BuildLaunchArgsFromAdvanced,
) -> None:
    if args.loadtest_command == "preconfigured":
        await _run_loadtest_preconfigured(
            args,
            create_launcher=create_launcher,
            get_launch_request=get_launch_request,
        )
    elif args.loadtest_command == "advanced":
        await _run_loadtest_advanced(
            args,
            create_launcher=create_launcher,
            build_launch_args_from_advanced=build_launch_args_from_advanced,
        )
    elif args.loadtest_command == "run":
        await _run_loadtest_against_existing_model(args, create_launcher=create_launcher)
