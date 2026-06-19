import pytest

import swiss_ai_model_launch.cli.loadtest as loadtest_module
from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth
from swiss_ai_model_launch.cli.loadtest import (
    _make_loadtest_server,
    _run_k6_on_cluster,
    _run_loadtest_for_submitted_job,
    _wait_until_model_healthy,
    run_loadtest_command,
)
from swiss_ai_model_launch.cli.main import _build_parser
from swiss_ai_model_launch.launchers import SlurmLauncher
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import Launcher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.loadtest.cluster import ClusterLoadtestSubmission
from swiss_ai_model_launch.loadtest.models import LoadtestConfig, ServerConfig


def test_make_loadtest_server_strips_trailing_slash() -> None:
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run", "--loadtest-server-url", "https://example.test/"])

    assert _make_loadtest_server(args, "secret", "test-model") == ServerConfig(
        url="https://example.test",
        api_key="secret",
        model="test-model",
        is_swissai=True,
    )


async def test_wait_until_model_healthy_returns_after_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[tuple[str, str]] = []

    async def fake_check_model_health(served_model_name: str, api_key: str) -> ModelHealth:
        checked.append((served_model_name, api_key))
        return ModelHealth.HEALTHY

    monkeypatch.setattr(loadtest_module, "check_model_health", fake_check_model_health)

    await _wait_until_model_healthy(
        "test-model",
        "secret",
        timeout_seconds=1,
        poll_interval_seconds=0,
    )

    assert checked == [("test-model", "secret")]


async def test_wait_until_model_healthy_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_check_model_health(served_model_name: str, api_key: str) -> ModelHealth:
        return ModelHealth.NOT_RESPONDING

    monkeypatch.setattr(loadtest_module, "check_model_health", fake_check_model_health)

    with pytest.raises(TimeoutError, match="Timed out after 0s"):
        await _wait_until_model_healthy(
            "test-model",
            "secret",
            timeout_seconds=0,
            poll_interval_seconds=0,
        )


async def test_run_k6_on_cluster_submits_resolved_config(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("{}\n")

    async def fake_submit_cluster_loadtest(**kwargs: object) -> ClusterLoadtestSubmission:
        calls.append(kwargs)
        return ClusterLoadtestSubmission(job_id=123, run_label="loadtest_run")

    monkeypatch.setattr(loadtest_module, "K6_SCRIPT", tmp_path / "script.js")
    (tmp_path / "script.js").write_text("export default function() {}\n")
    monkeypatch.setattr(loadtest_module, "submit_cluster_loadtest", fake_submit_cluster_loadtest)

    parser = _build_parser()
    args = parser.parse_args(
        [
            "loadtest",
            "run",
            "--loadtest-prompts-file",
            str(prompts_file),
            "--no-wait-for-loadtest",
            "--no-loadtest-metrics-remote-write",
        ]
    )
    launcher = SlurmLauncher(
        system_name="test-system",
        username="test-user",
        account="test-account",
        partition="test-partition",
    )

    await _run_k6_on_cluster(
        launcher=launcher,
        server=ServerConfig(url="https://example.test", api_key="secret", model="test-model", is_swissai=True),
        loadtest_config=LoadtestConfig(scenario="throughput", think_time="0", max_tokens="16"),
        summary_path=tmp_path / "summary.json",
        args=args,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["launcher"] is launcher
    assert call["k6_script"] == tmp_path / "script.js"
    assert call["prompts_file"] == prompts_file
    assert call["cluster"].wait is False


async def test_run_k6_on_cluster_requires_packaged_script(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loadtest_module, "K6_SCRIPT", tmp_path / "missing.js")
    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    with pytest.raises(FileNotFoundError, match="k6 script not found"):
        await _run_k6_on_cluster(
            launcher=SlurmLauncher(
                system_name="test-system",
                username="test-user",
                account="test-account",
                partition="test-partition",
            ),
            server=ServerConfig(url="https://example.test", api_key="secret", model="test-model", is_swissai=True),
            loadtest_config=LoadtestConfig(scenario="throughput", think_time="0", max_tokens="16"),
            summary_path=tmp_path / "summary.json",
            args=args,
        )


class DummyConfig:
    def get_non_none_value(self, name: str) -> str:
        return {"cscs_api_key": "secret"}[name]


class DummyLauncher(Launcher):
    def __init__(self) -> None:
        super().__init__(
            system_name="test-system",
            username="test-user",
            account="test-account",
            partition="test-partition",
        )
        self.cancelled_jobs: list[int] = []

    async def get_preconfigured_models(self) -> list[ModelCatalogEntry]:
        return []

    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]:
        return 123, "served-model"

    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]:
        return 123, "served-model"

    async def get_job_status(self, job_id: int) -> JobStatus:
        return JobStatus.RUNNING

    async def get_job_logs(self, job_id: int) -> tuple[str, str]:
        return "", ""

    async def cancel_job(self, job_id: int) -> None:
        self.cancelled_jobs.append(job_id)

    def get_log_dir(self, job_id: int) -> str:
        return f"/logs/{job_id}"

    def get_tail_hint(self, job_id: int) -> str:
        return f"tail -f /logs/{job_id}/log.out"

    async def read_job_file(self, job_id: int, filename: str) -> str | None:
        return None


async def test_run_loadtest_for_submitted_job_cancels_after_loadtest(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def fake_run_k6_on_cluster(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setenv("SML_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(loadtest_module, "_run_k6_on_cluster", fake_run_k6_on_cluster)

    parser = _build_parser()
    args = parser.parse_args(
        [
            "loadtest",
            "advanced",
            "--serving-framework",
            "sglang",
            "--slurm-environment",
            "/path/to/env.toml",
            "--no-wait-until-healthy",
            "--cancel-after-loadtest",
        ]
    )
    launcher = DummyLauncher()

    await _run_loadtest_for_submitted_job(
        launcher=launcher,
        job_id=123,
        served_model_name="served-model",
        cscs_api_key="secret",
        args=args,
        loadtest_config=LoadtestConfig(scenario="throughput", think_time="0", max_tokens="16"),
        loadtest_reservation="reservation",
    )

    assert launcher.cancelled_jobs == [123]
    assert len(calls) == 1
    assert calls[0]["launcher"] is launcher
    assert calls[0]["server"] == ServerConfig(
        url="https://api.swissai.svc.cscs.ch",
        api_key="secret",
        model="served-model",
        is_swissai=True,
    )
    assert calls[0]["summary_path"].parent == tmp_path / "config" / "loadtest" / "123"
    assert calls[0]["reservation"] == "reservation"


async def test_run_loadtest_command_run_path_skips_health_without_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    launcher = DummyLauncher()

    async def create_launcher(config: DummyConfig, args: object, non_interactive: bool = False) -> DummyLauncher:
        assert isinstance(config, DummyConfig)
        assert non_interactive is True
        return launcher

    async def fake_run_k6_on_cluster(**kwargs: object) -> None:
        calls.append(kwargs)

    def fail_health(*args: object, **kwargs: object) -> None:
        raise AssertionError("health check should be skipped without --loadtest-model")

    monkeypatch.setenv("SML_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(loadtest_module.InitConfig, "exists", classmethod(lambda cls: True))
    monkeypatch.setattr(loadtest_module.InitConfig, "load", classmethod(lambda cls: DummyConfig()))
    monkeypatch.setattr(loadtest_module, "_run_k6_on_cluster", fake_run_k6_on_cluster)
    monkeypatch.setattr(loadtest_module, "_wait_until_model_healthy", fail_health)

    parser = _build_parser()
    args = parser.parse_args(["loadtest", "run"])

    await run_loadtest_command(
        args,
        create_launcher=create_launcher,
        get_launch_request=lambda launcher, args=None: pytest.fail("run should not build a launch request"),
        build_launch_args_from_advanced=lambda args, **kwargs: pytest.fail("run should not build launch args"),
    )

    assert len(calls) == 1
    assert calls[0]["launcher"] is launcher
    assert calls[0]["server"] == ServerConfig(
        url="https://api.swissai.svc.cscs.ch",
        api_key="secret",
        model="",
        is_swissai=True,
    )
    assert calls[0]["summary_path"].parent == tmp_path / "config" / "loadtest" / "external"
