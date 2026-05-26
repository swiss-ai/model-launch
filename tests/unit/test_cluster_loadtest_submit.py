from pathlib import Path

import pytest

import swiss_ai_model_launch.loadtest.cluster as cluster_module
from swiss_ai_model_launch.launchers import SlurmLauncher
from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.job_status import JobStatus
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.launcher import Launcher
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry
from swiss_ai_model_launch.loadtest.cluster import (
    ClusterLoadtestConfig,
    _wait_for_firecrest_job,
    _wait_for_local_slurm_job,
    _write_local_run_files,
    submit_cluster_loadtest,
)
from swiss_ai_model_launch.loadtest.models import LoadtestConfig, ServerConfig


@pytest.fixture
def bench() -> LoadtestConfig:
    return LoadtestConfig(scenario="throughput", think_time="0", max_tokens="512")


def test_write_local_run_files_copies_script_and_config(tmp_path: Path, bench: LoadtestConfig) -> None:
    k6_script = tmp_path / "source-script.js"
    k6_script.write_text("export default function() {}\n")
    run_dir = tmp_path / "run"

    _write_local_run_files(
        run_dir=run_dir,
        server=ServerConfig(url="https://example.test", api_key="secret", model="test-model", is_swissai=True),
        bench=bench,
        k6_script=k6_script,
    )

    assert (run_dir / "script.js").read_text() == "export default function() {}\n"
    assert '"server_url":"https://example.test"' in (run_dir / "run_config.json").read_text()
    assert '"scenario":"throughput"' in (run_dir / "run_config.json").read_text()


async def test_wait_for_local_slurm_job_returns_on_completed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["RUNNING\n", "", "COMPLETED|0:0\n"])

    async def fake_run_checked(*cmd: str) -> str:
        return next(outputs)

    async def fake_sleep(seconds: int) -> None:
        assert seconds == 10

    monkeypatch.setattr(cluster_module, "_run_checked", fake_run_checked)
    monkeypatch.setattr(cluster_module.asyncio, "sleep", fake_sleep)

    await _wait_for_local_slurm_job(123, log_path=Path("/logs/123/loadtest.out"))


async def test_wait_for_local_slurm_job_raises_on_failed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["", "FAILED|1:0\n"])

    async def fake_run_checked(*cmd: str) -> str:
        return next(outputs)

    monkeypatch.setattr(cluster_module, "_run_checked", fake_run_checked)

    with pytest.raises(RuntimeError, match="did not complete successfully"):
        await _wait_for_local_slurm_job(123, log_path=Path("/logs/123/loadtest.out"))


class FakeFirecrestClient:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.created_dirs: list[str] = []
        self.uploads: list[tuple[Path, str]] = []
        self.submitted_scripts: list[str] = []

    async def job_info(self, **kwargs: object) -> list[dict[str, dict[str, str]]]:
        state = self.states.pop(0)
        return [{"status": {"state": state}}]

    async def mkdir(self, **kwargs: object) -> None:
        self.created_dirs.append(str(kwargs["path"]))

    async def upload(self, **kwargs: object) -> None:
        self.uploads.append((Path(kwargs["local_file"]), str(kwargs["filename"])))

    async def submit(self, **kwargs: object) -> dict[str, str]:
        self.submitted_scripts.append(str(kwargs["script_str"]))
        return {"jobId": "456"}


async def test_wait_for_firecrest_job_returns_on_completed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeFirecrestClient()
    client.states = ["RUNNING foo", "COMPLETED"]
    launcher = FirecRESTLauncher(client, "test-system", "test-user", "test-account", "test-partition")

    async def fake_sleep(seconds: int) -> None:
        assert seconds == 10

    monkeypatch.setattr(cluster_module.asyncio, "sleep", fake_sleep)

    await _wait_for_firecrest_job(launcher, 123, log_path="/logs/123/loadtest.out")


async def test_wait_for_firecrest_job_raises_on_failed_state() -> None:
    client = FakeFirecrestClient()
    client.states = ["FAILED"]
    launcher = FirecRESTLauncher(client, "test-system", "test-user", "test-account", "test-partition")

    with pytest.raises(RuntimeError, match="ended with state FAILED"):
        await _wait_for_firecrest_job(launcher, 123, log_path="/logs/123/loadtest.out")


async def test_submit_cluster_loadtest_returns_job_id_and_run_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bench: LoadtestConfig,
) -> None:
    async def fake_run_checked(*cmd: str) -> str:
        assert cmd[0] == "sbatch"
        return "Submitted batch job 123\n"

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cluster_module, "_run_checked", fake_run_checked)
    monkeypatch.setattr(cluster_module, "create_salt", lambda length: "X" * length)

    k6_script = tmp_path / "script.js"
    k6_script.write_text("export default function() {}\n")
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("{}\n")

    submission = await submit_cluster_loadtest(
        launcher=SlurmLauncher(
            system_name="test-system",
            username="test-user",
            account="test-account",
            partition="test-partition",
        ),
        server=ServerConfig(url="https://example.test", api_key="secret", model="test-model", is_swissai=True),
        bench=bench,
        k6_script=k6_script,
        prompts_file=prompts_file,
        summary_path=tmp_path / "summary.json",
        cluster=ClusterLoadtestConfig(container_image="/images/k6.sqsh", wait=False),
    )

    assert submission.job_id == 123
    assert submission.run_label.startswith("loadtest_throughput_")
    assert submission.run_label.endswith("_XXXXXX")


async def test_submit_cluster_loadtest_waits_and_copies_local_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bench: LoadtestConfig,
) -> None:
    async def fake_run_checked(*cmd: str) -> str:
        return "Submitted batch job 123\n"

    async def fake_wait_for_local_slurm_job(job_id: int, *, log_path: Path, poll_seconds: int = 10) -> None:
        working_dir = log_path.parents[2]
        [run_dir] = [path for path in working_dir.glob("loadtest_*") if path.is_dir()]
        (run_dir / "summary.json").write_text('{"ok":true}\n')

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cluster_module, "_run_checked", fake_run_checked)
    monkeypatch.setattr(cluster_module, "_wait_for_local_slurm_job", fake_wait_for_local_slurm_job)
    monkeypatch.setattr(cluster_module, "create_salt", lambda length: "X" * length)

    k6_script = tmp_path / "script.js"
    k6_script.write_text("export default function() {}\n")
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("{}\n")
    summary_path = tmp_path / "summary" / "summary.json"

    submission = await submit_cluster_loadtest(
        launcher=SlurmLauncher(
            system_name="test-system",
            username="test-user",
            account="test-account",
            partition="test-partition",
        ),
        server=ServerConfig(url="https://example.test", api_key="secret", model="test-model", is_swissai=True),
        bench=bench,
        k6_script=k6_script,
        prompts_file=prompts_file,
        summary_path=summary_path,
        cluster=ClusterLoadtestConfig(container_image="/images/k6.sqsh", wait=True),
    )

    assert submission.job_id == 123
    assert summary_path.read_text() == '{"ok":true}\n'


async def test_submit_cluster_loadtest_firecrest_uploads_and_submits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bench: LoadtestConfig,
) -> None:
    client = FakeFirecrestClient()
    monkeypatch.setattr(cluster_module, "create_salt", lambda length: "X" * length)
    k6_script = tmp_path / "script.js"
    k6_script.write_text("export default function() {}\n")
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("{}\n")

    submission = await submit_cluster_loadtest(
        launcher=FirecRESTLauncher(
            client,
            "test-system",
            "test-user",
            "test-account",
            "test-partition",
            reservation="launcher-reservation",
        ),
        server=ServerConfig(url="https://example.test", api_key="secret", model="test-model", is_swissai=True),
        bench=bench,
        k6_script=k6_script,
        prompts_file=prompts_file,
        summary_path=tmp_path / "summary.json",
        cluster=ClusterLoadtestConfig(container_image="/images/k6.sqsh", wait=False),
    )

    assert submission.job_id == 456
    assert client.created_dirs == [f"/users/test-user/.sml/{submission.run_label}"]
    assert [filename for _, filename in client.uploads] == ["script.js", "run_config.json"]
    assert "#SBATCH --reservation=launcher-reservation" in client.submitted_scripts[0]


class UnsupportedLauncher(Launcher):
    async def get_preconfigured_models(self) -> list[ModelCatalogEntry]:
        return []

    async def launch_model(self, launch_request: LaunchRequest) -> tuple[int, str]:
        return 1, "model"

    async def launch_with_args(self, launch_args: LaunchArgs) -> tuple[int, str]:
        return 1, "model"

    async def get_job_status(self, job_id: int) -> JobStatus:
        return JobStatus.UNKNOWN

    async def get_job_logs(self, job_id: int) -> tuple[str, str]:
        return "", ""

    async def cancel_job(self, job_id: int) -> None:
        return None

    def get_log_dir(self, job_id: int) -> str:
        return ""


async def test_submit_cluster_loadtest_rejects_unsupported_launcher(
    tmp_path: Path,
    bench: LoadtestConfig,
) -> None:
    k6_script = tmp_path / "script.js"
    k6_script.write_text("export default function() {}\n")
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text("{}\n")

    with pytest.raises(TypeError, match="UnsupportedLauncher"):
        await submit_cluster_loadtest(
            launcher=UnsupportedLauncher("test-system", "test-user", "test-account", "test-partition"),
            server=ServerConfig(url="https://example.test", api_key="secret", model="test-model", is_swissai=True),
            bench=bench,
            k6_script=k6_script,
            prompts_file=prompts_file,
            summary_path=tmp_path / "summary.json",
            cluster=ClusterLoadtestConfig(container_image="/images/k6.sqsh", wait=False),
        )
