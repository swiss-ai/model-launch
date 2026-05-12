import shlex
from pathlib import Path

import pytest

from swiss_ai_model_launch.loadtest.cluster import (
    ClusterLoadtestConfig,
    _container_mounts_for_external_prompts,
    build_cluster_loadtest_script,
)
from swiss_ai_model_launch.loadtest.models import LoadtestConfig


@pytest.fixture
def bench() -> LoadtestConfig:
    return LoadtestConfig(scenario="throughput", think_time="0", max_tokens="512")


@pytest.fixture
def cluster() -> ClusterLoadtestConfig:
    return ClusterLoadtestConfig(container_image="/images/k6.sqsh", cpus_per_task=4)


def _make_script(bench: LoadtestConfig, cluster: ClusterLoadtestConfig, **kwargs: object) -> str:
    values = {
        "bench": bench,
        "cluster": cluster,
        "account": "testaccount",
        "partition": "testpartition",
        "reservation": None,
        "run_label": "loadtest_throughput_20260101_000000_XXXXXX",
        "prompts_path": "/capstor/prompts.jsonl",
        "container_mounts": "/work:/work,/capstor:/capstor",
        "model": "test-model",
    }
    values.update(kwargs)
    return build_cluster_loadtest_script(**values)


def test_script_sbatch_directives(bench: LoadtestConfig, cluster: ClusterLoadtestConfig) -> None:
    script = _make_script(bench, cluster)
    assert "#SBATCH --account=testaccount" in script
    assert "#SBATCH --partition=testpartition" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=4" in script


def test_script_no_reservation_by_default(bench: LoadtestConfig, cluster: ClusterLoadtestConfig) -> None:
    script = _make_script(bench, cluster)
    assert "--reservation" not in script


def test_script_reservation_line_included(bench: LoadtestConfig, cluster: ClusterLoadtestConfig) -> None:
    script = build_cluster_loadtest_script(
        bench=bench,
        cluster=cluster,
        account="testaccount",
        partition="testpartition",
        reservation="my_reservation",
        run_label="loadtest_throughput_20260101_000000_XXXXXX",
        prompts_path="/capstor/prompts.jsonl",
        container_mounts="/work:/work",
        model="test-model",
    )
    assert "#SBATCH --reservation=my_reservation" in script


def test_script_container_image_in_srun(bench: LoadtestConfig, cluster: ClusterLoadtestConfig) -> None:
    script = _make_script(bench, cluster)
    assert '--container-image="/images/k6.sqsh"' in script


def test_script_container_workdir(bench: LoadtestConfig, cluster: ClusterLoadtestConfig) -> None:
    script = _make_script(bench, cluster)
    assert "--container-workdir=/work" in script


def test_script_k6_run_command(bench: LoadtestConfig, cluster: ClusterLoadtestConfig) -> None:
    script = _make_script(bench, cluster)
    assert "k6 run" in script
    assert "--summary-export /work/summary.json" in script
    assert "/work/script.js" in script


def test_script_k6_remote_write_disabled_by_default(bench: LoadtestConfig, cluster: ClusterLoadtestConfig) -> None:
    script = _make_script(bench, cluster)
    assert "experimental-prometheus-rw" not in script
    assert "K6_PROMETHEUS_RW_SERVER_URL" not in script


def test_script_k6_remote_write_enabled(bench: LoadtestConfig) -> None:
    cluster = ClusterLoadtestConfig(
        container_image="/images/k6.sqsh",
        cpus_per_task=4,
        metrics_remote_write_url="https://prometheus.example/api/v1/write",
    )
    script = _make_script(bench, cluster)

    assert "-o experimental-prometheus-rw" in script
    assert "K6_PROMETHEUS_RW_SERVER_URL=https://prometheus.example/api/v1/write" in script
    assert "K6_PROMETHEUS_RW_TREND_STATS=" in script
    assert "min,avg,med,p(90),p(95),p(99),max" in script
    assert "K6_PROMETHEUS_RW_TREND_AS_NATIVE_HISTOGRAM" not in script
    assert "--tag scenario=throughput" in script
    assert "--tag model=test-model" in script


def test_script_k6_run_command_shell_quotes_dynamic_values(
    bench: LoadtestConfig,
    cluster: ClusterLoadtestConfig,
) -> None:
    script = _make_script(
        bench,
        cluster,
        run_label="run-$USER's-label",
        prompts_path="/capstor/prompt files/$USER/prompts.json",
        model="model-$USER's-name",
    )

    command_line = next(line.strip() for line in script.splitlines() if line.strip().startswith("sh -lc "))
    inner_command = shlex.split(command_line)[2]
    tokens = shlex.split(inner_command)

    assert tokens[tokens.index("--tag") + 1] == "scenario=throughput"
    assert f"run_label=run-$USER's-label" in tokens
    assert f"model=model-$USER's-name" in tokens
    assert f"PROMPTS_FILE=/capstor/prompt files/$USER/prompts.json" in tokens
    assert 'sh -lc "' not in script


def test_container_mounts_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        _container_mounts_for_external_prompts("my_run", Path("relative/prompts.jsonl"))


def test_container_mounts_includes_work_mount() -> None:
    _, mounts = _container_mounts_for_external_prompts("my_run", Path("/capstor/prompts.jsonl"))
    assert "${PWD}/my_run:/work" in mounts


def test_container_mounts_includes_top_level_dir() -> None:
    _, mounts = _container_mounts_for_external_prompts("my_run", Path("/capstor/prompts.jsonl"))
    assert "/capstor:/capstor" in mounts


def test_container_mounts_prompts_path_returned() -> None:
    prompts_path, _ = _container_mounts_for_external_prompts("my_run", Path("/scratch/data/prompts.jsonl"))
    assert prompts_path == "/scratch/data/prompts.jsonl"
