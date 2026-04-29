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
    return build_cluster_loadtest_script(
        bench=bench,
        cluster=cluster,
        account="testaccount",
        partition="testpartition",
        reservation=None,
        run_label="loadtest_throughput_20260101_000000_XXXXXX",
        prompts_path="/capstor/prompts.jsonl",
        container_mounts="/work:/work,/capstor:/capstor",
        **kwargs,
    )


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
