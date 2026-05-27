from typing import Any

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology


def _make_args(**overrides: Any) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="vendor/model-abc1",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="sglang",
    )
    return LaunchArgs(**{**defaults, **overrides})


def test_to_sbatch_args_contains_required() -> None:
    args = _make_args(time="02:00:00", topology=Topology(replicas=2, nodes_per_replica=2))
    sbatch = args.to_sbatch_args()
    assert "--job-name=test_job" in sbatch
    assert "--account=proj01" in sbatch
    assert "--time=02:00:00" in sbatch
    assert "--exclusive" in sbatch
    assert "--nodes=4" in sbatch
    assert "--partition=normal" in sbatch
    assert "--output=logs/%j/log.out" in sbatch
    assert "--error=logs/%j/log.err" in sbatch


def test_to_sbatch_args_nodes_auto_computed() -> None:
    args = _make_args(topology=Topology(replicas=3, nodes_per_replica=4))
    assert "--nodes=12" in args.to_sbatch_args()


def test_to_sbatch_args_reservation_included() -> None:
    args = _make_args()
    assert "--reservation=my-reservation" in args.to_sbatch_args(reservation="my-reservation")


def test_to_sbatch_args_no_reservation() -> None:
    sbatch = _make_args().to_sbatch_args()
    assert not any(a.startswith("--reservation") for a in sbatch)
