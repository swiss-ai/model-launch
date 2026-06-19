from typing import Any

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.utils import render_sbatch_header


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


def test_render_sbatch_header_structure() -> None:
    args = _make_args(time="03:00:00")
    header = render_sbatch_header(args)
    assert header.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=test_job" in header
    assert "#SBATCH --account=proj01" in header
    assert "#SBATCH --time=03:00:00" in header
    assert "#SBATCH --exclusive" in header
    assert "#SBATCH --partition=normal" in header
    assert "#SBATCH --output=logs/%j/log.out" in header
    assert "#SBATCH --error=logs/%j/log.err" in header


def test_render_sbatch_header_with_reservation() -> None:
    args = _make_args()
    assert "#SBATCH --reservation=my-reservation" in render_sbatch_header(args, reservation="my-reservation")


def test_render_sbatch_header_without_reservation() -> None:
    header = render_sbatch_header(_make_args())
    assert "--reservation" not in header
