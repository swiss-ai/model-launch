from typing import Any

import pytest
from pydantic import ValidationError

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs


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


def test_metrics_require_remote_write_url() -> None:
    with pytest.raises(
        ValidationError,
        match="Metrics require a remote write URL",
    ):
        _make_args(metrics_remote_write_url="")


def test_disabled_metrics_allow_no_remote_write_url() -> None:
    _make_args(
        disable_metrics=True,
        metrics_remote_write_url="",
        disable_dcgm_exporter=True,
    )
