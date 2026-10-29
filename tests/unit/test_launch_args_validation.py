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


def test_shadow_of_defaults_to_none() -> None:
    assert _make_args().shadow_of is None


def test_shadow_of_accepts_three_segment_model_id() -> None:
    args = _make_args(shadow_of="SwissAI-Research/swiss-ai/Apertus-8B-Instruct-2509")
    assert args.shadow_of == "SwissAI-Research/swiss-ai/Apertus-8B-Instruct-2509"


@pytest.mark.parametrize("shadow_of", ["", "vendor/model", "a/b/c/d", "a//c", "/b/c"])
def test_shadow_of_rejects_malformed_model_id(shadow_of: str) -> None:
    with pytest.raises(ValidationError, match="shadow_of must be a model id"):
        _make_args(shadow_of=shadow_of)


def test_shadow_of_rejects_disabled_opentela() -> None:
    with pytest.raises(ValidationError, match="cannot be combined with disable_opentela"):
        _make_args(shadow_of="alice/vendor/model", disable_opentela=True)
