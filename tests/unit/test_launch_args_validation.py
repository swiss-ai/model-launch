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


def test_authorization_defaults_to_empty() -> None:
    # Empty means no label is emitted — the pre-feature (public) behavior.
    assert _make_args().authorization == ""


@pytest.mark.parametrize(
    "value",
    ["", "public", "user@epfl.ch", "user1@epfl.ch,user2@ethz.ch"],
)
def test_authorization_accepts_resolved_values(value: str) -> None:
    assert _make_args(authorization=value).authorization == value


def test_authorization_rejects_unresolved_private() -> None:
    # "private" reaching LaunchArgs means resolve_authorization was skipped.
    with pytest.raises(ValidationError, match="resolved"):
        _make_args(authorization="private")


@pytest.mark.parametrize(
    "value",
    ["nonsense", "user@", "a@b.ch,nope", "a@b.ch,,c@d.ch"],
)
def test_authorization_rejects_non_grammar_values(value: str) -> None:
    with pytest.raises(ValidationError, match="authorization"):
        _make_args(authorization=value)
