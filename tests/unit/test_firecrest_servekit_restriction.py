import pytest

from swiss_ai_model_launch.launchers.firecrest_launcher import FirecRESTLauncher
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.launch_request import LaunchRequest


def _make_launcher() -> FirecRESTLauncher:
    return FirecRESTLauncher(client=object(), system_name="clariden", username="u", account="a", partition="p")


def _make_launch_args(**overrides) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="vendor/model-abc1",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="sglang",
    )
    return LaunchArgs(**{**defaults, **overrides})


def _make_launch_request(**overrides) -> LaunchRequest:
    defaults = dict(
        model="vendor/model",
        framework="sglang",
        nodes_per_replica=1,
        replicas=1,
        time="02:00:00",
    )
    return LaunchRequest(**{**defaults, **overrides})


def test_firecrest_get_launch_args_rejects_servekit_optims() -> None:
    launcher = _make_launcher()
    with pytest.raises(ValueError, match="only supported with the direct SLURM launcher"):
        launcher._get_launch_args_from_request(_make_launch_request(servekit_optims=True))


async def test_firecrest_prepare_launch_args_rejects_servekit_optims() -> None:
    launcher = _make_launcher()
    with pytest.raises(ValueError, match="only supported with the direct SLURM launcher"):
        await launcher._prepare_launch_args(_make_launch_args(servekit_optims=True))
