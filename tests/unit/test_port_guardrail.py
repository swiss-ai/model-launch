import warnings

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs


def _make_args(**overrides) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="vendor/model-abc1",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="sglang",
    )
    return LaunchArgs(**{**defaults, **overrides})


def test_redundant_port_in_framework_args_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_args(framework_args="--port 9000 --tp 4")
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("--port" in m and "redundant" in m for m in msgs)
