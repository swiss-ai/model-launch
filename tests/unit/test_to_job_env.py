from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology


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


def test_to_job_env_keys():
    expected = {
        "FRAMEWORK",
        "SML_ENVIRONMENT",
        "FRAMEWORK_ARGS",
        "PRE_LAUNCH_CMDS",
        "REPLICAS",
        "NODES_PER_REPLICA",
        "USE_ROUTER",
        "ROUTER_ENVIRONMENT",
        "ROUTER_ARGS",
        "USE_OCF",
        "SERVED_MODEL_NAME",
        "METRICS_REMOTE_WRITE_URL",
        "METRICS_AGENT_BIN",
        "TELEMETRY_ENDPOINT",
        "SML_TIME",
    }
    assert set(_make_args().to_job_env().keys()) == expected


def test_to_job_env_values():
    args = _make_args(
        framework="vllm",
        environment="/envs/vllm.toml",
        framework_args="--tp 4",
        topology=Topology(replicas=2, nodes_per_replica=2),
        time="01:00:00",
        served_model_name="vendor/model-xyz",
        telemetry_endpoint="http://telemetry.example.com",
    )
    env = args.to_job_env()
    assert env["FRAMEWORK"] == "vllm"
    assert env["SML_ENVIRONMENT"] == "/envs/vllm.toml"
    assert env["ROUTER_ENVIRONMENT"] == "/envs/vllm.toml"
    assert env["FRAMEWORK_ARGS"] == "--port 8080 --tp 4"
    assert env["REPLICAS"] == "2"
    assert env["NODES_PER_REPLICA"] == "2"
    assert env["SML_TIME"] == "01:00:00"
    assert env["SERVED_MODEL_NAME"] == "vendor/model-xyz"
    assert env["TELEMETRY_ENDPOINT"] == "http://telemetry.example.com"


def test_to_job_env_injects_port_with_no_user_args():
    env = _make_args().to_job_env()
    assert env["FRAMEWORK_ARGS"] == "--port 8080"


def test_to_job_env_use_router():
    assert _make_args(use_router=False).to_job_env()["USE_ROUTER"] == "false"
    assert _make_args(use_router=True).to_job_env()["USE_ROUTER"] == "true"


def test_to_job_env_use_ocf():
    assert _make_args(disable_ocf=False).to_job_env()["USE_OCF"] == "true"
    assert _make_args(disable_ocf=True).to_job_env()["USE_OCF"] == "false"


def test_to_job_env_telemetry_endpoint_none():
    assert _make_args(telemetry_endpoint=None).to_job_env()["TELEMETRY_ENDPOINT"] == ""


def test_to_job_env_all_values_are_strings():
    for k, v in _make_args().to_job_env().items():
        assert isinstance(v, str), f"{k} is not a string"
