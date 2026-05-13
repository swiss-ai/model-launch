import warnings
from pathlib import Path

from swiss_ai_model_launch.launchers.launch_args import LaunchArgs, Topology
from swiss_ai_model_launch.launchers.utils import render_sbatch_header, resolve_model_path


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


# ── to_job_env ───────────────────────────────────────────────────────────────


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


# ── to_sbatch_args ───────────────────────────────────────────────────────────


def test_to_sbatch_args_contains_required():
    args = _make_args(time="02:00:00", topology=Topology(replicas=2, nodes_per_replica=2))
    sbatch = args.to_sbatch_args()
    assert "--job-name=test_job" in sbatch
    assert "--account=proj01" in sbatch
    assert "--time=02:00:00" in sbatch
    assert "--exclusive" in sbatch
    assert "--nodes=4" in sbatch
    assert "--partition=normal" in sbatch
    assert "--output=logs/%j/log.out" in sbatch
    assert "--error=logs/%j/log.out" in sbatch


def test_to_sbatch_args_nodes_auto_computed():
    args = _make_args(topology=Topology(replicas=3, nodes_per_replica=4))
    assert "--nodes=12" in args.to_sbatch_args()


def test_to_sbatch_args_reservation_included():
    args = _make_args(reservation="my-reservation")
    assert "--reservation=my-reservation" in args.to_sbatch_args()


def test_to_sbatch_args_no_reservation():
    sbatch = _make_args().to_sbatch_args()
    assert not any(a.startswith("--reservation") for a in sbatch)


# ── render_sbatch_header ──────────────────────────────────────────────────────


def test_render_sbatch_header_structure():
    args = _make_args(time="03:00:00")
    header = render_sbatch_header(args)
    assert header.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=test_job" in header
    assert "#SBATCH --account=proj01" in header
    assert "#SBATCH --time=03:00:00" in header
    assert "#SBATCH --exclusive" in header
    assert "#SBATCH --partition=normal" in header
    assert "#SBATCH --output=logs/%j/log.out" in header
    assert "#SBATCH --error=logs/%j/log.out" in header


def test_render_sbatch_header_with_reservation():
    args = _make_args(reservation="my-reservation")
    assert "#SBATCH --reservation=my-reservation" in render_sbatch_header(args)


def test_render_sbatch_header_without_reservation():
    header = render_sbatch_header(_make_args())
    assert "--reservation" not in header


# ── resolve_model_path ────────────────────────────────────────────────────────


def test_resolve_model_path_uses_registry():
    registry = Path("/store/models")
    assert resolve_model_path("vendor/name", registry) == "/store/models/vendor/name"


def test_resolve_model_path_explicit_path_takes_precedence():
    registry = Path("/store/models")
    assert resolve_model_path("vendor/name", registry, "/custom/path") == "/custom/path"


# ── legacy field migration ────────────────────────────────────────────────────


def test_legacy_workers_migrates_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        args = _make_args(workers=4, nodes_per_worker=2)
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("workers" in m for m in msgs)
    assert any("nodes_per_worker" in m for m in msgs)
    assert args.topology.replicas == 4
    assert args.topology.nodes_per_replica == 2


def test_legacy_worker_port_ignored_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_args(worker_port=9999)
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("worker_port" in m and "hardcoded" in m for m in msgs)


def test_redundant_port_in_framework_args_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_args(framework_args="--port 9000 --tp 4")
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("--port" in m and "redundant" in m for m in msgs)


def test_legacy_and_new_keys_prefer_new():
    # If both are passed, the explicit topology wins; legacy is ignored after warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        args = _make_args(workers=4, topology=Topology(replicas=2))
    assert args.topology.replicas == 2


# ── deprecated kwarg interaction ──────────────────────────────────────────────


def test_passing_only_topology_emits_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_args(topology=Topology(replicas=2, nodes_per_replica=2))
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


# ── pydantic field validation ─────────────────────────────────────────────────


def test_topology_defaults():
    t = Topology()
    assert t.replicas == 1
    assert t.nodes_per_replica == 1


def test_launch_args_default_topology():
    args = _make_args()
    assert args.topology.replicas == 1
    assert args.topology.nodes_per_replica == 1


def test_nodes_derived_from_topology():
    args = _make_args(topology=Topology(replicas=2, nodes_per_replica=3))
    assert args.total_nodes == 6


def test_legacy_nodes_kwarg_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_args(nodes=99, topology=Topology(replicas=2, nodes_per_replica=3))
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("nodes" in m and "derived" in m for m in msgs)
