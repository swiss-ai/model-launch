from swiss_ai_model_launch.launchers.framework import render_rank_scripts
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology


def _make_args(**overrides) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="vendor/model-abc",
        account="proj01",
        partition="normal",
        environment="/path/to/env.toml",
        framework="sglang",
        framework_args="--model /path/to/model",
    )
    return LaunchArgs(**{**defaults, **overrides})


def test_sglang_singular_has_only_head():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=1))
    scripts = render_rank_scripts(args)
    assert set(scripts) == {"head.sh"}


def test_sglang_multi_node_has_head_and_follower():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=4))
    scripts = render_rank_scripts(args)
    assert set(scripts) == {"head.sh", "follower.sh"}


def test_router_only_when_multi_replica_and_sgl():
    # No router when single replica
    args = _make_args(router="SGL", topology=Topology(replicas=1, nodes_per_replica=4))
    assert "router.sh" not in render_rank_scripts(args)
    # No router when multi-replica but router="OPENTELA"
    args = _make_args(router="OPENTELA", topology=Topology(replicas=2, nodes_per_replica=4))
    assert "router.sh" not in render_rank_scripts(args)
    # Router when both
    args = _make_args(router="SGL", topology=Topology(replicas=2, nodes_per_replica=4))
    assert "router.sh" in render_rank_scripts(args)
