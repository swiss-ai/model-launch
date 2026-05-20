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
