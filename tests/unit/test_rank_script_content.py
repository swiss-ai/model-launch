import pytest

from swiss_ai_model_launch.launchers.framework import (
    OPENTELA_BOOTSTRAP_ADDR,
    OPENTELA_BOOTSTRAP_ADDR_DEV,
    render_master,
    render_rank_scripts,
)
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


def test_sglang_head_singular_uses_entrypoint_directly():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=1))
    head = render_rank_scripts(args)["head.sh"]
    assert "python3 -m sglang.launch_server" in head
    assert "--dist-init-addr" not in head  # singular has no dist init


def test_sglang_multi_node_passes_node_rank():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=4))
    follower = render_rank_scripts(args)["follower.sh"]
    # Unquoted because the cmd is reused inside OpenTela's --subprocess "..."; the
    # surrounding outer quotes there make inner quotes redundant (shellcheck
    # SC2027). node_rank is a small int passed as $1 from master, safe.
    assert "--node-rank $node_rank" in follower
    assert "--nnodes 4" in follower


def test_vllm_multi_node_uses_ray_with_script_on_disk():
    args = _make_args(framework="vllm", topology=Topology(replicas=1, nodes_per_replica=4))
    head = render_rank_scripts(args)["head.sh"]
    # Single-quoted heredoc keeps $-constructs literal in the on-disk file.
    assert "<<'__SML_RAY_HEAD_EOF__'" in head
    assert "ray start --head" in head
    assert "bash $ray_head_script" in head or 'bash "$ray_head_script"' in head


def test_vllm_follower_joins_ray():
    args = _make_args(framework="vllm", topology=Topology(replicas=1, nodes_per_replica=4))
    follower = render_rank_scripts(args)["follower.sh"]
    assert "ray start --address=" in follower
    assert "vllm.entrypoints" not in follower  # follower doesn't run the API server


def test_opentela_disabled_omits_wrap():
    args = _make_args(disable_opentela=True, topology=Topology(replicas=1, nodes_per_replica=1))
    head = render_rank_scripts(args)["head.sh"]
    assert "$OPENTELA_BIN start" not in head
    assert "--bootstrap.static" not in head


def test_opentela_enabled_wraps_head_and_follower():
    args = _make_args(
        framework="sglang",
        disable_opentela=False,
        topology=Topology(replicas=1, nodes_per_replica=4),
    )
    scripts = render_rank_scripts(args)
    assert "$OPENTELA_BIN start" in scripts["head.sh"]
    assert "--service.name llm" in scripts["head.sh"]
    assert "$OPENTELA_BIN start" in scripts["follower.sh"]
    assert "--service.name" not in scripts["follower.sh"]


def test_opentela_labels_include_started_at_and_expires_at():
    args = _make_args(
        framework="sglang",
        disable_opentela=False,
        time="06:00:00",
        topology=Topology(replicas=1, nodes_per_replica=1),
    )
    head = render_rank_scripts(args)["head.sh"]
    assert "--label started_at=$(date -u +%FT%TZ)" in head
    assert '--label expires_at=$(date -u -d "+21600 seconds" +%FT%TZ)' in head


def test_opentela_labels_expires_at_scales_with_time():
    args = _make_args(time="00:05:00", topology=Topology(replicas=1, nodes_per_replica=1))
    head = render_rank_scripts(args)["head.sh"]
    assert '--label expires_at=$(date -u -d "+300 seconds" +%FT%TZ)' in head


def test_opentela_labels_include_framework_args():
    args = _make_args(
        framework="sglang",
        disable_opentela=False,
        framework_args="--model /path/to/model --tp 4",
        topology=Topology(replicas=1, nodes_per_replica=1),
    )
    head = render_rank_scripts(args)["head.sh"]
    assert "--label 'framework_args=--port 8080 --model /path/to/model --tp 4'" in head


def test_opentela_labels_framework_args_whitespace_normalised():
    args = _make_args(
        framework_args="--model /m     --tp 4\n    --max-len 8192",
        topology=Topology(replicas=1, nodes_per_replica=1),
    )
    head = render_rank_scripts(args)["head.sh"]
    assert "--label 'framework_args=--port 8080 --model /m --tp 4 --max-len 8192'" in head


def test_opentela_bootstrap_addr_defaults_to_prod():
    args = _make_args(topology=Topology(replicas=1, nodes_per_replica=2))
    scripts = render_rank_scripts(args)
    for s in (scripts["head.sh"], scripts["follower.sh"]):
        assert f'--bootstrap.static "{OPENTELA_BOOTSTRAP_ADDR}"' in s
        assert OPENTELA_BOOTSTRAP_ADDR_DEV not in s


def test_opentela_bootstrap_addr_dev_override():
    args = _make_args(
        topology=Topology(replicas=1, nodes_per_replica=2),
        opentela_bootstrap_addr=OPENTELA_BOOTSTRAP_ADDR_DEV,
    )
    scripts = render_rank_scripts(args)
    for s in (scripts["head.sh"], scripts["follower.sh"]):
        assert f'--bootstrap.static "{OPENTELA_BOOTSTRAP_ADDR_DEV}"' in s
        assert OPENTELA_BOOTSTRAP_ADDR not in s


def test_opentela_bootstrap_addr_custom_override():
    custom = "/ip4/10.0.0.99/tcp/43905/p2p/QmCustomPeerIdForTestingOnlyXXXXXXXXXXXXXX"
    args = _make_args(
        topology=Topology(replicas=1, nodes_per_replica=1),
        opentela_bootstrap_addr=custom,
    )
    head = render_rank_scripts(args)["head.sh"]
    assert f'--bootstrap.static "{custom}"' in head


def test_telemetry_payload_uses_resolved_bootstrap_addr():
    args = _make_args(
        telemetry_endpoint="https://telemetry.example.com/jobs",
        opentela_bootstrap_addr=OPENTELA_BOOTSTRAP_ADDR_DEV,
    )
    master = render_master(args)
    assert f'"ocf_bootstrap_addr": "{OPENTELA_BOOTSTRAP_ADDR_DEV}"' in master


def test_vllm_follower_opentela_metrics_only():
    args = _make_args(
        framework="vllm",
        disable_opentela=False,
        topology=Topology(replicas=1, nodes_per_replica=2),
    )
    follower = render_rank_scripts(args)["follower.sh"]
    assert "$OPENTELA_BIN start" in follower
    assert "--service.name" not in follower
    assert "ray start --address=" in follower


def test_router_registers_as_llm_front_door_on_router_port():
    # With a router fronting the replicas, the router is the OpenTela `llm`
    # endpoint, advertised on the router port (not the framework port).
    args = _make_args(
        disable_opentela=False,
        topology=Topology(replicas=2, nodes_per_replica=1),
        router="sglang",
    )
    router = render_rank_scripts(args)["router.sh"]
    assert "$OPENTELA_BIN start" in router
    assert "--service.name llm" in router
    assert "--service.port 30000" in router
    # The wrapped subprocess is still the router launcher.
    assert "sglang_router.launch_router" in router


def test_head_goes_metrics_only_when_fronted_by_router():
    # The heads must NOT advertise `llm` when a router fronts them, otherwise
    # OpenTela would route straight to a replica and bypass the router.
    args = _make_args(
        disable_opentela=False,
        topology=Topology(replicas=2, nodes_per_replica=1),
        router="sglang",
    )
    head = render_rank_scripts(args)["head.sh"]
    assert "$OPENTELA_BIN start" in head
    assert "--service.name" not in head


def test_head_registers_as_llm_without_router():
    # Multi-replica but no router: each head is its own `llm` endpoint.
    args = _make_args(
        disable_opentela=False,
        topology=Topology(replicas=2, nodes_per_replica=1),
        router="opentela",
    )
    head = render_rank_scripts(args)["head.sh"]
    assert "--service.name llm" in head


def test_router_omits_opentela_wrap_when_disabled():
    args = _make_args(
        disable_opentela=True,
        topology=Topology(replicas=2, nodes_per_replica=1),
        router="sglang",
    )
    router = render_rank_scripts(args)["router.sh"]
    assert "$OPENTELA_BIN start" not in router
    assert "--bootstrap.static" not in router
    assert "sglang_router.launch_router" in router


def test_telemetry_opentela_service_port_follows_router():
    # The telemetry record reports where `llm` is advertised: the router port
    # when fronted by a router, otherwise the framework port.
    with_router = render_master(
        _make_args(
            telemetry_endpoint="https://telemetry.example.com/jobs",
            topology=Topology(replicas=2, nodes_per_replica=1),
            router="sglang",
        )
    )
    assert '"ocf_service_port": 30000' in with_router

    without_router = render_master(
        _make_args(
            telemetry_endpoint="https://telemetry.example.com/jobs",
            topology=Topology(replicas=2, nodes_per_replica=1),
            router="opentela",
        )
    )
    assert '"ocf_service_port": 8080' in without_router


def test_master_telemetry_omitted_when_no_endpoint():
    args = _make_args(telemetry_endpoint=None)
    master = render_master(args)
    assert "TELEMETRY_ENDPOINT" not in master
    assert "telemetry.example.com" not in master


def test_master_telemetry_included_when_set():
    args = _make_args(telemetry_endpoint="https://telemetry.example.com/jobs")
    master = render_master(args)
    assert "https://telemetry.example.com/jobs" in master


def test_master_replica_loop_unrolled():
    args = _make_args(framework="vllm", topology=Topology(replicas=2, nodes_per_replica=4))
    master = render_master(args)
    for r in range(2):
        for k in range(4):
            role = "head" if k == 0 else "follower"
            assert f"# replica {r}, rank {k} ({role})" in master


def test_master_self_extracts_rank_scripts():
    args = _make_args(framework="vllm", topology=Topology(replicas=1, nodes_per_replica=4))
    master = render_master(args)
    assert 'cat > "$RANKS_DIR/head.sh"' in master
    assert 'cat > "$RANKS_DIR/follower.sh"' in master
    # In the working directory the job runs from (~/.sml unless configured).
    assert 'RANKS_DIR="$PWD/job-${SLURM_JOB_ID}"' in master


def test_master_binds_ranks_dir_into_container():
    args = _make_args(framework="vllm", topology=Topology(replicas=2, nodes_per_replica=4))
    master = render_master(args)
    srun_lines = [ln for ln in master.splitlines() if "--container-mounts" in ln]
    assert len(srun_lines) >= 1
    for ln in srun_lines:
        assert "$RANKS_DIR:$RANKS_DIR" in ln


# ── shadow deployments ──────────────────────────────────────────────────────
# A shadow registers its servable endpoint as `llm-shadow`, which OpenTela never
# matches for model requests, and tags every peer of the job with its base id.
# The exact `--service.name <name> \` line is asserted because a substring check
# for `llm` would also match `llm-shadow`.

_SHADOW_OF = "SwissAI-Research/swiss-ai/Apertus-8B-Instruct-2509"
_LLM_SERVICE_LINE = "    --service.name llm \\\n"
_SHADOW_SERVICE_LINE = "    --service.name llm-shadow \\\n"
_SHADOW_LABELS = ("    --label shadow=true \\\n", f"    --label shadow_of={_SHADOW_OF} \\\n")


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("nodes_per_replica", [1, 2])
def test_shadow_head_registers_as_llm_shadow(framework, nodes_per_replica):
    args = _make_args(
        framework=framework,
        topology=Topology(replicas=1, nodes_per_replica=nodes_per_replica),
        shadow_of=_SHADOW_OF,
    )
    head = render_rank_scripts(args)["head.sh"]
    assert _SHADOW_SERVICE_LINE in head
    assert _LLM_SERVICE_LINE not in head
    for label in _SHADOW_LABELS:
        assert label in head


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_shadow_follower_stays_metrics_only_with_shadow_labels(framework):
    args = _make_args(
        framework=framework,
        topology=Topology(replicas=1, nodes_per_replica=2),
        shadow_of=_SHADOW_OF,
    )
    follower = render_rank_scripts(args)["follower.sh"]
    assert "--service.name" not in follower
    for label in _SHADOW_LABELS:
        assert label in follower


def test_shadow_router_registers_as_llm_shadow_and_heads_stay_metrics_only():
    args = _make_args(
        topology=Topology(replicas=2, nodes_per_replica=1),
        router="sglang",
        shadow_of=_SHADOW_OF,
    )
    scripts = render_rank_scripts(args)
    assert _SHADOW_SERVICE_LINE in scripts["router.sh"]
    assert _LLM_SERVICE_LINE not in scripts["router.sh"]
    assert "--service.name" not in scripts["head.sh"]
    for label in _SHADOW_LABELS:
        assert label in scripts["router.sh"]
        assert label in scripts["head.sh"]


def test_non_shadow_head_registers_as_llm_without_shadow_labels():
    head = render_rank_scripts(_make_args())["head.sh"]
    assert _LLM_SERVICE_LINE in head
    assert "llm-shadow" not in head
    assert "shadow" not in head


def test_shadow_telemetry_reports_llm_shadow_service():
    master = render_master(_make_args(telemetry_endpoint="https://telemetry.example.com/jobs", shadow_of=_SHADOW_OF))
    assert '"ocf_service_name": "llm-shadow"' in master
    plain = render_master(_make_args(telemetry_endpoint="https://telemetry.example.com/jobs"))
    assert '"ocf_service_name": "llm"' in plain


def test_shadow_vmagent_labels_on_batch_and_worker_nodes():
    # 2 nodes with DCGM on: the batch node's vmagent plus one per-worker vmagent.
    args = _make_args(topology=Topology(replicas=1, nodes_per_replica=2), shadow_of=_SHADOW_OF)
    master = render_master(args)
    assert '        -remoteWrite.label="shadow=true" \\\n' in master
    assert f'        -remoteWrite.label="shadow_of={_SHADOW_OF}" \\\n' in master
    assert '-remoteWrite.label=\\"shadow=true\\" \\\n' in master
    assert f'-remoteWrite.label=\\"shadow_of={_SHADOW_OF}\\" \\\n' in master


def test_non_shadow_vmagent_has_no_shadow_labels():
    master = render_master(_make_args(topology=Topology(replicas=1, nodes_per_replica=2)))
    assert "-remoteWrite.label" in master
    assert "shadow" not in master
