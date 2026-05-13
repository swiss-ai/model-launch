# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
import shutil
import subprocess
from pathlib import Path

import pytest

from swiss_ai_model_launch.launchers.framework import (
    Sglang,
    Vllm,
    _make_framework,
    render_master,
    render_rank_scripts,
)
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs, Topology

_HAS_SHELLCHECK = shutil.which("shellcheck") is not None


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


# ── factory ────────────────────────────────────────────────────────────────


def test_make_framework_known():
    assert _make_framework("sglang").name == "sglang"
    assert _make_framework("vllm").name == "vllm"


def test_make_framework_unknown_raises():
    with pytest.raises(ValueError, match="Unknown framework"):
        _make_framework("nonexistent")


# ── shape tests ────────────────────────────────────────────────────────────


def test_sglang_singular_has_only_head():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=1))
    scripts = render_rank_scripts(args)
    assert set(scripts) == {"head.sh"}


def test_sglang_multi_node_has_head_and_follower():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=4))
    scripts = render_rank_scripts(args)
    assert set(scripts) == {"head.sh", "follower.sh"}


def test_router_only_when_multi_replica_and_use_router():
    # No router when single replica
    args = _make_args(use_router=True, topology=Topology(replicas=1, nodes_per_replica=4))
    assert "router.sh" not in render_rank_scripts(args)
    # No router when multi-replica but use_router=False
    args = _make_args(use_router=False, topology=Topology(replicas=2, nodes_per_replica=4))
    assert "router.sh" not in render_rank_scripts(args)
    # Router when both
    args = _make_args(use_router=True, topology=Topology(replicas=2, nodes_per_replica=4))
    assert "router.sh" in render_rank_scripts(args)


# ── content sanity ─────────────────────────────────────────────────────────


def test_sglang_head_singular_uses_entrypoint_directly():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=1))
    head = render_rank_scripts(args)["head.sh"]
    assert "python3 -m sglang.launch_server" in head
    assert "--dist-init-addr" not in head  # singular has no dist init


def test_sglang_multi_node_passes_node_rank():
    args = _make_args(framework="sglang", topology=Topology(replicas=1, nodes_per_replica=4))
    follower = render_rank_scripts(args)["follower.sh"]
    assert '--node-rank "$node_rank"' in follower
    assert "--nnodes 4" in follower


def test_vllm_multi_node_uses_ray_with_script_on_disk():
    args = _make_args(framework="vllm", topology=Topology(replicas=1, nodes_per_replica=4))
    head = render_rank_scripts(args)["head.sh"]
    # Single-quoted heredoc keeps $-constructs literal in the on-disk file —
    # this is PR #124's fix preserved through the refactor.
    assert "<<'__SML_RAY_HEAD_EOF__'" in head
    assert "ray start --head" in head
    assert "bash $ray_head_script" in head or 'bash "$ray_head_script"' in head


def test_vllm_follower_joins_ray():
    args = _make_args(framework="vllm", topology=Topology(replicas=1, nodes_per_replica=4))
    follower = render_rank_scripts(args)["follower.sh"]
    assert "ray start --address=" in follower
    assert "vllm.entrypoints" not in follower  # follower doesn't run the API server


def test_ocf_disabled_omits_wrap():
    args = _make_args(disable_ocf=True, topology=Topology(replicas=1, nodes_per_replica=1))
    head = render_rank_scripts(args)["head.sh"]
    assert "$OCF_BIN start" not in head
    assert "--bootstrap.addr" not in head


def test_ocf_enabled_wraps_head_only():
    args = _make_args(
        framework="sglang",
        disable_ocf=False,
        topology=Topology(replicas=1, nodes_per_replica=4),
    )
    scripts = render_rank_scripts(args)
    assert "$OCF_BIN start" in scripts["head.sh"]
    assert "$OCF_BIN start" not in scripts["follower.sh"]


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
    """Each (replica, rank) gets an explicit srun block — no bash for-loops."""
    args = _make_args(framework="vllm", topology=Topology(replicas=2, nodes_per_replica=4))
    master = render_master(args)
    # Headers are present for every (replica, rank) pair
    for r in range(2):
        for k in range(4):
            role = "head" if k == 0 else "follower"
            assert f"# replica {r}, rank {k} ({role})" in master


def test_master_self_extracts_rank_scripts():
    args = _make_args(framework="vllm", topology=Topology(replicas=1, nodes_per_replica=4))
    master = render_master(args)
    assert 'cat > "$RANKS_DIR/head.sh"' in master
    assert 'cat > "$RANKS_DIR/follower.sh"' in master
    assert 'RANKS_DIR="$HOME/.sml/job-${SLURM_JOB_ID}"' in master


def test_master_binds_ranks_dir_into_container():
    """Each srun must bind-mount RANKS_DIR so the rank script is visible
    inside the pyxis container — see the vllm.toml mount restrictions."""
    args = _make_args(framework="vllm", topology=Topology(replicas=2, nodes_per_replica=4))
    master = render_master(args)
    # Every srun line should have the --container-mounts flag for $RANKS_DIR.
    srun_lines = [ln for ln in master.splitlines() if "--container-mounts" in ln]
    assert len(srun_lines) >= 1
    for ln in srun_lines:
        assert "$RANKS_DIR:$RANKS_DIR" in ln


# ── env_exports ────────────────────────────────────────────────────────────


def test_sglang_exports_both_jit_deepgemm_names():
    """SGL_* is the upstream-historical name, SGLANG_* is the current one
    (transitioning). Both are exported during the upstream transition."""
    exports = "\n".join(Sglang.env_exports)
    assert 'SGL_ENABLE_JIT_DEEPGEMM="false"' in exports
    assert 'SGLANG_ENABLE_JIT_DEEPGEMM="false"' in exports


def test_vllm_exports_ray_cgraph_timeout():
    assert "export RAY_CGRAPH_get_timeout=1800" in Vllm.env_exports


# ── matrix: bash -n + shellcheck on rendered output ───────────────────────


_MATRIX_CONFIGS = [
    pytest.param(
        fw,
        replicas,
        npr,
        use_router,
        disable_ocf,
        telemetry,
        id=f"{fw}-r{replicas}-n{npr}-{'router' if use_router else 'norouter'}-"
        f"{'noocf' if disable_ocf else 'ocf'}-"
        f"{'tele' if telemetry else 'notele'}",
    )
    for fw in ("sglang", "vllm")
    for replicas in (1, 2)
    for npr in (1, 4)
    for use_router in (False, True)
    for disable_ocf in (False, True)
    for telemetry in (False, True)
]


@pytest.mark.parametrize("framework,replicas,npr,use_router,disable_ocf,telemetry", _MATRIX_CONFIGS)
def test_rendered_scripts_pass_bash_n(
    tmp_path: Path,
    framework: str,
    replicas: int,
    npr: int,
    use_router: bool,
    disable_ocf: bool,
    telemetry: bool,
):
    args = _make_args(
        framework=framework,
        topology=Topology(replicas=replicas, nodes_per_replica=npr),
        use_router=use_router,
        disable_ocf=disable_ocf,
        telemetry_endpoint="https://telemetry.example.com/jobs" if telemetry else None,
    )
    master_path = tmp_path / "master.sh"
    master_path.write_text("#!/bin/bash\n" + render_master(args))
    result = subprocess.run(["bash", "-n", str(master_path)], capture_output=True)
    assert result.returncode == 0, f"bash -n failed for master.sh:\n{result.stderr.decode()}"
    # Also bash -n each individual rank script (renders should be standalone-valid).
    for filename, content in render_rank_scripts(args).items():
        path = tmp_path / filename
        path.write_text(content)
        r = subprocess.run(["bash", "-n", str(path)], capture_output=True)
        assert r.returncode == 0, f"bash -n failed for {filename}:\n{r.stderr.decode()}"


@pytest.mark.skipif(not _HAS_SHELLCHECK, reason="shellcheck not installed")
@pytest.mark.parametrize("framework,replicas,npr,use_router,disable_ocf,telemetry", _MATRIX_CONFIGS)
def test_rendered_scripts_pass_shellcheck(
    tmp_path: Path,
    framework: str,
    replicas: int,
    npr: int,
    use_router: bool,
    disable_ocf: bool,
    telemetry: bool,
):
    args = _make_args(
        framework=framework,
        topology=Topology(replicas=replicas, nodes_per_replica=npr),
        use_router=use_router,
        disable_ocf=disable_ocf,
        telemetry_endpoint="https://telemetry.example.com/jobs" if telemetry else None,
    )
    master_path = tmp_path / "master.sh"
    master_path.write_text("#!/bin/bash\n" + render_master(args))
    result = subprocess.run(
        ["shellcheck", "-S", "warning", str(master_path)],
        capture_output=True,
    )
    assert result.returncode == 0, f"shellcheck failed for master.sh:\n{result.stdout.decode()}"
    for filename, content in render_rank_scripts(args).items():
        path = tmp_path / filename
        path.write_text(content)
        r = subprocess.run(["shellcheck", "-S", "warning", str(path)], capture_output=True)
        assert r.returncode == 0, f"shellcheck failed for {filename}:\n{r.stdout.decode()}"
