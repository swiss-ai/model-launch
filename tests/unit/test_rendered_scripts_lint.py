# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
import shutil
import subprocess
from pathlib import Path

import pytest

from swiss_ai_model_launch.launchers.framework import render_master, render_rank_scripts
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology

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


_MATRIX_CONFIGS = [
    pytest.param(
        fw,
        replicas,
        npr,
        use_router,
        disable_opentela,
        telemetry,
        id=f"{fw}-r{replicas}-n{npr}-{'router' if use_router else 'norouter'}-"
        f"{'noopentela' if disable_opentela else 'opentela'}-"
        f"{'tele' if telemetry else 'notele'}",
    )
    for fw in ("sglang", "vllm")
    for replicas in (1, 2)
    for npr in (1, 4)
    for use_router in (False, True)
    for disable_opentela in (False, True)
    for telemetry in (False, True)
]


@pytest.mark.parametrize("framework,replicas,npr,use_router,disable_opentela,telemetry", _MATRIX_CONFIGS)
def test_rendered_scripts_pass_bash_n(
    tmp_path: Path,
    framework: str,
    replicas: int,
    npr: int,
    use_router: bool,
    disable_opentela: bool,
    telemetry: bool,
):
    args = _make_args(
        framework=framework,
        topology=Topology(replicas=replicas, nodes_per_replica=npr),
        router="SGLANG" if use_router else "OPENTELA",
        disable_opentela=disable_opentela,
        telemetry_endpoint="https://telemetry.example.com/jobs" if telemetry else None,
    )
    master_path = tmp_path / "master.sh"
    master_path.write_text("#!/bin/bash\n" + render_master(args))
    result = subprocess.run(["bash", "-n", str(master_path)], capture_output=True)
    assert result.returncode == 0, f"bash -n failed for master.sh:\n{result.stderr.decode()}"
    for filename, content in render_rank_scripts(args).items():
        path = tmp_path / filename
        path.write_text(content)
        r = subprocess.run(["bash", "-n", str(path)], capture_output=True)
        assert r.returncode == 0, f"bash -n failed for {filename}:\n{r.stderr.decode()}"


@pytest.mark.skipif(not _HAS_SHELLCHECK, reason="shellcheck not installed")
@pytest.mark.parametrize("framework,replicas,npr,use_router,disable_opentela,telemetry", _MATRIX_CONFIGS)
def test_rendered_scripts_pass_shellcheck(
    tmp_path: Path,
    framework: str,
    replicas: int,
    npr: int,
    use_router: bool,
    disable_opentela: bool,
    telemetry: bool,
):
    args = _make_args(
        framework=framework,
        topology=Topology(replicas=replicas, nodes_per_replica=npr),
        router="SGLANG" if use_router else "OPENTELA",
        disable_opentela=disable_opentela,
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
