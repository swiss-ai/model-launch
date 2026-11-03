from pathlib import Path
from typing import Any

import pytest

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.container import PyxisEnvironment
from swiss_ai_model_launch.launchers.framework import render_master
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology
from swiss_ai_model_launch.launchers.utils import render_sbatch_header

_TOML = """
image = "/scratch/$USER/images/vllm-{arch}.sqsh"
mounts = ["/scratch", "/data:/mnt/data:ro"]
workdir = "/opt"

[env]
HF_HOME = "/scratch/$USER/hf"

[annotations]
com.hooks.cxi.enabled = "true"
"""


def _make_args(env_path: Path, **overrides: Any) -> LaunchArgs:
    defaults = dict(
        job_name="test_job",
        served_model_name="vendor/model-abc1",
        account="proj01",
        partition="normal",
        environment=str(env_path),
        framework="vllm",
        container_runtime="pyxis",
        disable_opentela=True,
        disable_metrics=True,
    )
    return LaunchArgs(**{**defaults, **overrides})


@pytest.fixture
def env_path(tmp_path: Path) -> Path:
    p = tmp_path / "env.toml"
    p.write_text(_TOML)
    return p


def test_pyxis_environment_from_toml(env_path: Path) -> None:
    env = PyxisEnvironment.from_toml(str(env_path))
    assert env.image == "/scratch/$USER/images/vllm-{arch}.sqsh"
    assert env.container_mounts() == ["/scratch:/scratch", "/data:/mnt/data:ro"]
    assert env.workdir == "/opt"
    assert env.env == {"HF_HOME": "/scratch/$USER/hf"}


def test_pyxis_environment_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reads the environment toml locally"):
        PyxisEnvironment.from_toml(str(tmp_path / "nope.toml"))


def test_pyxis_master_uses_container_image(env_path: Path) -> None:
    master = render_master(_make_args(env_path, topology=Topology(replicas=2)))
    assert "--environment=" not in master
    assert 'SML_IMAGE="/scratch/$USER/images/vllm-${SML_ARCH}.sqsh"' in master
    assert 'export HF_HOME="/scratch/$USER/hf"' in master
    assert 'SML_CONTAINER_MOUNTS="/scratch:/scratch,/data:/mnt/data:ro"' in master
    assert master.count('--container-image="$SML_IMAGE"') == 2
    assert master.count('--container-mounts="$RANKS_DIR:$RANKS_DIR,$SML_CONTAINER_MOUNTS"') == 2
    assert '--container-workdir="/opt"' in master
    assert master.count("--container-env=HF_HOME") == 2


def test_pyxis_router_uses_container_image(env_path: Path) -> None:
    master = render_master(_make_args(env_path, framework="sglang", router="sglang", topology=Topology(replicas=2)))
    assert master.count('--container-image="$SML_IMAGE"') == 3


def test_edf_is_default(env_path: Path) -> None:
    master = render_master(_make_args(env_path, container_runtime="edf"))
    assert '--environment="${SML_ENV_FILE}"' in master
    assert "--container-image" not in master


def test_sbatch_extra_args(env_path: Path) -> None:
    args = _make_args(env_path, sbatch_extra_args=["--comment=field=nlp;appl=vllm", "--gres=gpu:4"])
    assert args.to_sbatch_args()[-2:] == ["--comment=field=nlp;appl=vllm", "--gres=gpu:4"]
    header = render_sbatch_header(args)
    assert "#SBATCH --comment=field=nlp;appl=vllm\n" in header
    assert "#SBATCH --gres=gpu:4\n" in header


def test_cli_wiring(env_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_SBATCH_ARGS", "--comment='field=nlp;appl=vllm'")
    args = _build_parser().parse_args(
        [
            "advanced",
            "--framework",
            "vllm",
            "--environment",
            str(env_path),
            "--container-runtime",
            "pyxis",
            "--sbatch-arg=--gres=gpu:4",
            "--framework-args",
            "--model m --served-model-name m",
        ]
    )
    launch_args = build_launch_args_from_advanced(args, username="alice", account="proj01", partition="normal")
    assert launch_args.container_runtime == "pyxis"
    assert launch_args.sbatch_extra_args == ["--comment=field=nlp;appl=vllm", "--gres=gpu:4"]
