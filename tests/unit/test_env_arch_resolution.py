# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""The {arch} placeholder in an env toml's image path, resolved on the batch node.

These run the generated shell for real: the bug this guards against (a relative
resolved path, which pyxis reinterprets as an EDF *name* and suffixes with
".toml") is invisible to any test that only inspects the rendered string.
"""

import subprocess
from pathlib import Path

from swiss_ai_model_launch.launchers.framework import _render_env_file_resolution
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs

_IMAGE_DIR = "imgs"


def _args(environment: str) -> LaunchArgs:
    return LaunchArgs(
        job_name="j",
        served_model_name="vendor/model-abc1",
        account="proj01",
        partition="normal",
        environment=environment,
        framework="vllm",
    )


def _run(env_file: Path, arch: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    block = _render_env_file_resolution(_args(str(env_file)))
    script = f'set -euo pipefail\nSML_ARCH={arch}\n{block}\necho "RESOLVED=$SML_ENV_FILE"\n'
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,  # deliberately not the env file's directory
        env={"SLURM_JOB_ID": "2724141", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


def _resolved(proc: subprocess.CompletedProcess[str]) -> str:
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESOLVED="))
    return line.removeprefix("RESOLVED=")


def _write_env(tmp_path: Path, image: str) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    env_file = work / "env_vllm_abc.toml"
    env_file.write_text(f'image = "{image}"\n')
    return env_file


def test_placeholder_resolves_to_absolute_path(tmp_path: Path) -> None:
    """pyxis needs a path, not a bare name: a value with no "/" becomes an EDF name."""
    (tmp_path / _IMAGE_DIR).mkdir()
    (tmp_path / _IMAGE_DIR / "vllm-arm64.sqsh").touch()
    env_file = _write_env(tmp_path, f"{tmp_path}/{_IMAGE_DIR}/vllm-{{arch}}.sqsh")

    proc = _run(env_file, "arm64", cwd=tmp_path)

    assert proc.returncode == 0, proc.stderr
    resolved = Path(_resolved(proc))
    assert resolved.is_absolute()
    assert resolved.read_text() == f'image = "{tmp_path}/{_IMAGE_DIR}/vllm-arm64.sqsh"\n'
    # Lands in the job's working dir, not beside the source toml — which for the
    # local SLURM launcher is the read-only packaged asset directory.
    assert resolved.parent == tmp_path
    assert not list(env_file.parent.glob("env_resolved_*"))


def test_placeholder_resolves_per_arch(tmp_path: Path) -> None:
    (tmp_path / _IMAGE_DIR).mkdir()
    (tmp_path / _IMAGE_DIR / "vllm-amd64.sqsh").touch()
    env_file = _write_env(tmp_path, f"{tmp_path}/{_IMAGE_DIR}/vllm-{{arch}}.sqsh")

    proc = _run(env_file, "amd64", cwd=tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "vllm-amd64.sqsh" in Path(_resolved(proc)).read_text()


def test_pinned_path_is_passed_through_untouched(tmp_path: Path) -> None:
    (tmp_path / _IMAGE_DIR).mkdir()
    (tmp_path / _IMAGE_DIR / "vllm-arm64.sqsh").touch()
    env_file = _write_env(tmp_path, f"{tmp_path}/{_IMAGE_DIR}/vllm-arm64.sqsh")

    proc = _run(env_file, "arm64", cwd=tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _resolved(proc) == str(env_file)


def test_missing_image_fails_with_the_wanted_path(tmp_path: Path) -> None:
    """Otherwise pyxis reports it as an opaque failure deep inside srun."""
    env_file = _write_env(tmp_path, f"{tmp_path}/{_IMAGE_DIR}/absent-{{arch}}.sqsh")

    proc = _run(env_file, "arm64", cwd=tmp_path)

    assert proc.returncode == 1
    assert f"{tmp_path}/{_IMAGE_DIR}/absent-arm64.sqsh" in proc.stderr
