# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries
"""Render selected real example shell scripts and validate their bash output.

Each test reads an actual file from ``examples/``, parses its `sml advanced`
invocation through the production CLI parser, builds a LaunchArgs via the
same helper used at runtime, and renders + shellchecks the output. If the
example file is renamed or its flags change incompatibly, the test breaks.
"""

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.framework import render_master, render_rank_scripts

_HAS_SHELLCHECK = shutil.which("shellcheck") is not None
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_sml_advanced_script(content: str):
    """Extract the `sml advanced` flags from an example shell script.

    Strips comments and bash line-continuations (``\\\\\\n``) then tokenises
    via shlex and parses with the production CLI parser. Returns the
    argparse Namespace.
    """
    text = content
    text = re.sub(r"\\\n", " ", text)  # collapse line continuations
    text = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)  # strip comments
    tokens = shlex.split(text)
    if tokens[:2] != ["sml", "advanced"]:
        raise ValueError(f"Not an `sml advanced` script (got {tokens[:2]!r})")
    parser = _build_parser()
    return parser.parse_args(tokens[1:])  # drop 'sml', parse from 'advanced'


# A handful of real examples covering distinct shapes:
# - singular sglang (most common case)
# - singular sglang on a different system + partition (beverin/mi300, ROCm env)
# - singular vllm (the common single-node vllm case)
# - multi-node single-replica sglang (--slurm-nodes-per-replica > 1)
# - multi-replica multi-node with router (--use-router + --slurm-replicas)
# - multi-node single-replica vllm (vLLM Ray bootstrapping path)
_SELECTED_EXAMPLES = [
    "examples/clariden/cli/swiss-ai/Apertus-8B-Instruct-2509-sglang.sh",
    "examples/beverin/cli/swiss-ai/Apertus-70B-Instruct-2509-sglang-rocm.sh",
    "examples/clariden/cli/mistralai/Ministral-3-3B-Instruct-2512-vllm.sh",
    "examples/clariden/cli/deepseek-ai/DeepSeek-V3.1-sglang.sh",
    "examples/clariden/cli/deepseek-ai/DeepSeek-V3.1-sglang-router.sh",
    "examples/clariden/cli/qwen/Qwen3-235B-A22B-Instruct-2507-vllm.sh",
]


@pytest.mark.parametrize("example_path", _SELECTED_EXAMPLES, ids=lambda p: Path(p).stem)
def test_example_renders_valid_bash(tmp_path: Path, example_path: str):
    full_path = _REPO_ROOT / example_path
    if not full_path.exists():
        pytest.skip(f"Example not found: {example_path}")

    args = _parse_sml_advanced_script(full_path.read_text())
    # Synthesise the launcher-supplied bits: in production these come from
    # InitConfig + the FirecREST/Slurm launcher's account+partition.
    launch_args = build_launch_args_from_advanced(
        args,
        account="proj01-test",
        partition="normal",
    )

    master_path = tmp_path / "master.sh"
    master_path.write_text("#!/bin/bash\n" + render_master(launch_args))
    rank_paths = []
    for filename, content in render_rank_scripts(launch_args).items():
        p = tmp_path / filename
        p.write_text(content)
        rank_paths.append(p)
    files = {"master.sh": master_path, **{p.name: p for p in rank_paths}}
    for filename, path in files.items():
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True)
        assert result.returncode == 0, f"bash -n failed for {example_path} → {filename}:\n{result.stderr.decode()}"


@pytest.mark.skipif(not _HAS_SHELLCHECK, reason="shellcheck not installed")
@pytest.mark.parametrize("example_path", _SELECTED_EXAMPLES, ids=lambda p: Path(p).stem)
def test_example_passes_shellcheck(tmp_path: Path, example_path: str):
    full_path = _REPO_ROOT / example_path
    if not full_path.exists():
        pytest.skip(f"Example not found: {example_path}")

    args = _parse_sml_advanced_script(full_path.read_text())
    launch_args = build_launch_args_from_advanced(
        args,
        account="proj01-test",
        partition="normal",
    )

    master_path = tmp_path / "master.sh"
    master_path.write_text("#!/bin/bash\n" + render_master(launch_args))
    rank_paths = []
    for filename, content in render_rank_scripts(launch_args).items():
        p = tmp_path / filename
        p.write_text(content)
        rank_paths.append(p)
    files = {"master.sh": master_path, **{p.name: p for p in rank_paths}}
    for filename, path in files.items():
        result = subprocess.run(
            ["shellcheck", "-S", "warning", str(path)],
            capture_output=True,
        )
        assert result.returncode == 0, f"shellcheck failed for {example_path} → {filename}:\n{result.stdout.decode()}"
