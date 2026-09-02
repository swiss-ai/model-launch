#!/usr/bin/env python3

"""Resolve the latest upstream versions for images that opt into the nightly.

An image opts in by shipping a nightly.toml next to its Dockerfile, mapping
each tracked Dockerfile ARG to how its value is resolved:

    [nightly]
    vllm_arg = "VLLM_VERSION"

    [track.VLLM_VERSION]
    source = "pypi"
    package = "vllm"

    [track.TORCH_VERSION]
    source = "vllm-pin"
    package = "torch"

"pypi" takes the latest stable release. "vllm-pin" reads the `==` pin out of
the resolved vLLM release's requirements/cuda.txt: vLLM pins the torch trio and
flashinfer exactly, so bumping those to their own latest makes the resolve
unsatisfiable. They follow the vLLM release instead.

Rewrites the ARG lines in place and reports what moved. Exits 0 with
changed=false when every tracked ARG is already current -- a quiet night is a
no-op, not a failure.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import tomllib
from packaging.version import InvalidVersion, Version

_IMAGES = Path("images")
_MARKER = "nightly.toml"
_PYPI = "https://pypi.org/pypi/{package}/json"
_VLLM_CUDA_REQS = "https://raw.githubusercontent.com/vllm-project/vllm/v{version}/requirements/cuda.txt"
_TIMEOUT = 60
# Matches `name==version`, ignoring environment markers and trailing comments.
_PIN_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s;#]+)")


def _get(url: str) -> str:
    # The scheme is checked rather than trusted (ruff S310): these URLs are
    # built from module constants plus a package name read out of nightly.toml.
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to fetch non-https URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "swiss-ai-model-launch-nightly"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
        return str(response.read().decode("utf-8"))


def pypi_latest(package: str) -> str:
    """Latest non-prerelease version of a distribution on PyPI.

    info.version is not used directly: it tracks the most recent upload, which
    can be a pre-release. Nightly bumps should not put an rc into a published
    image, so the maximum stable version is computed from the release list.
    """
    data = json.loads(_get(_PYPI.format(package=package)))
    stable = []
    for raw, files in data.get("releases", {}).items():
        # An empty or fully yanked release is not installable.
        if not files or all(f.get("yanked") for f in files):
            continue
        try:
            parsed = Version(raw)
        except InvalidVersion:
            continue
        if not parsed.is_prerelease:
            stable.append(parsed)
    if not stable:
        raise RuntimeError(f"No stable release found on PyPI for '{package}'")
    return str(max(stable))


def vllm_pins(vllm_version: str) -> dict[str, str]:
    """The `==` pins from a vLLM release's requirements/cuda.txt."""
    try:
        body = _get(_VLLM_CUDA_REQS.format(version=vllm_version))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Cannot read requirements/cuda.txt for vLLM v{vllm_version}: {e}") from e
    pins = {}
    for line in body.splitlines():
        match = _PIN_RE.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def read_args(dockerfile: Path) -> dict[str, str]:
    args = {}
    for line in dockerfile.read_text().splitlines():
        match = re.match(r"^ARG\s+([A-Z0-9_]+)=(\S*)\s*$", line)
        if match:
            args[match.group(1)] = match.group(2)
    return args


def rewrite_args(dockerfile: Path, updates: dict[str, str]) -> None:
    text = dockerfile.read_text()
    for name, value in updates.items():
        pattern = re.compile(rf"^ARG\s+{re.escape(name)}=\S*$", re.MULTILINE)
        text, count = pattern.subn(f"ARG {name}={value}", text)
        if count != 1:
            raise RuntimeError(f"Expected exactly one 'ARG {name}=' line in {dockerfile}, found {count}")
    dockerfile.write_text(text)


def resolve(image_dir: Path) -> dict[str, tuple[str, str]]:
    """Tracked ARGs whose resolved value differs from the Dockerfile's.

    Returns {arg: (current, latest)}. Entries that are already current are
    omitted, so an empty result means there is nothing to bump.
    """
    config = tomllib.loads((image_dir / _MARKER).read_text())
    tracked = config.get("track", {})
    dockerfile = image_dir / "Dockerfile"
    current = read_args(dockerfile)

    # "vllm-pin" entries follow whatever VLLM_VERSION resolves to in this same
    # run, so the pins are read from the release being proposed rather than the
    # one currently in the Dockerfile.
    resolved: dict[str, str] = {}
    for arg, entry in tracked.items():
        if entry.get("source") == "pypi":
            resolved[arg] = pypi_latest(entry["package"])

    pin_args = {a: e for a, e in tracked.items() if e.get("source") == "vllm-pin"}
    if pin_args:
        vllm_arg = config.get("nightly", {}).get("vllm_arg", "VLLM_VERSION")
        vllm_version = resolved.get(vllm_arg, current.get(vllm_arg))
        if not vllm_version:
            raise RuntimeError(f"{image_dir}: '{vllm_arg}' is needed for vllm-pin entries but is not set")
        pins = vllm_pins(vllm_version)
        for arg, entry in pin_args.items():
            package = entry["package"].lower()
            if package not in pins:
                raise RuntimeError(f"vLLM v{vllm_version} requirements/cuda.txt has no '=={package}' pin")
            resolved[arg] = pins[package]

    changes = {}
    for arg, latest in resolved.items():
        if arg not in current:
            raise RuntimeError(f"{dockerfile} has no 'ARG {arg}=' line to bump")
        if current[arg] != latest:
            changes[arg] = (current[arg], latest)
    return changes


def main() -> int:
    images = sorted(d for d in _IMAGES.iterdir() if (d / _MARKER).is_file())
    if not images:
        print(f"No image declares {_MARKER}; nothing to do.")
        return _emit(changed=False, images=[], summary="No image opts into the nightly.")

    bumped = []
    sections = []
    for image_dir in images:
        print(f"== {image_dir.name}")
        changes = resolve(image_dir)
        if not changes:
            print("  already current")
            continue
        rewrite_args(image_dir / "Dockerfile", {a: new for a, (_, new) in changes.items()})
        bumped.append(image_dir.name)
        rows = "\n".join(f"| `{arg}` | {old} | **{new}** |" for arg, (old, new) in sorted(changes.items()))
        sections.append(f"### `{image_dir.name}`\n\n| ARG | from | to |\n| --- | --- | --- |\n{rows}")
        for arg, (old, new) in sorted(changes.items()):
            print(f"  {arg}: {old} -> {new}")

    if not bumped:
        return _emit(changed=False, images=[], summary="Every tracked dependency is already current.")
    return _emit(changed=True, images=bumped, summary="\n\n".join(sections))


def _emit(*, changed: bool, images: list[str], summary: str) -> int:
    print(f"\nchanged={str(changed).lower()} images={images}")
    Path("nightly-summary.md").write_text(summary + "\n")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a") as handle:
            handle.write(f"changed={str(changed).lower()}\n")
            handle.write(f"images={json.dumps(images)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
