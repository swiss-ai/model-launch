"""Pull the cluster paths an example recipe hands to its framework.

`sml advanced` examples carry their model in a `--framework-args` string, either
as an absolute path or as an HF repo id resolved under the model registry, and
some assign it to a shell variable first. This module extracts those references
so CI can confirm they still point at something loadable; the checks themselves
live in `swiss_ai_model_launch.launchers.path_check`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from swiss_ai_model_launch.launchers.path_check import MODEL_MARKERS, TOKENIZER_MARKERS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Only clariden: these are the recipes CI's FirecREST credentials can see. The
# beverin/bristen examples target other clusters with their own filesystems.
CLARIDEN_EXAMPLES = REPO_ROOT / "examples" / "clariden"

# Framework flags whose value is a path (or repo id) on the cluster, and the
# marker files that prove the directory behind it is the right kind of thing.
# sglang spells the model flag `--model-path`, vllm `--model`.
_PATH_FLAGS: dict[str, tuple[str, ...]] = {
    "model-path": MODEL_MARKERS,
    "model": MODEL_MARKERS,
    "tokenizer": TOKENIZER_MARKERS,
}

# Longest flag name first: `--model-path` must not be read as `--model`.
_FLAG_PATTERN = re.compile(
    r"--(" + "|".join(sorted(_PATH_FLAGS, key=len, reverse=True)) + r")(?:=|\s+)[\"']?([^\s\\\"']+)"
)

# `MODEL="/capstor/..."`, optionally exported. Only literal values: anything with
# a substitution or command in it is left for the caller to notice as unresolved.
_ASSIGNMENT_PATTERN = re.compile(
    r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(?:"([^"$`\n]*)"|'([^'\n]*)'|([^\s"'$`]+))\s*$""",
    re.MULTILINE,
)

_VARIABLE_PATTERN = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")

_LINE_CONTINUATION_PATTERN = re.compile(r"\\\n")
_COMMENT_PATTERN = re.compile(r"^\s*#.*$", re.MULTILINE)


@dataclass(frozen=True)
class PathReference:
    """One path-valued flag found in one example script."""

    script: Path  # relative to the repo root
    flag: str  # e.g. "--model-path"
    value: str  # after shell-variable expansion
    markers: tuple[str, ...]

    @property
    def kind(self) -> Literal["absolute", "repo-id"]:
        """Absolute paths are used as-is; a repo id resolves under the registry."""
        return "absolute" if self.value.startswith("/") else "repo-id"

    @property
    def unresolved(self) -> bool:
        """True when a shell substitution survived expansion, so CI can't check it."""
        return "$" in self.value

    @property
    def label(self) -> str:
        return f"{self.script} ({self.flag})"

    def resolve(self, registry: Path) -> str:
        """The cluster path this reference names, resolving repo ids under ``registry``."""
        return self.value if self.kind == "absolute" else str(registry / self.value)


def extract_references(script: Path, text: str) -> list[PathReference]:
    """Every path-valued flag in ``text``, with simple shell variables expanded."""
    assignments = {
        match.group(1): next(group for group in match.groups()[1:] if group is not None)
        for match in _ASSIGNMENT_PATTERN.finditer(text)
    }
    body = _COMMENT_PATTERN.sub("", _LINE_CONTINUATION_PATTERN.sub(" ", text))

    references = []
    for flag, raw_value in _FLAG_PATTERN.findall(body):
        expanded = _VARIABLE_PATTERN.sub(lambda m: assignments.get(m.group(1), m.group(0)), raw_value)
        references.append(
            PathReference(
                script=script,
                flag=f"--{flag}",
                # A trailing slash is harmless to a launch but would make the same
                # directory look like two distinct references.
                value=expanded.rstrip("/"),
                markers=_PATH_FLAGS[flag],
            )
        )
    return references


def discover_references(examples_dir: Path = CLARIDEN_EXAMPLES) -> list[PathReference]:
    """Every path reference in every example script under ``examples_dir``."""
    references = []
    for script in sorted(examples_dir.rglob("*.sh")):
        references.extend(extract_references(script.relative_to(REPO_ROOT), script.read_text()))
    return references


@dataclass(frozen=True)
class ReferenceGroup:
    """One distinct path, plus every script/flag that names it.

    Examples repeat the same weights a lot (a model and its sglang/vllm variants,
    a shared tokenizer), so checking per group keeps the sweep to one listing per
    directory while still naming every script in a failure.
    """

    reference: PathReference
    labels: tuple[str, ...]

    @property
    def value(self) -> str:
        return self.reference.value

    @property
    def markers(self) -> tuple[str, ...]:
        return self.reference.markers

    @property
    def label(self) -> str:
        return "; ".join(self.labels)

    def resolve(self, registry: Path) -> str:
        return self.reference.resolve(registry)


def group_references(references: list[PathReference]) -> list[ReferenceGroup]:
    """Collapse references naming the same path, keeping discovery order."""
    grouped: dict[tuple[str, tuple[str, ...]], list[PathReference]] = {}
    for reference in references:
        grouped.setdefault((reference.value, reference.markers), []).append(reference)
    return [
        ReferenceGroup(reference=members[0], labels=tuple(sorted({m.label for m in members})))
        for members in grouped.values()
    ]
