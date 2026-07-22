#!/usr/bin/env python3
"""Scan a pushed container image for exposed secrets, layer by layer.

Scans the layered image on GHCR rather than the flattened sqsh, because the
registry image is what actually leaks: a file added in one layer and deleted
in a later one disappears from the sqsh but stays pullable from the registry
forever. For each layer this script checks

  * file contents with trufflehog (every layer independently, so
    deleted-in-a-later-layer files are covered),
  * filenames against credential-shaped patterns (.netrc, id_rsa, ...),
  * and the image config (build history commands, ENV, labels).

Failure policy: a finding fails the scan when trufflehog verified it against
the issuing service, or when it comes from a high-signal detector (private
keys, GitHub/AWS/HF tokens, credentialed URIs, ...). Everything else is
reported as a warning only — big ML images are full of high-entropy noise
(checksums, static libraries, sample notebooks). Known-benign findings are
suppressed via .github/image-scan-allowlist.txt.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REGISTRY = "ghcr.io"
_ORG = "swiss-ai"
_ALLOWLIST = Path(".github/image-scan-allowlist.txt")
_CHANNEL_RE = re.compile(r"^(latest|pr-\d+)$")

_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)

# Detectors whose matches are structurally credential-like; unverified hits
# from these fail the scan unless allowlisted. Everything else (generic
# entropy matches in binaries, checksums, locale packs) is warn-only.
_HIGH_SIGNAL_DETECTORS = {
    "anthropic",
    "aws",
    "awssessionkey",
    "azurestorage",
    "dockerhub",
    "ftp",
    "gcp",
    "gcpapplicationdefaultcredentials",
    "github",
    "githubapp",
    "githuboauth2",
    "gitlab",
    "gitlabv2",
    "huggingface",
    "jdbc",
    "mongodb",
    "npmtoken",
    "openai",
    "postgres",
    "privatekey",
    "pypi",
    "redis",
    "sendgrid",
    "slack",
    "slackwebhook",
    "stripe",
    "telegrambottoken",
    "twilio",
    "uri",
}

# Documentation and test fixtures across the Python ecosystem use these in
# example credential URIs; a real leaked credential never points there.
_PLACEHOLDER_RE = re.compile(
    r"user:pass\b|username:password|REDACTED|example\.(com|org)|localhost|127\.0\.0\.1",
    re.I,
)

# Credential-shaped file names. Matched against every path in every layer,
# with whiteout prefixes stripped so a deleted credential file still trips.
_SUSPICIOUS_NAME_RE = re.compile(
    r"(^|/)("
    r"\.netrc|_netrc|\.git-credentials|\.npmrc|\.pypirc|"
    r"id_rsa[^/]*|id_dsa[^/]*|id_ecdsa[^/]*|id_ed25519[^/]*|"
    r"\.docker/config\.json|\.aws/credentials|\.kube/config|"
    r"[^/]+\.keytab|\.bash_history|\.zsh_history|\.python_history"
    r")$"
)
_PUBLIC_KEY_RE = re.compile(r"\.pub$")

# Patterns that must never appear in build history commands, ENV, or labels.
_CONFIG_SECRET_RE = re.compile(
    r"ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-|"
    r"password=|passwd=|--password|api[_-]?key\s*="
)


def _load_allowlist() -> list[tuple[str, str]]:
    rules = []
    if _ALLOWLIST.exists():
        for line in _ALLOWLIST.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kind, _, pattern = line.partition(":")
            if kind not in ("raw", "path") or not pattern:
                print(f"WARNING: ignoring malformed allowlist rule: {line}")
                continue
            rules.append((kind, pattern))
    return rules


def _allowed(rules: list[tuple[str, str]], path: str, raw: str = "") -> bool:
    for kind, pattern in rules:
        if kind == "path" and fnmatch.fnmatch(path, pattern):
            return True
        if kind == "raw" and raw and re.search(pattern, raw):
            return True
    return False


def _registry_token(repo: str) -> str:
    cmd = ["curl", "-sf", "--retry", "3", f"https://{_REGISTRY}/token?scope=repository:{repo}:pull"]
    # Anonymous pull works for public packages; use the workflow token when
    # available so private packages and PR-restricted visibility also work.
    if os.environ.get("GHCR_TOKEN"):
        cmd += ["-u", f"{os.environ.get('GHCR_ACTOR', 'x')}:{os.environ['GHCR_TOKEN']}"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)["token"]


def _fetch_json(repo: str, url_path: str, accept: str) -> dict:
    token = _registry_token(repo)
    out = subprocess.run(
        [
            "curl",
            "-sfL",
            "--retry",
            "3",
            "-H",
            f"Authorization: Bearer {token}",
            "-H",
            f"Accept: {accept}",
            f"https://{_REGISTRY}/v2/{repo}/{url_path}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def _decompressor(media_type: str) -> list[str]:
    if media_type.endswith("+zstd"):
        return ["zstd", "-dc"]
    if media_type.endswith("+gzip") or media_type.endswith(".gzip"):
        return ["gzip", "-dc"]
    return ["cat"]


def _extract_layer(repo: str, digest: str, media_type: str, dest: str) -> list[str]:
    """Stream a layer blob into dest, returning the list of extracted paths."""
    token = _registry_token(repo)  # fresh per layer: registry tokens are short-lived
    url = f"https://{_REGISTRY}/v2/{repo}/blobs/{digest}"
    curl = subprocess.Popen(
        ["curl", "-sfL", "--retry", "3", "-H", f"Authorization: Bearer {token}", url],
        stdout=subprocess.PIPE,
    )
    dec = subprocess.Popen(_decompressor(media_type), stdin=curl.stdout, stdout=subprocess.PIPE)
    # Device nodes can't be created unprivileged; nothing secret lives in /dev.
    tar = subprocess.Popen(
        [
            "tar",
            "-xv",
            "-C",
            dest,
            "--no-same-owner",
            "--no-same-permissions",
            "--exclude",
            "dev/*",
        ],
        stdin=dec.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if curl.stdout is None or dec.stdout is None:
        raise RuntimeError("failed to open download pipeline")
    curl.stdout.close()
    dec.stdout.close()
    names_out, tar_err = tar.communicate()
    curl.wait()
    dec.wait()
    if curl.returncode != 0:
        raise RuntimeError(f"blob download failed for {digest}")
    if dec.returncode != 0:
        raise RuntimeError(f"decompression failed for {digest} ({media_type})")
    if tar.returncode != 0:
        print(f"  WARNING: tar reported errors for {digest}: {tar_err.strip()[:500]}")
    subprocess.run(["chmod", "-R", "u+rX", dest], check=False)
    return [n for n in names_out.splitlines() if n]


def _run_trufflehog(directory: str) -> list[dict]:
    # Trufflehog defaults --concurrency to the CPU count; on many-core hosts
    # that many workers on a multi-GB layer gets the process OOM-killed.
    concurrency = min(os.cpu_count() or 4, 16)
    result = subprocess.run(
        [
            "trufflehog",
            "filesystem",
            directory,
            "--json",
            "--no-update",
            f"--concurrency={concurrency}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"trufflehog failed: {result.stderr.strip()[:500]}")
    findings = []
    for line in result.stdout.splitlines():
        try:
            findings.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return findings


def _scan_config(repo: str, config_digest: str, rules: list[tuple[str, str]]) -> list[str]:
    config = _fetch_json(repo, f"blobs/{config_digest}", "application/octet-stream")
    failures = []
    entries = [("history", h.get("created_by", "")) for h in config.get("history", [])]
    entries += [("env", e) for e in config.get("config", {}).get("Env", [])]
    entries += [("label", f"{k}={v}") for k, v in (config.get("config", {}).get("Labels") or {}).items()]
    for where, text in entries:
        if _CONFIG_SECRET_RE.search(text) and not _allowed(rules, where, text):
            failures.append(f"config {where}: {text[:200]}")
    return failures


def _scan_layers(
    repo: str,
    manifest: dict,
    rules: list[tuple[str, str]],
    workdir: str,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    for i, layer in enumerate(manifest["layers"]):
        digest, media_type = layer["digest"], layer["mediaType"]
        print(f"--- layer {i} ({layer['size'] / 1e6:.0f} MB, {digest[:19]})")
        extract_dir = os.path.join(workdir, "extract")
        shutil.rmtree(extract_dir, ignore_errors=True)
        os.makedirs(extract_dir)
        try:
            names = _extract_layer(repo, digest, media_type, extract_dir)

            for name in names:
                # Whiteouts mark deletions; the deleted file's content was
                # already scanned in the layer that added it, but its name
                # should still trip the filename check.
                plain = name.replace("/.wh.", "/")
                if _SUSPICIOUS_NAME_RE.search(plain) and not _PUBLIC_KEY_RE.search(plain):
                    if not _allowed(rules, plain):
                        failures.append(f"layer {i}: credential-shaped file: {name}")

            for f in _run_trufflehog(extract_dir):
                meta = f.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {})
                path = str(meta.get("file", "")).removeprefix(extract_dir).lstrip("/")
                raw = f.get("Raw", "")
                detector = f.get("DetectorName", "?")
                desc = f"layer {i}: {detector} in {path}: {raw[:6]}..."
                if _allowed(rules, path, raw):
                    continue
                if f.get("Verified"):
                    failures.append(f"{desc} (VERIFIED against live service)")
                elif detector.lower() in _HIGH_SIGNAL_DETECTORS and not _PLACEHOLDER_RE.search(raw):
                    failures.append(desc)
                else:
                    warnings.append(desc)
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
    return failures, warnings


def main(image_name: str, arch: str, channel: str) -> int:
    repo = f"{_ORG}/{image_name}"
    tag = f"{channel}-{arch}"
    rules = _load_allowlist()
    print(f"Scanning {_REGISTRY}/{repo}:{tag}")

    manifest = _fetch_json(repo, f"manifests/{tag}", _MANIFEST_ACCEPT)
    # A per-arch tag is normally a plain manifest, but tolerate an index
    # wrapper (skip attestation entries, which report platform os "unknown").
    manifests = [manifest]
    if "layers" not in manifest:
        manifests = [
            _fetch_json(repo, f"manifests/{m['digest']}", m["mediaType"])
            for m in manifest.get("manifests", [])
            if m.get("platform", {}).get("os") != "unknown"
        ]

    failures: list[str] = []
    warnings: list[str] = []
    workdir = os.environ.get("SCAN_WORKDIR") or tempfile.mkdtemp(prefix="image-scan-")
    os.makedirs(workdir, exist_ok=True)
    try:
        for m in manifests:
            failures += _scan_config(repo, m["config"]["digest"], rules)
            layer_failures, layer_warnings = _scan_layers(repo, m, rules, workdir)
            failures += layer_failures
            warnings += layer_warnings
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if warnings:
        print(f"\n{len(warnings)} low-signal finding(s), not failing the scan:")
        for w in warnings:
            print(f"  WARN {w}")
    if failures:
        print(f"\n{len(failures)} finding(s) require attention:")
        for f in failures:
            print(f"  FAIL {f}")
        print(
            "\nIf a finding is a confirmed false positive, add a raw:<regex> or "
            f"path:<glob> rule to {_ALLOWLIST}. Otherwise rotate the credential: "
            "pushed layers are permanently pullable from the registry."
        )
        return 1
    print("\nNo exposed secrets found.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <image_name> <arch> <channel>", file=sys.stderr)
        sys.exit(2)
    image_arg, arch_arg, channel_arg = sys.argv[1:4]
    if arch_arg not in ("arm64", "amd64"):
        print(f"Unsupported arch '{arch_arg}' (expected arm64 or amd64)", file=sys.stderr)
        sys.exit(2)
    if not _CHANNEL_RE.match(channel_arg):
        print(f"Unsupported channel '{channel_arg}' (expected 'latest' or 'pr-<number>')", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(image_arg, arch_arg, channel_arg))
