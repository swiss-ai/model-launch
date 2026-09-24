#!/usr/bin/env python3

import asyncio
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from textwrap import dedent

import firecrest as f7t

from swiss_ai_model_launch.launchers.firecrest_auth import build_client_from_env
from swiss_ai_model_launch.launchers.firecrest_launcher import _primary_group_name
from swiss_ai_model_launch.launchers.utils import call_with_firecrest_retry

_CAPSTOR_IMAGES = "/capstor/store/cscs/swissai/infra01/container-images/ci"
_RELEASE_CHANNEL = "latest"
# Anything outside this set lands in a filesystem path and a registry tag, so
# it must not contain path separators or shell metacharacters.
_CHANNEL_RE = re.compile(r"^(latest|pr-\d+)$")
_POLL_INTERVAL = 60
_TIMEOUT = 4 * 3600
# A FirecREST error says nothing about the SLURM job: 2026-09-24 its health checks
# timed out for hours while jobs ran on. Submits are adopted by job name; status polls
# ride out this much continuous failure, then the workflow fails *cheaply* (runner
# minutes) with the job left running — a re-run adopts it by name.
_ADOPT_WAIT = 120
_POLL_FAILURE_BUDGET = 5 * 60
_LIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
}


def sqsh_path(image_name: str, arch: str, channel: str) -> str:
    """Where a build's squashfs lands on the shared capstor store.

    The release channel keeps its historical flat path, because the env TOMLs
    and other consumers point straight at it. Pre-release channels get their
    own subdirectory so a PR build can never overwrite what main published.
    """
    if channel == _RELEASE_CHANNEL:
        return f"{_CAPSTOR_IMAGES}/{image_name}-{arch}.sqsh"
    return f"{_CAPSTOR_IMAGES}/{channel}/{image_name}-{arch}.sqsh"


def _build_slurm_script(
    image_name: str,
    arch: str,
    channel: str,
    account: str,
    partition: str,
    reservation: str | None,
    remote_logs_dir: str,
    output_sqsh: str,
    ghcr_token: str,
    ghcr_actor: str,
    dispatch_repo: str = "",
) -> str:
    reservation_line = f"#SBATCH --reservation={reservation}" if reservation else ""
    # Push to a channel- and arch-specific tag; a later merge step combines the
    # per-arch tags into a single multi-arch manifest list under ":<channel>".
    ghcr_image = f"ghcr.io/swiss-ai/{image_name}:{channel}-{arch}"
    return dedent(
        f"""
        #!/bin/bash
        #SBATCH --job-name=build-{image_name}-{arch}-{channel}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=64
        #SBATCH --time=04:00:00
        #SBATCH --account={account}
        #SBATCH --partition={partition}
        {reservation_line}
        #SBATCH --output={remote_logs_dir}/%j.out
        #SBATCH --error={remote_logs_dir}/%j.err

        set -euo pipefail

        # Batch nodes have no D-Bus session and /run/user/<uid> doesn't exist.
        # Point podman's runtime dir to a writable temp location.
        export DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null
        export XDG_RUNTIME_DIR="${{TMPDIR:-/tmp}}/podman-runtime-$$"
        mkdir -p "${{XDG_RUNTIME_DIR}}"

        # Rootless podman ignores graphroot/runroot from /etc/containers/storage.conf
        # and falls back to $HOME/.local/share/containers/storage. Home is NFS, which
        # has no user xattrs, so even pulling the base image dies with
        # "lsetxattr ...: operation not supported". Personal accounts have a
        # ~/.config/containers/storage.conf pointing at tmpfs; the CI service account
        # has none, so the job writes its own. tmpfs has no user xattrs either below
        # kernel 6.6, hence fuse-overlayfs rather than the kernel overlay driver.
        PODMAN_STORAGE="/dev/shm/${{USER}}/podman-${{SLURM_JOB_ID}}"
        mkdir -p "${{PODMAN_STORAGE}}/root" "${{PODMAN_STORAGE}}/runroot"

        FUSE_OVERLAYFS=""
        for candidate in \
            /usr/local/vs-ce-podman/fuse-overlayfs \
            /usr/bin/fuse-overlayfs-1.13 \
            "$(command -v fuse-overlayfs || true)"; do
            if [ -x "${{candidate}}" ]; then
                FUSE_OVERLAYFS="${{candidate}}"
                break
            fi
        done
        if [ -z "${{FUSE_OVERLAYFS}}" ]; then
            echo "ERROR: no fuse-overlayfs on $(hostname); podman storage on tmpfs needs it"
            exit 1
        fi

        # Kept outside PODMAN_STORAGE: the cleanup below still needs a valid
        # config to tear that directory down.
        export CONTAINERS_STORAGE_CONF="${{XDG_RUNTIME_DIR}}/storage.conf"
        cat > "${{CONTAINERS_STORAGE_CONF}}" <<EOF
        [storage]
        driver = "overlay"
        graphroot = "${{PODMAN_STORAGE}}/root"
        runroot = "${{PODMAN_STORAGE}}/runroot"

        [storage.options.overlay]
        mount_program = "${{FUSE_OVERLAYFS}}"
        EOF

        # Seed the same config in this account's home, the way personal accounts
        # have it, so podman run outside this script (interactive debugging, a
        # future script that forgets the env var) also lands on tmpfs. The
        # per-job CONTAINERS_STORAGE_CONF above still wins for this build: two
        # builds can share a node, and a fixed home-level graphroot would have
        # them trampling each other's layers and cleanup.
        HOME_STORAGE_CONF="${{HOME}}/.config/containers/storage.conf"
        if [ ! -e "${{HOME_STORAGE_CONF}}" ]; then
            mkdir -p "$(dirname "${{HOME_STORAGE_CONF}}")"
            cat > "${{HOME_STORAGE_CONF}}" <<EOF
        [storage]
        driver = "overlay"
        graphroot = "/dev/shm/${{USER}}/root"
        runroot = "/dev/shm/${{USER}}/runroot"

        [storage.options.overlay]
        mount_program = "${{FUSE_OVERLAYFS}}"
        EOF
            echo "Seeded ${{HOME_STORAGE_CONF}}"
        fi

        IMAGE_TAG="{image_name}-{arch}-{channel}:${{SLURM_JOB_ID}}"
        SCRATCH_SQSH="${{SCRATCH}}/{image_name}-{arch}-{channel}.sqsh"

        cleanup() {{
            podman logout ghcr.io 2>/dev/null || true
            # Storage is job-scoped, so a full reset is safe and is the only
            # thing that reliably empties it: layers are owned by mapped subuids
            # and sit behind fuse mounts, so a plain rm hits "Permission denied"
            # / "Device or resource busy". Left behind they occupy the node's RAM.
            podman system reset --force 2>/dev/null || true
            rm -f "${{SCRATCH_SQSH}}" 2>/dev/null || true
            rm -rf "${{PODMAN_STORAGE}}" "${{XDG_RUNTIME_DIR}}" 2>/dev/null || true
        }}
        # Tell GitHub the build is over (success or failure) so image-builds.yml
        # runs on the event instead of polling. Best effort: a missed dispatch is
        # caught by that workflow's daily tick.
        notify() {{
          [ -n "{dispatch_repo}" ] || return 0
          payload='{{"event_type":"image-build-finished","client_payload":'
          payload="$payload"'{{"image":"{image_name}","arch":"{arch}","channel":"{channel}",'
          payload="$payload"'"job":"'"${{SLURM_JOB_ID}}"'"}}}}'
          curl -sS -m 30 -X POST "https://api.github.com/repos/{dispatch_repo}/dispatches" \
            -H "Authorization: Bearer {ghcr_token}" -H "Accept: application/vnd.github+json" \
            -d "$payload" \
            || echo "WARNING: repository_dispatch failed"
        }}
        trap 'cleanup; notify' EXIT

        echo "=== Building {image_name} on $(hostname) at $(date) ==="
        # --format docker: honor SHELL instructions (OCI format silently
        # ignores SHELL, so RUN steps needing bash/pipefail break under sh).
        # The OCI source label links the GHCR package to this repository, so the
        # package inherits the repo's permissions (the pushing account only needs
        # write on the repo, not on each org package).
        podman build --format docker -t "${{IMAGE_TAG}}" \
          --label org.opencontainers.image.source="https://github.com/{dispatch_repo or "swiss-ai/model-launch"}" \
          --label org.opencontainers.image.revision="{image_name}-{arch}-{channel}" .

        echo "=== Pushing to GHCR ==="
        echo "{ghcr_token}" | podman login ghcr.io -u "{ghcr_actor}" --password-stdin
        podman push "${{IMAGE_TAG}}" "{ghcr_image}"

        echo "=== Converting to sqsh ==="
        rm -f "${{SCRATCH_SQSH}}"
        enroot import -o "${{SCRATCH_SQSH}}" "podman://${{IMAGE_TAG}}" || true
        if [ ! -s "${{SCRATCH_SQSH}}" ]; then
            echo "ERROR: enroot import produced no output"
            exit 1
        fi

        echo "=== Saving to capstor ==="
        mkdir -p "$(dirname "{output_sqsh}")"
        chmod o+rx "$(dirname "{output_sqsh}")"
        cp "${{SCRATCH_SQSH}}" "{output_sqsh}.tmp"
        mv "{output_sqsh}.tmp" "{output_sqsh}"
        chmod o+rx "{output_sqsh}"

        echo "=== Done: {image_name} -> {output_sqsh} at $(date) ==="
    """
    ).lstrip("\n")


async def _print_logs(
    client: f7t.v2.AsyncFirecrest,
    system_name: str,
    account: str,
    logs_dir: str,
    job_id: int,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("out", "err"):
            remote_path = f"{logs_dir}/{job_id}.{suffix}"
            local_path = Path(tmp) / f"{job_id}.{suffix}"
            try:
                await client.download(
                    system_name=system_name,
                    source_path=remote_path,
                    target_path=local_path,
                    account=account,
                    blocking=True,
                )
                content = local_path.read_text()
                if content.strip():
                    print(f"\n=== {suffix.upper()} ===\n{content}")
            except Exception as e:  # noqa: BLE001
                print(f"  Could not retrieve {suffix} log: {e}")


def _arch_env(base_key: str, arch: str) -> str | None:
    """Resolve a per-arch FireCREST setting.

    arm64 (the original Grace cluster) uses the base var, e.g. SML_FIRECREST_URL.
    Other arches use a strictly arch-suffixed var, e.g. SML_FIRECREST_URL_AMD64,
    with NO fallback to the base var — falling back would silently build on the
    wrong cluster. Credentials and token URI are shared across clusters.
    Reservation is optional per arch (e.g. the amd64 cluster has none).
    """
    if arch == "arm64":
        return os.environ.get(base_key)
    return os.environ.get(f"{base_key}_{arch.upper()}")


async def _find_live_job(client: f7t.v2.AsyncFirecrest, system_name: str, job_name: str) -> int | None:
    """This account's pending/running job with that name, if any (FirecREST's own `name`
    filter needs API >= 2.6, which CSCS does not run — same as the launcher's find_job)."""
    jobs = await call_with_firecrest_retry(lambda: client.job_info(system_name=system_name))
    for job in jobs:
        if job.get("name") != job_name:
            continue
        state = str((job.get("status") or {}).get("state", "")).split()[0].rstrip("+")
        if state in _LIVE_STATES:
            return int(job["jobId"])
    return None


class _Site:
    """One arch's cluster: FirecREST client, system, account and the remote dirs."""

    def __init__(self, arch: str):
        self.arch = arch
        self.firecrest_url = _arch_env("SML_FIRECREST_URL", arch)
        self.system_name = _arch_env("SML_SYSTEM", arch)
        self.partition = _arch_env("SML_PARTITION", arch)
        self.reservation = _arch_env("SML_RESERVATION", arch)
        missing = [
            name
            for name, val in (
                ("SML_FIRECREST_URL", self.firecrest_url),
                ("SML_SYSTEM", self.system_name),
                ("SML_PARTITION", self.partition),
            )
            if not val
        ]
        if missing:
            raise SystemExit(
                f"Missing FireCREST config for arch '{arch}': {', '.join(missing)} "
                f"(set <VAR>_{arch.upper()} or the base <VAR>)"
            )
        # Credentials (service-account API key, or client ID/secret) are shared
        # across both clusters and read from the environment.
        self.client = build_client_from_env(self.firecrest_url)
        self.username = ""
        self.account = ""

    async def connect(self) -> "_Site":
        user_info = await call_with_firecrest_retry(lambda: self.client.userinfo(self.system_name))
        self.username = user_info["user"]["name"]
        # FirecREST 2.6.0+ dropped the top-level `group` (#221 fixed the launcher; the CI
        # build died on the same KeyError for PR #230's vllm_0.30.0 image, 2026-09-24)
        self.account = _primary_group_name(user_info)
        return self

    @property
    def logs_dir(self) -> str:
        return f"/users/{self.username}/.sml/image-builds/logs"

    def build_dir(self, image_name: str, channel: str) -> str:
        # Arch- and channel-suffixed so concurrent builds (arm64/amd64, main/PR)
        # don't clobber each other's uploaded build context on a shared home
        # filesystem.
        return f"/users/{self.username}/.sml/image-builds/{image_name}-{self.arch}-{channel}"


async def submit_build(site: _Site, image_name: str, channel: str) -> tuple[int, bool]:
    """Adopt the live build with this name or submit one. Returns (job id, adopted)."""
    job_name = f"build-{image_name}-{site.arch}-{channel}"
    job_id = await _find_live_job(site.client, site.system_name, job_name)
    if job_id is not None:
        # A previous workflow attempt (or a re-run after a FirecREST error) already has
        # this build running: adopt it rather than build it twice.
        print(f"Job {job_id} ({job_name}) is already live; adopting it")
        return job_id, True

    remote_build_dir = site.build_dir(image_name, channel)
    print("Creating remote directories...")
    await site.client.mkdir(site.system_name, remote_build_dir, create_parents=True)
    await site.client.mkdir(site.system_name, site.logs_dir, create_parents=True)

    local_image_dir = Path("images") / image_name
    print(f"Uploading {local_image_dir} -> {remote_build_dir}")
    for local_file in sorted(local_image_dir.iterdir()):
        if local_file.is_file():
            print(f"  {local_file.name}")
            # The build dir is reused across submits and FirecREST's upload does not
            # truncate an existing file: a shorter Dockerfile kept the old tail
            # (" /opt" became a 13th step, vllm_0.30.0 pr-234, 2026-09-24). Remove first.
            try:
                await site.client.rm(system_name=site.system_name, path=f"{remote_build_dir}/{local_file.name}")
            except Exception as rm_exc:  # noqa: BLE001 — not there yet, usually
                print(f"  (no previous {local_file.name} to remove: {type(rm_exc).__name__})")
            await site.client.upload(
                system_name=site.system_name,
                local_file=str(local_file),
                directory=remote_build_dir,
                filename=local_file.name,
                account=site.account,
                blocking=True,
            )

    script = _build_slurm_script(
        image_name=image_name,
        arch=site.arch,
        channel=channel,
        account=site.account,
        partition=site.partition,
        reservation=site.reservation,
        remote_logs_dir=site.logs_dir,
        # Arch-suffixed: capstor is a shared store, so per-arch builds must not
        # write to the same path.
        output_sqsh=sqsh_path(image_name, site.arch, channel),
        # The push (and the dispatch) happen hours after the runner is gone: GITHUB_TOKEN
        # dies with the job, so the asynchronous mode needs a long-lived GHCR_PUSH_TOKEN
        # (classic PAT: write:packages + repo). The synchronous `build` mode may still use
        # the job's own token — the runner is alive until the push.
        ghcr_token=os.environ.get("GHCR_PUSH_TOKEN") or os.environ["GHCR_TOKEN"],
        ghcr_actor=os.environ["GHCR_ACTOR"],
        dispatch_repo=os.environ.get("GITHUB_REPOSITORY", "") if os.environ.get("GHCR_PUSH_TOKEN") else "",
    )
    print(f"Submitting SLURM job for {image_name}...")
    try:
        result = await site.client.submit(
            system_name=site.system_name,
            working_dir=remote_build_dir,
            script_str=script,
            account=site.account,
        )
        return int(result["jobId"]), False
    except Exception as exc:  # noqa: BLE001 — the sbatch may have run regardless
        print(f"Submit errored ({type(exc).__name__}: {str(exc)[:200]}); looking for {job_name}")
        deadline = time.time() + _ADOPT_WAIT
        while time.time() < deadline:
            await asyncio.sleep(10)
            try:
                job_id = await _find_live_job(site.client, site.system_name, job_name)
            except Exception as look_exc:  # noqa: BLE001 — FirecREST still flapping
                print(f"  look-up failed ({type(look_exc).__name__}); retrying")
                continue
            if job_id is not None:
                print(f"Submit had gone through: adopting job {job_id}")
                return job_id, True
        raise


async def wait_build(site: _Site, image_name: str, channel: str, job_id: int) -> int:
    """Poll the job to its end (the synchronous `build` mode: manual runs, local use)."""
    output_sqsh = sqsh_path(image_name, site.arch, channel)
    start = time.time()
    failing_since: float | None = None
    while time.time() - start < _TIMEOUT:
        await asyncio.sleep(_POLL_INTERVAL)
        elapsed = int(time.time() - start)
        try:
            info = await call_with_firecrest_retry(
                lambda: site.client.job_info(system_name=site.system_name, jobid=str(job_id))
            )
            state = str(info[0]["status"]["state"])
        except Exception as exc:  # noqa: BLE001 — the job is still running or done; keep polling
            failing_since = failing_since or time.time()
            print(f"[{elapsed}s] Job {job_id}: status poll failed ({type(exc).__name__}), retrying")
            if time.time() - failing_since > _POLL_FAILURE_BUDGET:
                print(
                    f"FirecREST has not answered for {_POLL_FAILURE_BUDGET}s; job {job_id} may "
                    f"still be running — re-run this workflow to adopt it (state unknown)"
                )
                return 1
            continue
        failing_since = None
        print(f"[{elapsed}s] Job {job_id}: {state}")
        if state == "COMPLETED":
            print(f"Build succeeded: {output_sqsh}")
            return 0
        if state in _TERMINAL_STATES:
            print(f"Build failed with state: {state}")
            await _print_logs(site.client, site.system_name, site.account, site.logs_dir, job_id)
            return 1
    print(f"Timed out after {_TIMEOUT}s waiting for job {job_id}.")
    return 1


async def main(image_name: str, arch: str, channel: str) -> int:
    """`build`: submit (or adopt) and wait — the synchronous mode."""
    site = await _Site(arch).connect()
    job_id, _adopted = await submit_build(site, image_name, channel)
    print(f"Job ID: {job_id}")
    return await wait_build(site, image_name, channel, job_id)


# ---- asynchronous mode: `submit` registers a check run, `finish` (a scheduled workflow) -----
# completes it. The GitHub runner no longer waits hours on a SLURM job (Rob, 2026-09-24):
# the check run *is* the state — its external_id carries what `finish` needs.

_CHECK_PREFIX = "Image "
_GITHUB_API = "https://api.github.com"
_STALE_AFTER = 6 * 3600


def _gh(method: str, path: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(  # noqa: S310 — api.github.com only
        f"{_GITHUB_API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — api.github.com only
        return json.loads(resp.read() or b"{}")


def _check_name(image_name: str, arch: str, channel: str) -> str:
    return f"{_CHECK_PREFIX}{image_name} ({arch}, {channel})"


async def submit_and_register(image_name: str, arch: str, channel: str, head_sha: str) -> int:
    """`submit`: adopt-or-submit the build, register an in_progress check run on `head_sha`
    whose external_id says which SLURM job on which system to finish, and exit."""
    repo = os.environ["GITHUB_REPOSITORY"]
    if not os.environ.get("GHCR_PUSH_TOKEN"):
        print(
            "GHCR_PUSH_TOKEN is not set: the build's GHCR push and its finished-dispatch run "
            "after this job's GITHUB_TOKEN has expired. Add the secret (classic PAT with "
            "write:packages + repo) before using the asynchronous mode.",
            file=sys.stderr,
        )
        return 1
    site = await _Site(arch).connect()
    name = _check_name(image_name, arch, channel)
    # an in_progress check for this build already exists (a re-run): nothing to submit
    for run in _list_checks(repo, head_sha):
        if run["name"] == name:
            print(f"check run {run['id']} for {name} already in progress; nothing to do")
            return 0
    job_id, adopted = await submit_build(site, image_name, channel)
    external = {"image": image_name, "arch": arch, "channel": channel, "job_id": job_id,
                "system": site.system_name, "logs_dir": site.logs_dir, "t0": int(time.time())}  # fmt: skip
    run = _gh(
        "POST",
        f"/repos/{repo}/check-runs",
        {
            "name": name,
            "head_sha": head_sha,
            "status": "in_progress",
            "external_id": json.dumps(external),
            "output": {
                "title": f"SLURM job {job_id} on {site.system_name}" + (" (adopted)" if adopted else ""),
                "summary": "Building; a scheduled workflow (image-builds.yml, every 15 min) "
                f"finishes this check. Output: `{sqsh_path(image_name, arch, channel)}`.",
            },
        },
    )
    print(f"Job ID: {job_id}; check run {run['id']} ({name}) registered on {head_sha[:7]}")
    return 0


def _list_checks(repo: str, sha: str) -> list[dict]:
    data = _gh("GET", f"/repos/{repo}/commits/{sha}/check-runs?status=in_progress&per_page=100")
    return [r for r in data.get("check_runs", []) if r["name"].startswith(_CHECK_PREFIX)]


def _heads(repo: str) -> list[str]:
    """Commits that may carry pending image checks: open PR heads and recent main."""
    shas = [pr["head"]["sha"] for pr in _gh("GET", f"/repos/{repo}/pulls?state=open&per_page=100")]
    shas += [c["sha"] for c in _gh("GET", f"/repos/{repo}/commits?sha=main&per_page=10")]
    return list(dict.fromkeys(shas))


async def finish_pending() -> list[dict]:
    """`finish`: complete every in_progress image check whose SLURM job has ended. Returns
    the (image, channel, sha) triples whose arches are now all successful — the scan and
    manifest jobs take it from there."""
    repo = os.environ["GITHUB_REPOSITORY"]
    sites: dict[str, _Site] = {}
    done: dict[tuple, dict[str, str]] = {}
    for sha in _heads(repo):
        checks = _list_checks(repo, sha)
        for run in checks:
            try:
                ext = json.loads(run.get("external_id") or "{}")
                arch, job_id = ext["arch"], int(ext["job_id"])
            except (ValueError, KeyError):
                print(f"check run {run['id']} ({run['name']}) has no usable external_id; skipping")
                continue
            key = (ext["image"], ext["channel"], sha)
            if arch not in sites:
                sites[arch] = await _Site(arch).connect()
            site = sites[arch]
            try:
                info = await call_with_firecrest_retry(
                    lambda s=site, j=job_id: s.client.job_info(system_name=s.system_name, jobid=str(j))
                )
                state = str(info[0]["status"]["state"])
            except Exception as exc:  # noqa: BLE001 — FirecREST down: try again next tick
                print(f"{run['name']}: status poll failed ({type(exc).__name__}); next tick")
                if time.time() - int(ext.get("t0", time.time())) > _STALE_AFTER:
                    _complete(repo, run["id"], "timed_out", f"no answer about job {job_id} for 6 h")
                continue
            print(f"{run['name']}: job {job_id} {state}")
            if state == "COMPLETED":
                _complete(
                    repo,
                    run["id"],
                    "success",
                    f"job {job_id} COMPLETED — `{sqsh_path(ext['image'], arch, ext['channel'])}`",
                )
                done.setdefault(key, {})[arch] = "success"
            elif state in _TERMINAL_STATES:
                tail = await _log_tail(site, job_id)
                _complete(repo, run["id"], "failure", f"job {job_id} {state}", tail)
                done.setdefault(key, {})[arch] = "failure"
            elif time.time() - int(ext.get("t0", time.time())) > _STALE_AFTER:
                _complete(repo, run["id"], "timed_out", f"job {job_id} still {state} after 6 h")
                done.setdefault(key, {})[arch] = "failure"
        # arches that completed on an earlier tick count too
        for run in _gh("GET", f"/repos/{repo}/commits/{sha}/check-runs?status=completed&per_page=100").get(
            "check_runs", []
        ):
            if run["name"].startswith(_CHECK_PREFIX) and run.get("conclusion") == "success":
                try:
                    ext = json.loads(run.get("external_id") or "{}")
                    key = (ext["image"], ext["channel"], sha)
                    done.setdefault(key, {}).setdefault(ext["arch"], "success")
                except (ValueError, KeyError):
                    pass
    ready = [
        {"image": image, "channel": channel, "sha": sha}
        for (image, channel, sha), arches in done.items()
        if all(arches.get(a) == "success" for a in ("arm64", "amd64")) and not _manifest_done(repo, sha, image, channel)
    ]
    return ready


def _manifest_done(repo: str, sha: str, image: str, channel: str) -> bool:
    name = f"{_CHECK_PREFIX}{image} manifest ({channel})"
    for run in _gh("GET", f"/repos/{repo}/commits/{sha}/check-runs?check_name={urllib.request.quote(name)}").get(
        "check_runs", []
    ):
        if run["status"] == "completed" and run["conclusion"] == "success":
            return True
    return False


def _complete(repo: str, run_id: int, conclusion: str, summary: str, text: str = "") -> None:
    _gh(
        "PATCH",
        f"/repos/{repo}/check-runs/{run_id}",
        {
            "status": "completed",
            "conclusion": conclusion,
            "output": {"title": summary[:120], "summary": summary, "text": text[-60000:]},
        },  # fmt: skip
    )


async def _log_tail(site: _Site, job_id: int) -> str:
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for suffix in ("out", "err"):
            local_path = Path(tmp) / f"{job_id}.{suffix}"
            try:
                await site.client.download(
                    system_name=site.system_name,
                    source_path=f"{site.logs_dir}/{job_id}.{suffix}",
                    target_path=local_path,
                    account=site.account,
                    blocking=True,
                )
                parts.append(f"=== {suffix.upper()} ===\n{local_path.read_text()[-20000:]}")
            except Exception as e:  # noqa: BLE001
                parts.append(f"(could not retrieve {suffix} log: {e})")
    return "\n\n".join(parts)


def _write_output(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"{name}={value}\n")
    print(f"{name}={value}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    mode = "build"
    if argv and argv[0] in ("build", "submit", "finish"):
        mode, argv = argv[0], argv[1:]
    if mode == "finish":
        ready = asyncio.run(finish_pending())
        _write_output("ready", json.dumps(ready))
        _write_output("any", "true" if ready else "false")
        sys.exit(0)
    if len(argv) not in (1, 2, 3):
        print(f"Usage: {sys.argv[0]} [build|submit] <image_name> [arch] [channel]  |  finish", file=sys.stderr)
        sys.exit(1)
    image_arg = argv[0]
    arch_arg = argv[1] if len(argv) >= 2 else "arm64"
    channel_arg = argv[2] if len(argv) == 3 else _RELEASE_CHANNEL
    if arch_arg not in ("arm64", "amd64"):
        print(f"Unsupported arch '{arch_arg}' (expected arm64 or amd64)", file=sys.stderr)
        sys.exit(1)
    if not _CHANNEL_RE.match(channel_arg):
        print(f"Unsupported channel '{channel_arg}' (expected 'latest' or 'pr-<number>')", file=sys.stderr)
        sys.exit(1)
    if mode == "submit":
        head = os.environ.get("IMAGE_BUILD_SHA") or os.environ["GITHUB_SHA"]
        sys.exit(asyncio.run(submit_and_register(image_arg, arch_arg, channel_arg, head)))
    sys.exit(asyncio.run(main(image_arg, arch_arg, channel_arg)))
