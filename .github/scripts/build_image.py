#!/usr/bin/env python3

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from textwrap import dedent

import firecrest as f7t

_CAPSTOR_IMAGES = "/capstor/store/cscs/swissai/infra01/container-images/ci"
_POLL_INTERVAL = 60
_TIMEOUT = 4 * 3600
_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
}


def _build_slurm_script(
    image_name: str,
    arch: str,
    account: str,
    partition: str,
    reservation: str | None,
    remote_logs_dir: str,
    output_sqsh: str,
    ghcr_token: str,
    ghcr_actor: str,
) -> str:
    reservation_line = f"#SBATCH --reservation={reservation}" if reservation else ""
    # Push to an arch-specific tag; a later merge step combines the per-arch
    # tags into a single multi-arch manifest list under ":latest".
    ghcr_image = f"ghcr.io/swiss-ai/{image_name}:latest-{arch}"
    return dedent(
        f"""
        #!/bin/bash
        #SBATCH --job-name=build-{image_name}-{arch}
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

        # Container storage must live on node-local disk. $HOME is a network
        # filesystem (Lustre/GPFS): the overlay driver's per-layer xattrs fail
        # there with "lsetxattr ... operation not supported", and podman warns
        # it is an unsupported backing store. Point graphroot/runroot at the
        # node-local $TMPDIR instead (same local store used for XDG_RUNTIME_DIR).
        #
        # ignore_chown_errors: the CI user has no /etc/subuid range, so rootless
        # podman uses a single-UID namespace and cannot chown files owned by a
        # non-zero GID (e.g. /etc/gshadow is root:shadow = 0:42) while unpacking
        # a base image; this skips those chowns instead of aborting the unpack.
        export PODMAN_STORE="${{TMPDIR:-/tmp}}/podman-store-$$"
        mkdir -p "${{PODMAN_STORE}}"
        export CONTAINERS_STORAGE_CONF="${{XDG_RUNTIME_DIR}}/storage.conf"
        cat > "${{CONTAINERS_STORAGE_CONF}}" <<STORAGE_CONF
        [storage]
        driver = "overlay"
        graphroot = "${{PODMAN_STORE}}/root"
        runroot = "${{PODMAN_STORE}}/runroot"
        [storage.options.overlay]
        ignore_chown_errors = "true"
        STORAGE_CONF

        IMAGE_TAG="{image_name}-{arch}:${{SLURM_JOB_ID}}"
        SCRATCH_SQSH="${{SCRATCH}}/{image_name}-{arch}.sqsh"

        cleanup() {{
            podman logout ghcr.io 2>/dev/null || true
            podman rmi "${{IMAGE_TAG}}" 2>/dev/null || true
            rm -f "${{SCRATCH_SQSH}}" 2>/dev/null || true
            rm -rf "${{XDG_RUNTIME_DIR}}" "${{PODMAN_STORE}}" 2>/dev/null || true
        }}
        trap cleanup EXIT

        # Log in before building: some images use a private ghcr.io base
        # (e.g. vllm_cxi's `FROM ghcr.io/swiss-ai/vllm_cuda13`), and the base
        # pull during `podman build` must be authenticated. Public-base images
        # are unaffected by an early login.
        echo "=== Logging in to GHCR ==="
        echo "{ghcr_token}" | podman login ghcr.io -u "{ghcr_actor}" --password-stdin

        echo "=== Building {image_name} on $(hostname) at $(date) ==="
        podman build -t "${{IMAGE_TAG}}" .

        echo "=== Pushing to GHCR ==="
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


async def main(image_name: str, arch: str) -> int:
    # Shared across both clusters.
    client_id = os.environ["SML_FIRECREST_CLIENT_ID"]
    client_secret = os.environ["SML_FIRECREST_CLIENT_SECRET"]
    token_uri = os.environ["SML_FIRECREST_TOKEN_URI"]

    # Per-arch: different cluster reached via a different FireCREST endpoint.
    firecrest_url = _arch_env("SML_FIRECREST_URL", arch)
    system_name = _arch_env("SML_SYSTEM", arch)
    partition = _arch_env("SML_PARTITION", arch)
    reservation = _arch_env("SML_RESERVATION", arch)
    missing = [
        name
        for name, val in (
            ("SML_FIRECREST_URL", firecrest_url),
            ("SML_SYSTEM", system_name),
            ("SML_PARTITION", partition),
        )
        if not val
    ]
    if missing:
        print(
            f"Missing FireCREST config for arch '{arch}': {', '.join(missing)} "
            f"(set <VAR>_{arch.upper()} or the base <VAR>)",
            file=sys.stderr,
        )
        return 1

    ghcr_token = os.environ["GHCR_TOKEN"]
    ghcr_actor = os.environ["GHCR_ACTOR"]

    auth = f7t.ClientCredentialsAuth(client_id, client_secret, token_uri, min_token_validity=90)
    client = f7t.v2.AsyncFirecrest(firecrest_url, authorization=auth)

    user_info = await client.userinfo(system_name)
    username = user_info["user"]["name"]
    account = user_info["group"]["name"]

    # Arch-suffixed so concurrent arm64/amd64 builds don't clobber each other's
    # uploaded build context on a shared home filesystem.
    remote_build_dir = f"/users/{username}/.sml/image-builds/{image_name}-{arch}"
    remote_logs_dir = f"/users/{username}/.sml/image-builds/logs"

    print("Creating remote directories...")
    await client.mkdir(system_name, remote_build_dir, create_parents=True)
    await client.mkdir(system_name, remote_logs_dir, create_parents=True)

    local_image_dir = Path("images") / image_name
    print(f"Uploading {local_image_dir} -> {remote_build_dir}")
    for local_file in sorted(local_image_dir.iterdir()):
        if local_file.is_file():
            print(f"  {local_file.name}")
            await client.upload(
                system_name=system_name,
                local_file=str(local_file),
                directory=remote_build_dir,
                filename=local_file.name,
                account=account,
                blocking=True,
            )

    # Arch-suffixed: capstor is a shared store, so per-arch builds must not
    # write to the same path.
    output_sqsh = f"{_CAPSTOR_IMAGES}/{image_name}-{arch}.sqsh"

    script = _build_slurm_script(
        image_name=image_name,
        arch=arch,
        account=account,
        partition=partition,
        reservation=reservation,
        remote_logs_dir=remote_logs_dir,
        output_sqsh=output_sqsh,
        ghcr_token=ghcr_token,
        ghcr_actor=ghcr_actor,
    )

    print(f"Submitting SLURM job for {image_name}...")
    result = await client.submit(
        system_name=system_name,
        working_dir=remote_build_dir,
        script_str=script,
        account=account,
    )
    job_id = int(result["jobId"])
    print(f"Job ID: {job_id}")

    start = time.time()
    while time.time() - start < _TIMEOUT:
        await asyncio.sleep(_POLL_INTERVAL)
        info = await client.job_info(system_name=system_name, jobid=str(job_id))
        state = str(info[0]["status"]["state"])
        elapsed = int(time.time() - start)
        print(f"[{elapsed}s] Job {job_id}: {state}")

        if state == "COMPLETED":
            print(f"Build succeeded: {output_sqsh}")
            return 0

        if state in _TERMINAL_STATES:
            print(f"Build failed with state: {state}")
            await _print_logs(client, system_name, account, remote_logs_dir, job_id)
            return 1

    print(f"Timed out after {_TIMEOUT}s waiting for job {job_id}.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <image_name> [arch]", file=sys.stderr)
        sys.exit(1)
    image_arg = sys.argv[1]
    arch_arg = sys.argv[2] if len(sys.argv) == 3 else "arm64"
    if arch_arg not in ("arm64", "amd64"):
        print(f"Unsupported arch '{arch_arg}' (expected arm64 or amd64)", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(main(image_arg, arch_arg)))
