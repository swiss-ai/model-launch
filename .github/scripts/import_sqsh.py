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
_POLL_INTERVAL = 30
_TIMEOUT = 30 * 60
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
    account: str,
    partition: str,
    reservation: str | None,
    remote_logs_dir: str,
    output_sqsh: str,
    ghcr_token: str,
    ghcr_actor: str,
) -> str:
    reservation_line = f"#SBATCH --reservation={reservation}" if reservation else ""
    ghcr_image = f"ghcr.io/swiss-ai/{image_name}:latest"
    return dedent(f"""
        #!/bin/bash
        #SBATCH --job-name=sqsh-{image_name}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=8
        #SBATCH --time=00:30:00
        #SBATCH --account={account}
        #SBATCH --partition={partition}
        {reservation_line}
        #SBATCH --output={remote_logs_dir}/%j.out
        #SBATCH --error={remote_logs_dir}/%j.err

        set -euo pipefail

        export DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null
        export XDG_RUNTIME_DIR="${{TMPDIR:-/tmp}}/podman-runtime-$$"
        mkdir -p "${{XDG_RUNTIME_DIR}}"

        SCRATCH_SQSH="${{SCRATCH}}/{image_name}.sqsh"

        cleanup() {{
            podman logout ghcr.io 2>/dev/null || true
            podman rmi "{ghcr_image}" 2>/dev/null || true
            rm -f "${{SCRATCH_SQSH}}" 2>/dev/null || true
            rm -rf "${{XDG_RUNTIME_DIR}}" 2>/dev/null || true
        }}
        trap cleanup EXIT

        echo "=== Logging in to GHCR ==="
        echo "{ghcr_token}" | podman login ghcr.io -u "{ghcr_actor}" --password-stdin

        echo "=== Pulling {ghcr_image} on $(hostname) at $(date) ==="
        podman pull "{ghcr_image}"

        echo "=== Converting to sqsh ==="
        rm -f "${{SCRATCH_SQSH}}"
        enroot import -o "${{SCRATCH_SQSH}}" "podman://{ghcr_image}" || true
        if [ ! -s "${{SCRATCH_SQSH}}" ]; then
            echo "ERROR: enroot import produced no output"
            exit 1
        fi

        echo "=== Saving to capstor ==="
        mkdir -p "$(dirname "{output_sqsh}")"
        cp "${{SCRATCH_SQSH}}" "{output_sqsh}.tmp"
        mv "{output_sqsh}.tmp" "{output_sqsh}"
        chmod o+rx "{output_sqsh}"

        echo "=== Done: {ghcr_image} -> {output_sqsh} at $(date) ==="
    """).lstrip("\n")


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


async def main(image_name: str) -> int:
    client_id = os.environ["SML_FIRECREST_CLIENT_ID"]
    client_secret = os.environ["SML_FIRECREST_CLIENT_SECRET"]
    token_uri = os.environ["SML_FIRECREST_TOKEN_URI"]
    firecrest_url = os.environ["SML_FIRECREST_URL"]
    system_name = os.environ["SML_FIRECREST_SYSTEM"]
    partition = os.environ["SML_PARTITION"]
    reservation = os.environ.get("SML_RESERVATION")

    ghcr_token = os.environ["GHCR_TOKEN"]
    ghcr_actor = os.environ["GHCR_ACTOR"]

    client = f7t.v2.AsyncFirecrest(
        firecrest_url=firecrest_url,
        authorization=f7t.ClientCredentialsAuth(
            client_id=client_id,
            client_secret=client_secret,
            token_uri=token_uri,
            min_token_validity=90,
        ),
    )

    user_info = await client.userinfo(system_name)
    username = user_info["user"]["name"]
    account = user_info["group"]["name"]

    remote_work_dir = f"/users/{username}/.sml/sqsh-imports/{image_name}"
    remote_logs_dir = f"/users/{username}/.sml/sqsh-imports/logs"

    print("Creating remote directories...")
    await client.mkdir(system_name, remote_work_dir, create_parents=True)
    await client.mkdir(system_name, remote_logs_dir, create_parents=True)

    output_sqsh = f"{_CAPSTOR_IMAGES}/{image_name}.sqsh"

    script = _build_slurm_script(
        image_name=image_name,
        account=account,
        partition=partition,
        reservation=reservation,
        remote_logs_dir=remote_logs_dir,
        output_sqsh=output_sqsh,
        ghcr_token=ghcr_token,
        ghcr_actor=ghcr_actor,
    )

    print(f"Submitting SLURM job for {image_name} sqsh import...")
    result = await client.submit(
        system_name=system_name,
        working_dir=remote_work_dir,
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
            print(f"sqsh ready: {output_sqsh}")
            return 0

        if state in _TERMINAL_STATES:
            print(f"Import failed with state: {state}")
            await _print_logs(client, system_name, account, remote_logs_dir, job_id)
            return 1

    print(f"Timed out after {_TIMEOUT}s waiting for job {job_id}.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image_name>", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(main(sys.argv[1])))
