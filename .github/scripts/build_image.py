#!/usr/bin/env python3
"""Submit a SLURM image-build job via FirecREST and wait for completion."""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import firecrest as f7t

_CAPSTOR_IMAGES = "/capstor/store/cscs/swissai/infra01/container-images"
_POLL_INTERVAL = 60  # seconds
_TIMEOUT = 4 * 3600  # 4 hours
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
    remote_logs_dir: str,
    output_sqsh: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=build-{image_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --output={remote_logs_dir}/%j.out
#SBATCH --error={remote_logs_dir}/%j.err

set -euo pipefail

IMAGE_TAG="{image_name}:${{SLURM_JOB_ID}}"
SCRATCH_SQSH="${{SCRATCH}}/{image_name}.sqsh"

cleanup() {{
    podman rmi "${{IMAGE_TAG}}" 2>/dev/null || true
    rm -f "${{SCRATCH_SQSH}}" 2>/dev/null || true
}}
trap cleanup EXIT

echo "=== Building {image_name} on $(hostname) at $(date) ==="
podman build -t "${{IMAGE_TAG}}" .

echo "=== Converting to sqsh ==="
enroot import -o "${{SCRATCH_SQSH}}" "podman://${{IMAGE_TAG}}"

echo "=== Saving to capstor ==="
mkdir -p "$(dirname "{output_sqsh}")"
cp "${{SCRATCH_SQSH}}" "{output_sqsh}"

echo "=== Done: {image_name} -> {output_sqsh} at $(date) ==="
"""


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

    auth = f7t.ClientCredentialsAuth(client_id, client_secret, token_uri)
    client = f7t.v2.AsyncFirecrest(firecrest_url, authorization=auth)

    user_info = await client.userinfo(system_name)
    username = user_info["user"]["name"]
    account = user_info["group"]["name"]

    remote_build_dir = f"/users/{username}/.sml/image-builds/{image_name}"
    remote_logs_dir = f"/users/{username}/.sml/image-builds/logs"

    # Create remote directories
    print("Creating remote directories...")
    await client.mkdir(system_name, remote_build_dir, create_parents=True)
    await client.mkdir(system_name, remote_logs_dir, create_parents=True)

    # Upload all files in the image directory
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

    pr_number = os.environ.get("PR_NUMBER")
    if pr_number:
        output_sqsh = f"{_CAPSTOR_IMAGES}/pr-{pr_number}/{image_name}.sqsh"
        await client.mkdir(
            system_name, f"{_CAPSTOR_IMAGES}/pr-{pr_number}", create_parents=True
        )
        print(f"PR build — staging path: {output_sqsh}")
    else:
        output_sqsh = f"{_CAPSTOR_IMAGES}/{image_name}.sqsh"

    script = _build_slurm_script(
        image_name=image_name,
        account=account,
        partition=partition,
        remote_logs_dir=remote_logs_dir,
        output_sqsh=output_sqsh,
    )

    # Submit job
    print(f"Submitting SLURM job for {image_name}...")
    result = await client.submit(
        system_name=system_name,
        working_dir=remote_build_dir,
        script_str=script,
        account=account,
    )
    job_id = int(result["jobId"])
    print(f"Job ID: {job_id}")

    # Poll until terminal state
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
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image_name>", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(main(sys.argv[1])))
