#!/usr/bin/env python3
"""Remove the staging sqsh directory for a closed PR via a SLURM job."""

import asyncio
import os
import sys
import time

import firecrest as f7t

_CAPSTOR_IMAGES = "/capstor/store/cscs/swissai/infra01/container-images"
_POLL_INTERVAL = 30  # seconds
_TIMEOUT = 10 * 60  # 10 minutes — rm is fast
_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
}


async def main(pr_number: str) -> int:
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

    staging_dir = f"{_CAPSTOR_IMAGES}/pr-{pr_number}"
    working_dir = f"/users/{username}/.sml/image-builds"

    script = f"""#!/bin/bash
#SBATCH --job-name=cleanup-pr-{pr_number}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --output={working_dir}/logs/%j.out
#SBATCH --error={working_dir}/logs/%j.err

set -euo pipefail

if [ -d "{staging_dir}" ]; then
    rm -rf "{staging_dir}"
    echo "Removed {staging_dir}"
else
    echo "Nothing to clean up: {staging_dir} does not exist"
fi
"""

    print(f"Submitting cleanup job for PR #{pr_number} ({staging_dir})...")
    result = await client.submit(
        system_name=system_name,
        working_dir=working_dir,
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
        print(f"[{int(time.time() - start)}s] Job {job_id}: {state}")

        if state == "COMPLETED":
            print("Cleanup complete.")
            return 0

        if state in _TERMINAL_STATES:
            print(f"Cleanup job failed with state: {state}")
            return 1

    print(f"Timed out after {_TIMEOUT}s waiting for job {job_id}.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pr_number>", file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(main(sys.argv[1])))
