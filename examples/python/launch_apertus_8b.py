#!/usr/bin/env python3
"""Launch Apertus-8B using the SML Python API.

Equivalent to examples/swiss-ai/Apertus-8B-Instruct-2509-sglang.sh
"""

import asyncio
import getpass
import grp
import os

from swiss_ai_model_launch import LaunchArgs, SlurmLauncher


async def main() -> None:
    username = getpass.getuser()
    account = grp.getgrgid(os.getgid()).gr_name

    launcher = SlurmLauncher(
        system_name="local",
        username=username,
        account=account,
        partition="normal",
    )

    args = LaunchArgs(
        job_name=f"sml_apertus_8b_{username}",
        served_model_name=f"swiss-ai/Apertus-8B-Instruct-2509-{username}",
        account=account,
        partition="normal",
        environment="src/swiss_ai_model_launch/assets/envs/sglang.toml",
        framework="sglang",
        framework_args=(
            "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 "
            f"--served-model-name swiss-ai/Apertus-8B-Instruct-2509-{username} "
            "--host 0.0.0.0 "
            "--port 8080"
        ),
        time="02:00:00",
        worker_port=8080,
    )

    job_id, served = await launcher.launch_with_args(args)
    print(f"Job submitted: {job_id}")
    print(f"Served model name: {served}")


if __name__ == "__main__":
    asyncio.run(main())
