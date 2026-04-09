#!/usr/bin/env python3
"""Launch multiple models concurrently using the SML Python API.

This is the use case where shell scripts fall short — you can launch
a batch of models in one shot without interactive prompts or TUI.
"""

import asyncio
import getpass
import grp
import os

from swiss_ai_model_launch import LaunchArgs, SlurmLauncher

MODELS = [
    {
        "name": "Apertus-8B-Instruct-2509",
        "vendor": "swiss-ai",
        "framework_args": "--dp-size 1",
    },
    {
        "name": "Apertus-70B-Instruct-2509",
        "vendor": "swiss-ai",
        "framework_args": "--dp-size 4",
        "nodes": 4,
    },
]


async def launch_model(launcher: SlurmLauncher, model: dict) -> tuple[str, int]:
    username = launcher.username
    vendor = model["vendor"]
    name = model["name"]
    served = f"{vendor}/{name}-{username}"

    args = LaunchArgs(
        job_name=f"sml_{name}_{username}",
        served_model_name=served,
        account=launcher.account,
        partition=launcher.partition,
        environment="src/swiss_ai_model_launch/assets/envs/sglang.toml",
        framework="sglang",
        framework_args=(
            f"--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/{vendor}/{name} "
            f"--served-model-name {served} "
            "--host 0.0.0.0 "
            "--port 8080 " + model.get("framework_args", "")
        ),
        nodes=model.get("nodes", 1),
        time="02:00:00",
        worker_port=8080,
    )

    job_id, served_name = await launcher.launch_with_args(args)
    return served_name, job_id


async def main() -> None:
    username = getpass.getuser()
    account = grp.getgrgid(os.getgid()).gr_name

    launcher = SlurmLauncher(
        system_name="local",
        username=username,
        account=account,
        partition="normal",
    )

    results = await asyncio.gather(*(launch_model(launcher, m) for m in MODELS))

    for served_name, job_id in results:
        print(f"  {served_name} -> job {job_id}")


if __name__ == "__main__":
    asyncio.run(main())
