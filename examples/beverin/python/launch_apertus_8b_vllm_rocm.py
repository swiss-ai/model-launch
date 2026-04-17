#!/usr/bin/env python3
"""Launch Apertus-8B on beverin (ROCm/MI300) with vLLM using the SML Python API."""

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
        partition="mi300",
    )

    args = LaunchArgs(
        job_name=f"sml_apertus_8b_vllm_rocm_{username}",
        served_model_name=f"swiss-ai/Apertus-8B-Instruct-2509-vllm-rocm-{username}",
        account=account,
        partition="mi300",
        environment="src/swiss_ai_model_launch/assets/envs/vllm_rocm.toml",
        framework="vllm",
        framework_args=(
            "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 "
            f"--served-model-name swiss-ai/Apertus-8B-Instruct-2509-vllm-rocm-{username} "
            "--host 0.0.0.0 "
            "--port 8080 "
            "--tensor-parallel-size 4 "
            "--gpu-memory-utilization 0.5"
        ),
        time="05:00:00",
        worker_port=8080,
    )

    job_id, served = await launcher.launch_with_args(args)
    print(f"Job submitted: {job_id}")
    print(f"Served model name: {served}")


if __name__ == "__main__":
    asyncio.run(main())
