#!/usr/bin/env python3
"""
Download a HuggingFace model snapshot.

Run with:
    uv-arm64 run download_model.py --model deepseek-ai/DeepSeek-V3.1
"""

# /// script
# dependencies = ["huggingface_hub"]
# ///

import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model snapshot.")
    parser.add_argument("--model", required=True, help="HuggingFace repo id, e.g. deepseek-ai/DeepSeek-V3.1")

    args = parser.parse_args()

    model = args.model
    target_dir = f"/capstor/store/cscs/swissai/infra01/hf_models/models/{model}"

    print(f"Downloading {model} to: {target_dir}")

    snapshot_download(repo_id=model, local_dir=target_dir, local_dir_use_symlinks=False, max_workers=1)


if __name__ == "__main__":
    main()
