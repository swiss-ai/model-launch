#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-Next-80B-A3B-Instruct \
    --served-model-name Qwen/Qwen3-Next-80B-A3B-Instruct-$(whoami) \
    --host 0.0.0.0 \
    --tp-size 4"
