#!/bin/bash
VLLM_ENV=src/swiss_ai_model_launch/assets/envs/vllm.toml

sml advanced \
  --slurm-nodes 4 \
  --serving-framework vllm \
  --disable-ocf \
  --worker-port 8080 \
  --slurm-environment "$VLLM_ENV" \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.5-397B-A17B \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 16 \
    --served-model-name Qwen/Qwen3.5-397B-A17B-$(whoami)"
