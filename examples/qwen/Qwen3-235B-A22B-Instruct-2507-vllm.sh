#!/bin/bash
VLLM_ENV=src/swiss_ai_model_launch/assets/envs/vllm.toml

sml advanced \
  --slurm-nodes 2 \
  --serving-framework vllm \
  --worker-port 8080 \
  --disable-ocf \
  --slurm-environment "$VLLM_ENV" \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-$(whoami) \
    --tensor-parallel-size 8"
