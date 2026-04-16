#!/bin/bash
sml advanced \
  --slurm-nodes 4 \
  --serving-framework vllm \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.5-397B-A17B \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 16 \
    --served-model-name Qwen/Qwen3.5-397B-A17B-$(whoami)"
