#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name Qwen/Qwen3-8B-$(whoami) \
    --dp-size 4"
