#!/bin/bash
# Note: a model named swiss-ai/Apertus-8B-Instruct-2509 is usually already running.
# The --served-model-name flag avoids name collisions.
sml advanced \
  --slurm-nodes 1 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_rocm.toml \
  --slurm-time "05:00:00" \
  --partition mi300 \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-rocm2 \
    --host 0.0.0.0 \
    --port 8080"
