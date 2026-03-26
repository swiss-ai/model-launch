#!/bin/bash
SGLANG_ENV=src/swiss_ai_model_launch/assets/envs/sglang.toml

sml advanced \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment "$SGLANG_ENV" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-Small-24B-Instruct-2501 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name mistralai/Mistral-Small-24B-Instruct-2501-$(whoami) \
    --dp-size 4"
