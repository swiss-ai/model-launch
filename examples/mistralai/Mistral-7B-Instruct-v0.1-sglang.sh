#!/bin/bash
SGLANG_ENV=src/swiss_ai_model_launch/assets/envs/sglang.toml

sml advanced \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --slurm-environment "$SGLANG_ENV" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-7B-Instruct-v0.1 \
    --served-model-name mistralai/Mistral-7B-Instruct-v0.1-$(whoami) \
    --host 0.0.0.0 \
    --port 8080"
