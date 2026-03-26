#!/bin/bash
VLLM_ENV=src/swiss_ai_model_launch/assets/envs/vllm.toml

sml advanced \
  --slurm-nodes 1 \
  --serving-framework vllm \
  --slurm-environment "$VLLM_ENV" \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/arcee-ai/Trinity-Mini \
    --served-model-name arcee-ai/Trinity-Mini-$(whoami) \
    --host 0.0.0.0 \
    --port 8080 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser hermes"
