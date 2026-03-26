#!/bin/bash
SGLANG_KIMI_ENV=src/swiss_ai_model_launch/assets/envs/sglang_kimi.toml

sml advanced \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment "$SGLANG_KIMI_ENV" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2.5 \
    --served-model-name moonshotai/Kimi-K2.5-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2"
