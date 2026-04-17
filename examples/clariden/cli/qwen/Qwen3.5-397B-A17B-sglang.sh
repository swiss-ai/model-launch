#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 4 \
  --slurm-time 04:00:00 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.5-397B-A17B \
    --host 0.0.0.0 \
    --port 8080 \
    --tp-size 16 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --served-model-name Qwen/Qwen3.5-397B-A17B-$(whoami)" \
  --pre-launch-cmds "pip install nvidia-cudnn-cu12==9.16.0.29"

