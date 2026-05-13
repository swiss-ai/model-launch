#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-time 12:00:00 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --host 0.0.0.0 \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-$(whoami) \
    --tp-size 4 \
    --reasoning-parser deepseek-r1"
