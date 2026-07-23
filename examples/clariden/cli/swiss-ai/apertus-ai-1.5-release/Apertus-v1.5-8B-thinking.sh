#!/bin/bash
sml advanced \
  --tui \
  --partition normal \
  --framework vllm \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model apertus-ai/Apertus-v1.5-8B \
    --served-model-name apertus-ai/Apertus-v1.5-8B-thinking \
    --chat-template-content-format string \
    --gpu-memory-utilization 0.6 \
    --max-model-len 262144 \
    --reasoning-parser apertus \
    --default-chat-template-kwargs.enable_thinking true"
