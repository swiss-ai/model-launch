#!/bin/bash
sml advanced \
  --tui \
  --partition normal \
  --framework vllm \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B \
    --served-model-name swiss-ai/Apertus-v1.5-8B-thinking-vllm-$(whoami) \
    --chat-template-content-format string \
    --gpu-memory-utilization 0.6 \
    --max-model-len 262144 \
    --reasoning-parser apertus \
    --default-chat-template-kwargs.enable_thinking true"
