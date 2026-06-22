#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --served-model-name Qwen/Qwen3-8B-$(whoami) \
    --dp-size 4 \
    --enable-metrics"
