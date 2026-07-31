#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --nodes-per-replica 4 \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang_kimi.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2.5 \
    --served-model-name moonshotai/Kimi-K2.5-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-metrics"
