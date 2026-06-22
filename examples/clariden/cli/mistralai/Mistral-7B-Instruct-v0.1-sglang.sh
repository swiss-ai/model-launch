#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-7B-Instruct-v0.1 \
    --served-model-name mistralai/Mistral-7B-Instruct-v0.1-$(whoami) \
    --host 0.0.0.0 \
    --enable-metrics"
