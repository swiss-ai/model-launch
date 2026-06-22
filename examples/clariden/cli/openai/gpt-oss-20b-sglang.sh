#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --served-model-name openai/gpt-oss-20b-$(whoami) \
    --dp-size 4 \
    --enable-metrics"
