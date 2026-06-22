#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/utter-project/EuroLLM-9B-Instruct-2512 \
    --served-model-name utter-project/EuroLLM-9B-Instruct-2512-$(whoami) \
    --dp-size 4 \
    --host 0.0.0.0 \
    --enable-metrics"
