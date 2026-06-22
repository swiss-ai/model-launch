#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --nodes-per-replica 4 \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-Large-3-675B-Instruct-2512 \
    --host 0.0.0.0 \
    --served-model-name mistralai/Mistral-Large-3-675B-Instruct-2512-$(whoami) \
    --tensor-parallel-size 16"
