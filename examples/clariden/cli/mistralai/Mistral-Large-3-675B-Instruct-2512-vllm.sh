#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 4 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-Large-3-675B-Instruct-2512 \
    --host 0.0.0.0 \
    --served-model-name mistralai/Mistral-Large-3-675B-Instruct-2512-$(whoami) \
    --tensor-parallel-size 16"
