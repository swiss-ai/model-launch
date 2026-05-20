#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 2 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --host 0.0.0.0 \
    --tp-size 8 \
    --served-model-name mistralai/Mixtral-8x22B-Instruct-v0.1-$(whoami) \
    --enable-metrics"
