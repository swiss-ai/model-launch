#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 12:00:00 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name openai/gpt-oss-20b-$(whoami) \
    --dp-size 4 \
    --enable-metrics"
