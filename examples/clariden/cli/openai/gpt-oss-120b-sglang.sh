#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name openai/gpt-oss-120b-$(whoami) \
    --tp-size 4 \
    --ep-size 4 \
    --enable-metrics"