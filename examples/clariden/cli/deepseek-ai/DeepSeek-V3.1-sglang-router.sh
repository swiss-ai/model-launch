#!/bin/bash
# 2 replicas x 4 nodes each for increased throughput. Experimental.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-replicas 2 \
  --slurm-nodes-per-replica 4 \
  --use-router \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V3.1 \
    --served-model-name deepseek-ai/DeepSeek-V3.1-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    "
