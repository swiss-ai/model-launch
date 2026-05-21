#!/bin/bash
# 1.6T params, fp8 block-quantised. 8 nodes x TP32 (32 GH200s).
# Requires sglang with DeepseekV4ForCausalLM support and transformers >= 4.57.1.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 8 \
  --slurm-time 3:00:00 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Pro-Base \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro-Base-$(whoami) \
    --tp-size 32 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --enable-metrics"
