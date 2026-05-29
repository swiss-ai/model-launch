#!/bin/bash
# 1.6T params, fp8 block-quantised. 8 nodes x TP32 (32 GH200s).
# Served with vllm >= 0.21.0 (ships DeepseekV4ForCausalLM) + transformers >= 5.9.0
# (registers the `deepseek_v4` model_type), both baked into the vllm_deepseek_v4 image.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 8 \
  --slurm-time 3:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_deepseek_v4.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Pro-Base \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro-Base-$(whoami) \
    --tensor-parallel-size 32 \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --trust-remote-code"
