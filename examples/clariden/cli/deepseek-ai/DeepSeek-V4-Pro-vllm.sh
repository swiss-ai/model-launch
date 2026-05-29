#!/bin/bash
# Post-trained DeepSeek-V4-Pro, MXFP4-quantised (~806 GB on disk).
# 4 nodes x TP16 mirrors V3.1 density. Served with vllm >= 0.21.0
# (ships DeepseekV4ForCausalLM) + transformers >= 5.9.0 (registers the
# `deepseek_v4` model_type), both baked into the vllm_deepseek_v4 image.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 4 \
  --serving-framework vllm \
  --slurm-time 3:00:00 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_deepseek_v4.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Pro \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro-$(whoami) \
    --tensor-parallel-size 16 \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --trust-remote-code"
