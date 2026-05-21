#!/bin/bash
# Post-trained DeepSeek-V4-Pro, MXFP4-quantised (~806 GB on disk).
# 4 nodes x TP16 mirrors V3.1 density. Requires sglang with MXFP4 dequant
# kernels and DeepseekV4ForCausalLM support; transformers >= 4.57.1.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 4 \
  --serving-framework sglang \
  --slurm-time 3:00:00 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Pro \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --enable-metrics"
