#!/bin/bash
# Muse-Glimmer-30B: 29.6B dense vision-language model, 128K context.
#
# Weights are already staged on capstor. Tensor parallelism is 4 rather than
# the recipe's single-GPU DGX Spark variant: the bf16 weights are 56 GB and a
# GH200 exposes 95.6 GiB, so TP=1 would fit but leave little room for KV cache
# at this context length. The model has 2 KV heads, which vLLM replicates
# across 4 ranks.
sml advanced \
  --tui \
  --system clariden \
  --partition normal \
  --framework vllm \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_muse_glimmer.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/meta-models/Muse-Glimmer-30B \
    --served-model-name meta-models/Muse-Glimmer-30B \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --enable-auto-tool-choice \
    --tool-call-parser muse_glimmer \
    --reasoning-parser muse_glimmer \
    --generation-config auto \
    --host 0.0.0.0"
