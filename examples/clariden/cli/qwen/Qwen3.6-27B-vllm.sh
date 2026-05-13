#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.6-27B \
    --host 0.0.0.0 \
    --served-model-name Qwen/Qwen3.6-27B-$(whoami) \
    --data-parallel-size 2 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --reasoning-parser qwen3"
