#!/bin/bash
sml advanced \
  --tui \
  --partition normal \
  --framework vllm \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model swiss-ai/Apertus-v1.5-70B \
    --served-model-name swiss-ai/Apertus-v1.5-70B \
    --chat-template-content-format string \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --compilation-config.pass_config.fuse_allreduce_rms false"
