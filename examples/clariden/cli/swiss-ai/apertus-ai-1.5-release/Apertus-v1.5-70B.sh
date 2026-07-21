#!/bin/bash
sml advanced \
  --tui \
  --partition normal \
  --framework vllm \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model apertus-ai/Apertus-v1.5-70B \
    --served-model-name apertus-ai/Apertus-v1.5-70B \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --default-chat-template-kwargs.enable_thinking false \
    --compilation-config.pass_config.fuse_allreduce_rms false"
