#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --pre-launch-cmds "pip install librosa audioread" \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/MLLM/audio_asr/Qwen3-ASR-1.7B \
    --served-model-name Qwen/Qwen3-ASR-1.7B-$(whoami) \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --trust-remote-code"
