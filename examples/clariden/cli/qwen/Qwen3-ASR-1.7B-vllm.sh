#!/bin/bash
# Launch Qwen/Qwen3-ASR-1.7B (1.7B multilingual ASR, 52 langs incl.
# 22 Chinese dialects, built on Qwen3-Omni foundation) on one Clariden
# GH200 node with vLLM, DP=4 TP=1 (4 independent replicas, one per GPU).
# Suitable for high-throughput batch / streaming ASR over many audio clips.
#
# Qwen3-ASR uses the Qwen3ASRForConditionalGeneration architecture, which
# is registered in stock vLLM 0.19+ (no vllm-omni needed). The generic
# `vllm.toml` env points at the ci/vllm_cuda13 image (vLLM 0.19.1rc1,
# transformers 5.5.4, torchaudio 2.11) which has the full audio arch set
# and the newer Qwen3ASRConfig schema (with thinker_config). The image
# is missing librosa/audioread (vLLM's audio file loader), so we install
# them at launch via --pre-launch-cmds.
#
# Model weights (downloaded separately):
#   /capstor/store/cscs/swissai/infra01/MLLM/audio_asr/Qwen3-ASR-1.7B/
#
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-time 6:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --pre-launch-cmds "pip install librosa audioread" \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/MLLM/audio_asr/Qwen3-ASR-1.7B \
    --served-model-name Qwen/Qwen3-ASR-1.7B-$(whoami) \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --trust-remote-code"
