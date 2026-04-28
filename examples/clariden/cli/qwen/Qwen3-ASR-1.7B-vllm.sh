#!/bin/bash
# Launch Qwen/Qwen3-ASR-1.7B (1.7B multilingual ASR + 22 Chinese-dialect
# coverage built on Qwen3-Omni foundation) on one Clariden GH200 node with
# vLLM, DP=4 TP=1 (4 independent replicas, one per GPU). Suitable for
# high-throughput batch / streaming ASR over many audio clips.
#
# Qwen3-ASR uses the Qwen3ASRForConditionalGeneration architecture, which
# is registered in stock vLLM 0.18.2+ (no vllm-omni needed). The
# `apertus-vllm-13.0-prod` image (vLLM 0.19.1.dev) carries the registration.
#
# The accompanying `vllm_apertus_1.5.toml` env points to that image and
# keeps the standard Clariden NCCL / OFI tuning that the Apertus serving
# tree already validated.
#
# Model weights (downloaded separately):
#   /capstor/store/cscs/swissai/infra01/MLLM/audio_asr/Qwen3-ASR-1.7B/
#
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 6:00:00 \
  --serving-framework vllm \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/MLLM/audio_asr/Qwen3-ASR-1.7B \
    --served-model-name Qwen/Qwen3-ASR-1.7B-$(whoami) \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --trust-remote-code"
