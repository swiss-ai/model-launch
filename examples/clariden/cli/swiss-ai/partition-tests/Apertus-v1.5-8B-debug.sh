#!/bin/bash
# Partition smoke test: `debug` caps at 4 nodes / 90 node-minutes per job, and
# enforces a strict per-user limit (max 1 running + 2 submitted jobs) cluster-wide.
# Not for production workloads -- short debugging/testing only.
sml advanced \
  --no-tui \
  --system clariden \
  --partition debug \
  --time 00:20:00 \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model swiss-ai/Apertus-v1.5-8B \
    --served-model-name swiss-ai/Apertus-v1.5-8B \
    --chat-template-content-format string \
    --gpu-memory-utilization 0.6 \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus"
