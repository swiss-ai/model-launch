#!/bin/bash
# Partition smoke test: `low` requires --qos slowdown (its own default QOS,
# distinct from the account's normal association QOS) -- for projects that
# have exhausted their normal compute credits.
sml advanced \
  --no-tui \
  --system clariden \
  --partition low \
  --qos slowdown \
  --time 00:30:00 \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model swiss-ai/Apertus-v1.5-8B \
    --served-model-name swiss-ai/Apertus-v1.5-8B \
    --chat-template-content-format string \
    --gpu-memory-utilization 0.6 \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus"
