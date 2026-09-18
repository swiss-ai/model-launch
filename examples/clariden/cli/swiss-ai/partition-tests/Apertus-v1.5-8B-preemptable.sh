#!/bin/bash
# Partition smoke test: confirms sbatch accepts the job on `preemptable` (no --qos needed).
# Jobs here may be preempted by highprio work.
sml advanced \
  --no-tui \
  --system clariden \
  --partition preemptable \
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
