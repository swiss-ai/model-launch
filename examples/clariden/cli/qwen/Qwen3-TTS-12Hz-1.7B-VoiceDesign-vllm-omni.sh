#!/bin/bash
# Launch Qwen3-TTS 12Hz 1.7B VoiceDesign with vLLM on one Clariden GH200 node.
#
# Model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 6:00:00 \
  --serving-framework vllm \
  --served-model-name Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_cuda13_v2.toml \
  --framework-args "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --host 0.0.0.0 \
    --port 8080 \
    --deploy-config /opt/conda/lib/python3.12/site-packages/vllm_omni/deploy/qwen3_tts.yaml \
    --omni \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.40" \
  --no-tui
